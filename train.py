import argparse  # Added for dynamic runtime configuration
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import numpy as np
from glob import glob
from accelerate import Accelerator
from utils import PCDataset, EMDLoss, fscore, pytorch_chamfer_distance, SonarPhysicsLoss
import open3d as o3d
from tqdm import tqdm
import os

from model import PointCloudNet

if __name__ == "__main__":
    # ==============================================================================
    # CONFIGURATION & TRANSFER LEARNING ARGUMENTS
    # ==============================================================================
    parser = argparse.ArgumentParser(description="RGB2Point Sonar Training Pipeline")
    
    # Existing Arguments
    parser.add_argument("--transfer_learning", action="store_true", help="Enable transfer learning from an existing checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the initial .pth checkpoint file")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the pretrained ViT backbone weights")
    
    # NEW: Automated Sweep Arguments
    parser.add_argument("--experiment", type=str, required=True, help="Name of the dataset/experiment (e.g., original, blur)")
    parser.add_argument("--data_root", type=str, default="data_ready", help="Root folder containing the datasets")
    parser.add_argument("--comment", type=str, default="sweep", help="Tag to append to saved model names")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    
    args = parser.parse_args()

    accelerator = Accelerator(log_with="wandb")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Use the dynamic arguments instead of hardcoded variables
    batch_size = args.batch_size
    num_epochs = args.epochs
    device = accelerator.device

    CUSTOM_PC_SIZE = 1024 
    EXPERIMENT = args.experiment
    COMENT = args.comment
    
    # Dynamically point to the correct subfolder in data_ready
    DATASET = f"{args.data_root}/{EXPERIMENT}"
    
    model_save_name = f"best_model_{EXPERIMENT}-{COMENT}.pth"
    last_model_save_name = f"last_model_{EXPERIMENT}-{COMENT}.pth"

    # 1. Initialize base model architecture
    model = PointCloudNet(
        num_views=1, point_cloud_size=CUSTOM_PC_SIZE, num_heads=4, dim_feedforward=2048
    )
    
    # 2. Apply Transfer Learning Weights (If Toggled)
    if args.transfer_learning:
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            accelerator.print(f" --> Loading weights for Transfer Learning from: {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                accelerator.print(f"     Note: Missing keys ignored (Expected for new output dimensions): {len(missing_keys)}")
        else:
            accelerator.print(f" [ERROR] --transfer_learning was set, but --checkpoint_path is invalid or empty!")
            exit(1)

    # 3. Handle Parameter Freezing Strategy
    if args.freeze_backbone:
        accelerator.print(" --> Freezing Vision Transformer backbone. Training final layers only.")
        for param in model.vit.parameters():
            param.requires_grad = False
    else:
        accelerator.print(" --> Unfreezing full network. Optimization will adjust all weights.")
        for param in model.vit.parameters():
            param.requires_grad = True

    # Filter optimizer to only track parameters requiring gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=5e-4)
    
    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=5,
        min_lr=1e-5,
        threshold=0.01,
    )
    
    # Track the experiment name dynamically in wandb
    
    accelerator.init_trackers(
        project_name="wacv_pc1024", 
        config={
            "experiment": EXPERIMENT,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "transfer_learning": args.transfer_learning
        },
        init_kwargs={
            "wandb": {
                "name": f"{EXPERIMENT}-{COMENT}"
            }
        }
    )

    # The rest of your DataLoader and Training loop remains exactly the same below this...
    dataset = PCDataset(stage=f"{DATASET}/train", transform=transform)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12, pin_memory=True
    )
    
    test_dataset = PCDataset(stage=f"{DATASET}/test", transform=transform)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    
    model, optimizer, dataloader, test_dataloader, sche = accelerator.prepare(
        model, optimizer, dataloader, test_dataloader, sche
    )

    best = 10000

    physics_criterion = SonarPhysicsLoss(k_neighbors=8, lambda_smooth=1.0, lambda_incidence=0.5).to(device)

for epoch in range(num_epochs):
        # ==============================================================================
        # TRAINING LOOP
        # ==============================================================================
        model.train()
        train_loss_history = []
        
        train_pbar = tqdm(dataloader, desc=f"[Train Step | Epoch {epoch+1}]")
        
        # UPDATE: Unpack the centroid here
        for idx, (images, gt_pc, centroids, poses, name) in enumerate(train_pbar):
            gt_pc = gt_pc.float().to(device)
            images = images.to(device)
            centroids = centroids.float().to(device)
            poses = poses.float().to(device) # Move pose to GPU
            
            optimizer.zero_grad()
            
            # Pass both images and poses to the network
            out = model(images, poses)
            
            # 1. Data Loss (Chamfer)
            cd_loss = pytorch_chamfer_distance(out, gt_pc) #* 5.0 originalmente existe essa multiplicao porem a loss CD estava com uma varicao muito alta
            
            # 2. Physics-Informed Loss
            pinn_loss = physics_criterion(out, centroids)
            
            # Total Loss
            loss = cd_loss + pinn_loss
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()
            train_loss_history.append(loss.item())
            
            train_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "CD": f"{cd_loss.item():.4f}", 
                "PINN": f"{pinn_loss.item():.4f}"
            })

        mean_train_loss = np.mean(train_loss_history)
        accelerator.print(f"[Train] Epoch {epoch + 1}, Loss: {mean_train_loss:.4f}")
        accelerator.log({"train/loss": mean_train_loss, "train/epoch": epoch + 1})

       # ==============================================================================
        # TESTING / VALIDATION LOOP
        # ==============================================================================
        model.eval()
        test_loss_history = []
        test_cd_history = []
        test_pinn_history = []  # Added to track validation physics

        test_pbar = tqdm(test_dataloader, desc=f"[Test Step | Epoch {epoch+1}]")
        
        # UPDATE 1: Unpack 'centroids' alongside the other variables
        for idx, (images, gt_pc, centroids, poses, names) in enumerate(test_pbar):
            gt_pc = gt_pc.float().to(device)
            images = images.to(device)
            centroids = centroids.float().to(device)
            poses = poses.float().to(device) # Move pose to GPU
            
            with torch.no_grad():
                # Pass both images and poses to the network
                out = model(images, poses)

                # 1. Data Loss (Chamfer)
                cd_loss = pytorch_chamfer_distance(out, gt_pc) #* 5.0
                
                # 2. Physics Loss (To monitor validation compliance)
                pinn_loss = physics_criterion(out, centroids)
                
                # Total Validation Loss
                total_test_loss = cd_loss + pinn_loss

            test_loss_history.append(total_test_loss.item())
            test_cd_history.append(cd_loss.item())
            test_pinn_history.append(pinn_loss.item())
            
            test_pbar.set_postfix({
                "Loss": f"{total_test_loss.item():.4f}", 
                "CD": f"{cd_loss.item():.4f}", 
                "PINN": f"{pinn_loss.item():.4f}"
            })

        # Calculate epoch averages
        mean_test_loss = np.mean(test_loss_history)
        mean_cd = np.mean(test_cd_history)
        mean_pinn = np.mean(test_pinn_history)
        
        accelerator.print(f"[Test] Epoch {epoch + 1}, Total: {mean_test_loss:.4f}, CD: {mean_cd:.4f}, PINN: {mean_pinn:.4f}")
        
        # UPDATE 3: Log the split metrics so you can track them on your Weights & Biases dashboard
        accelerator.log({
            "test/total_loss": mean_test_loss, 
            "test/chamfer_dist": mean_cd, 
            "test/pinn_loss": mean_pinn,
            "test/epoch": epoch + 1
        })
        
        sche.step(mean_test_loss)
        
        if isinstance(model, nn.DataParallel) or hasattr(model, 'module'):
            current_state = {"model": model.module.state_dict()}
        else:
            current_state = {"model": model.state_dict()}
            
        torch.save(current_state, last_model_save_name)
        
        if mean_test_loss < best:
            best = mean_test_loss
            torch.save(current_state, model_save_name)
            accelerator.print(f" --> Saved new best model checkpoint to: {model_save_name}")