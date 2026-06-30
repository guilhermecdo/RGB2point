import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Import your model (Make sure model.py is in the same directory)
from model import PointCloudNet

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_NAME = "all_combined" 
TEST_DIR = f"data/all_combined/train"
#MODEL_WEIGHTS = f"best_model_all_combined-PINN-CROSMODAL-POSE.pth" 
#MODEL_WEIGHTS = f"/home/guilherme/git/RGB2point/weigths-PINN-CROSMODAL-POSE/best_model_all_combined-PINN-CROSMODAL-POSE.pth" 
MODEL_WEIGHTS = f"/home/guilherme/git/RGB2point/weights-PINN/best_model_Didson-Denoise-original-PINN.pth"

# Model parameters (Must match what you used in training!)
CUSTOM_PC_SIZE = 1024  
INPUT_FEATURE_DIM = 5120
DIM_FEEDFORWARD = 2048
POSE_DIM = 16

# ==============================================================================
# TARGET IMAGES TO EVALUATE
# ==============================================================================
# Paste the exact names (without .png) of the images you want to test here.
# If you leave this list empty [], it will grab random samples instead.
TARGET_IMAGES = [
    "blur_Didson-2-Sphere-5",
    "denoise_Didson-2-Sphere-5",
    "original_Didson-2-Sphere-5"
    
]

# Number of random samples to grab IF the TARGET_IMAGES list is empty
NUM_RANDOM_SAMPLES = 5 

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def load_test_samples(test_dir, target_names):
    """Loads specific samples if provided, otherwise grabs random ones."""
    image_dir = os.path.join(test_dir, "images")
    pc_dir = os.path.join(test_dir, "pointclouds")
    pose_dir = os.path.join(test_dir, "poses")
    
    samples = []
    
    # Mode 1: Use the specific list provided by the user
    if len(target_names) > 0:
        print(f"Loading {len(target_names)} specific images from target list...")
        selected_images = [os.path.join(image_dir, f"{name}.png") for name in target_names]
    
    # Mode 2: Fallback to random sampling if the list is empty
    else:
        print(f"Target list is empty. Grabbing {NUM_RANDOM_SAMPLES} random images...")
        all_images = glob.glob(os.path.join(image_dir, "*.png"))
        selected_images = random.sample(all_images, min(NUM_RANDOM_SAMPLES, len(all_images)))
    
    for img_path in selected_images:
        if not os.path.exists(img_path):
            print(f"  [WARNING] Could not find image: {img_path}")
            continue
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        pc_path = os.path.join(pc_dir, f"{base_name}.npy")
        pose_path = os.path.join(pose_dir, f"{base_name}.npy")
        
        samples.append({
            "name": base_name,
            "img_path": img_path,
            "pc_path": pc_path,
            "pose_path": pose_path
        })
        
    return samples

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that objects retain their true shape."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def evaluate_and_save():
    print(f"--- Evaluating Model on {EXPERIMENT_NAME} dataset ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model and Load Weights
    model = PointCloudNet(
        num_views=1, 
        point_cloud_size=CUSTOM_PC_SIZE, 
        num_heads=4, 
        dim_feedforward=DIM_FEEDFORWARD,
        pose_dim=POSE_DIM
    ).to(device)
    
    if os.path.exists(MODEL_WEIGHTS):
        print(f"Loading weights from {MODEL_WEIGHTS}...")
        checkpoint = torch.load(MODEL_WEIGHTS, map_location=device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print(f"[ERROR] Could not find {MODEL_WEIGHTS}. Please check the path.")
        return
        
    model.eval()

    # 2. Setup Image Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 3. Load Samples
    samples = load_test_samples(TEST_DIR, TARGET_IMAGES)
    
    # Create an output directory for the results
    output_folder = f"evaluation_results_{EXPERIMENT_NAME}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Running inference and saving plots to './{output_folder}/' ...")
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            # A. Load and transform Image (using the double unsqueeze fix for 5D tensor)
            raw_img = Image.open(sample["img_path"]).convert("RGB")
            img_tensor = transform(raw_img).unsqueeze(0).unsqueeze(0).to(device)
            
            # B. Load GT Point Cloud and Center it
            gt_pc = np.load(sample["pc_path"])
            gt_pc = gt_pc - np.mean(gt_pc, axis=0) 
            
            # C. Load Pose Matrix
            if os.path.exists(sample["pose_path"]):
                pose = np.load(sample["pose_path"]).flatten()
            else:
                pose = np.eye(4, dtype=np.float32).flatten()
            pose_tensor = torch.as_tensor(pose, dtype=torch.float32).unsqueeze(0).to(device)
            
            # D. INFERENCE: Predict Point Cloud
            predicted_pc = model(img_tensor, pose_tensor)
            predicted_pc = predicted_pc.squeeze(0).cpu().numpy()

            # E. PLOTTING (Create a fresh 1x3 figure for this specific sample)
            fig = plt.figure(figsize=(16, 5))
            fig.suptitle(f"Sample: {sample['name']}", fontsize=16)
            
            # Subplot 1: Input Image
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(raw_img)
            ax1.set_title("Input Sonar Image")
            ax1.axis("off")
            
            # Subplot 2: Ground Truth 3D
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            ax2.scatter(gt_pc[:, 0], gt_pc[:, 1], gt_pc[:, 2], c='g', s=2, marker='o')
            ax2.set_title("Ground Truth PC")
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            set_axes_equal(ax2)
            
            # Subplot 3: Predicted 3D
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            ax3.scatter(predicted_pc[:, 0], predicted_pc[:, 1], predicted_pc[:, 2], c='r', s=2, marker='o')
            ax3.set_title("Predicted PC")
            ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
            set_axes_equal(ax3)

            # F. Save and Clear Memory
            plt.tight_layout()
            save_path = os.path.join(output_folder, f"eval_{sample['name']}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig) # Extremely important: prevents memory leak when looping
            
            print(f"  -> Saved: {save_path}")

    print(f"\n✅ Evaluation complete! Processed {len(samples)} images.")

if __name__ == "__main__":
    evaluate_and_save()