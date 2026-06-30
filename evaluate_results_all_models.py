import os
import glob
import random
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import timm

# Import your CURRENT cross-modal model
from model import PointCloudNet, PointCloudGeneratorWithAttention

# ==============================================================================
# CONFIGURATION & CHECKPOINT PATHS
# ==============================================================================
EXPERIMENT_NAME = "all_combined" 
TEST_DIR = f"data/{EXPERIMENT_NAME}/train"

# IMPORTANT: Update these 3 paths to point to your specific saved models
WEIGHTS_BASELINE = f"/home/guilherme/git/RGB2point/weigths/best_model_Didson-blur-Transfer-DDO.pth"  # Standard RGB2POINT
WEIGHTS_PINN = f"/home/guilherme/git/RGB2point/weights-PINN/best_model_Didson-original-PINN-TRANSFER-100-epochs.pth"                # PINN Loss, NO Pose
WEIGHTS_CROSSMODAL = f"/home/guilherme/git/RGB2point/weigths-PINN-CROSMODAL-POSE/best_model_all_combined-PINN-CROSMODAL-POSE.pth"     # PINN Loss WITH Pose

# Model parameters
CUSTOM_PC_SIZE = 1024  
INPUT_FEATURE_DIM = 5120
DIM_FEEDFORWARD = 2048
POSE_DIM = 16

# Target images (Leave empty [] to grab random samples)
TARGET_IMAGES = [
    "blur_Didson-1-Sphere-5",
    "denoise_Didson-1-Sphere-5",
    "original_Didson-1-Sphere-5",
    "blur_Didson-1-Sphere-5",
    "denoise_Didson-2-Sphere-5",
    "original_Didson-2-Sphere-5"
    "blur_Didson-3-Sphere-5",
    "denoise_Didson-3-Sphere-5",
    "original_Didson-3-Sphere-5",
    "blur_Didson-4-Sphere-5",
    "denoise_Didson-4-Sphere-5",
    "original_Didson-4-Sphere-5",
    "original_Didson-1-Tripode-5",
    "blur_Didson-1-Tripode-5",
    "denoise_Didson-1-Tripode-5",
    "original_Didson-2-Tripode-5",
    "blur_Didson-2-Tripode-5",
    "denoise_Didson-2-Tripode-5",
    "original_Didson-3-Tripode-5",
    "blur_Didson-3-Tripode-5",
    "denoise_Didson-3-Tripode-5",
    "original_Didson-4-Tripode-5",
    "blur_Didson-4-Tripode-5",
    "denoise_Didson-4-Tripode-5",
]
NUM_RANDOM_SAMPLES = 5 

# ==============================================================================
# LEGACY ARCHITECTURE (For fair evaluation of older checkpoints)
# ==============================================================================
class LegacyPointCloudNet(nn.Module):
    """The original architecture before Cross-Modal Pose Fusion was added."""
    def __init__(self, num_views, point_cloud_size, num_heads, dim_feedforward):
        super(LegacyPointCloudNet, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        num_features = self.vit.num_features
        out_features = 1024 * 4
        self.aggregator = nn.Linear(num_features, out_features)
        
        # Generator expects ONLY image features (no pose embedding added)
        self.point_cloud_generator = PointCloudGeneratorWithAttention(
            input_feature_dim=out_features, 
            point_cloud_size=point_cloud_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, x):
        batch_size, num_views, C, H, W = x.shape
        x = x.view(batch_size * num_views, C, H, W)
        features = self.vit(x)
        features = features.view(batch_size, num_views, -1)
        mean_features = torch.mean(features, dim=1)
        
        aggregated_features = self.aggregator(mean_features)
        aggregated_features = aggregated_features.unsqueeze(1) # No pose concat here
        
        point_cloud = self.point_cloud_generator(aggregated_features)
        point_cloud = point_cloud.view(batch_size, -1, 3)
        return point_cloud

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def load_test_samples(test_dir, target_names):
    image_dir = os.path.join(test_dir, "images")
    pc_dir = os.path.join(test_dir, "pointclouds")
    pose_dir = os.path.join(test_dir, "poses")
    samples = []
    
    if len(target_names) > 0:
        selected_images = [os.path.join(image_dir, f"{name}.png") for name in target_names]
    else:
        all_images = glob.glob(os.path.join(image_dir, "*.png"))
        selected_images = random.sample(all_images, min(NUM_RANDOM_SAMPLES, len(all_images)))
    
    for img_path in selected_images:
        if not os.path.exists(img_path): continue
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        samples.append({
            "name": base_name,
            "img_path": img_path,
            "pc_path": os.path.join(pc_dir, f"{base_name}.npy"),
            "pose_path": os.path.join(pose_dir, f"{base_name}.npy")
        })
    return samples

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def load_weights(model, path, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return True
    return False

# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
def evaluate_comparison():
    print(f"--- Running 3-Way Model Comparison on {EXPERIMENT_NAME} dataset ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize all three models
    print("Initializing architectures...")
    model_baseline = LegacyPointCloudNet(1, CUSTOM_PC_SIZE, 4, DIM_FEEDFORWARD).to(device)
    model_pinn = LegacyPointCloudNet(1, CUSTOM_PC_SIZE, 4, DIM_FEEDFORWARD).to(device)
    model_cross = PointCloudNet(1, CUSTOM_PC_SIZE, 4, DIM_FEEDFORWARD, POSE_DIM).to(device)

    # 2. Load Weights
    print("Loading checkpoints...")
    if not load_weights(model_baseline, WEIGHTS_BASELINE, device): print(f"[WARN] Missing {WEIGHTS_BASELINE}")
    if not load_weights(model_pinn, WEIGHTS_PINN, device): print(f"[WARN] Missing {WEIGHTS_PINN}")
    if not load_weights(model_cross, WEIGHTS_CROSSMODAL, device): print(f"[WARN] Missing {WEIGHTS_CROSSMODAL}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    samples = load_test_samples(TEST_DIR, TARGET_IMAGES)
    output_folder = f"comparison_results_{EXPERIMENT_NAME}"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Running inference and saving 1x5 comparison grids to './{output_folder}/'...")
    
    with torch.no_grad():
        for sample in samples:
            # Data Prep
            raw_img = Image.open(sample["img_path"]).convert("RGB")
            img_tensor = transform(raw_img).unsqueeze(0).unsqueeze(0).to(device)
            
            gt_pc = np.load(sample["pc_path"])
            gt_pc = gt_pc - np.mean(gt_pc, axis=0) 
            
            if os.path.exists(sample["pose_path"]):
                pose = np.load(sample["pose_path"]).flatten()
            else:
                pose = np.eye(4, dtype=np.float32).flatten()
            pose_tensor = torch.as_tensor(pose, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Inference
            pred_base = model_baseline(img_tensor).squeeze(0).cpu().numpy()
            pred_pinn = model_pinn(img_tensor).squeeze(0).cpu().numpy()
            pred_cross = model_cross(img_tensor, pose_tensor).squeeze(0).cpu().numpy()

            # Plotting a 1x5 Grid
            fig = plt.figure(figsize=(25, 5))
            fig.suptitle(f"Ablation Study: {sample['name']}", fontsize=18)
            
            # 1. Input Image
            ax1 = fig.add_subplot(1, 5, 1)
            ax1.imshow(raw_img); ax1.set_title("Input Sonar Image"); ax1.axis("off")
            
            # 2. Ground Truth
            ax2 = fig.add_subplot(1, 5, 2, projection='3d')
            ax2.scatter(gt_pc[:,0], gt_pc[:,1], gt_pc[:,2], c='green', s=2)
            ax2.set_title("Ground Truth PC"); set_axes_equal(ax2)
            
            # 3. Baseline
            ax3 = fig.add_subplot(1, 5, 3, projection='3d')
            ax3.scatter(pred_base[:,0], pred_base[:,1], pred_base[:,2], c='gray', s=2)
            ax3.set_title("Baseline (RGB2POINT)"); set_axes_equal(ax3)
            
            # 4. PINN Only
            ax4 = fig.add_subplot(1, 5, 4, projection='3d')
            ax4.scatter(pred_pinn[:,0], pred_pinn[:,1], pred_pinn[:,2], c='orange', s=2)
            ax4.set_title("PINN (Lambertian Loss)"); set_axes_equal(ax4)

            # 5. PINN + Pose
            ax5 = fig.add_subplot(1, 5, 5, projection='3d')
            ax5.scatter(pred_cross[:,0], pred_cross[:,1], pred_cross[:,2], c='red', s=2)
            ax5.set_title("Cross-Modal PINN (Pose Fusion)"); set_axes_equal(ax5)

            plt.tight_layout()
            save_path = os.path.join(output_folder, f"compare_{sample['name']}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig) 
            print(f"  -> Saved Comparison: {save_path}")

if __name__ == "__main__":
    evaluate_comparison()