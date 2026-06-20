import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Dynamically get the absolute path of the folder where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lock the Root and Output paths to the script's exact location
RAW_ROOT = os.path.join(SCRIPT_DIR, "SEE") 
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "data_ready") 

# Train/Test Split Ratio
TEST_SIZE = 0.2

# Define your categories and the folder names they correspond to
IMAGE_CATEGORIES = {
    "original": "sonar-images",
    "blur": "sonar-images-blur",
    "gauss_blur": "sonar-images-gauss-blur",
    "denoise": "sonar-images-denoise"
}

PCL_DIR = "normalized-point-clouds"
POSE_DIR = "poses"

# Expected extensions 
IMG_EXT = ".png" 
PCL_EXT = ".npy" 
POSE_EXT = ".npy" 

def create_dir_structure(base_path):
    """Creates the train/test subfolders for images, pointclouds, and poses."""
    # Create the root base_path first to prevent intermediate missing folder errors
    os.makedirs(base_path, exist_ok=True)
    
    for stage in ["train", "test"]:
        os.makedirs(os.path.join(base_path, stage, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_path, stage, "pointclouds"), exist_ok=True)
        os.makedirs(os.path.join(base_path, stage, "poses"), exist_ok=True)

def gather_valid_triplets(image_folder_path):
    """Finds all images that have a perfectly matching point cloud and pose."""
    valid_triplets = []
    
    # Find all images in this category
    image_files = glob.glob(os.path.join(image_folder_path, f"*{IMG_EXT}"))
    
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        pcl_path = os.path.join(RAW_ROOT, PCL_DIR, f"{base_name}{PCL_EXT}")
        pose_path = os.path.join(RAW_ROOT, POSE_DIR, f"{base_name}{POSE_EXT}")
        
        # Only keep it if all 3 files exist
        if os.path.exists(pcl_path) and os.path.exists(pose_path):
            valid_triplets.append({
                "name": base_name,
                "img": img_path,
                "pcl": pcl_path,
                "pose": pose_path
            })
            
    return valid_triplets

def copy_triplets(triplets, stage, dataset_path, prefix=""):
    """Copies the triplets into the target dataset directory."""
    for item in triplets:
        # If we are combining all data, we use a prefix to avoid name collisions
        save_name = f"{prefix}{item['name']}"
        
        # Define destinations
        dest_img = os.path.join(dataset_path, stage, "images", f"{save_name}{IMG_EXT}")
        dest_pcl = os.path.join(dataset_path, stage, "pointclouds", f"{save_name}{PCL_EXT}")
        dest_pose = os.path.join(dataset_path, stage, "poses", f"{save_name}{POSE_EXT}")
        
        # Copy files
        shutil.copy2(item['img'], dest_img)
        shutil.copy2(item['pcl'], dest_pcl)
        shutil.copy2(item['pose'], dest_pose)

def build_datasets():
    print("--- Starting Sonar Dataset Builder ---")
    print(f"Target Output Directory locked to: {OUTPUT_ROOT}\n")
    
    # Pre-create the root output folder safely
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    all_combined_train = []
    all_combined_test = []
    
    # 1. Build the 4 Individual Datasets
    for cat_name, img_subfolder in IMAGE_CATEGORIES.items():
        print(f"Processing category: [{cat_name}]")
        
        img_folder_path = os.path.join(RAW_ROOT, img_subfolder)
        dataset_path = os.path.join(OUTPUT_ROOT, cat_name)
        
        create_dir_structure(dataset_path)
        
        triplets = gather_valid_triplets(img_folder_path)
        if not triplets:
            print(f"  -> WARNING: No valid triplets found for {cat_name}. Check paths/extensions.\n")
            continue
            
        print(f"  -> Found {len(triplets)} valid Image+PCL+Pose triplets.")
        
        # Split data
        train_trips, test_trips = train_test_split(triplets, test_size=TEST_SIZE, random_state=42)
        
        # Save individual datasets
        print(f"  -> Copying files (Train: {len(train_trips)} | Test: {len(test_trips)})...\n")
        copy_triplets(train_trips, "train", dataset_path)
        copy_triplets(test_trips, "test", dataset_path)
        
        # Add to the "all_combined" master lists (we attach the prefix here)
        for t in train_trips:
            all_combined_train.append((t, f"{cat_name}_"))
        for t in test_trips:
            all_combined_test.append((t, f"{cat_name}_"))

    # 2. Build the 5th "All Combined" Dataset
    print("Processing category: [all_combined]")
    combined_path = os.path.join(OUTPUT_ROOT, "all_combined")
    create_dir_structure(combined_path)
    
    print(f"  -> Total Combined Train files: {len(all_combined_train)}")
    print(f"  -> Total Combined Test files: {len(all_combined_test)}")
    print("  -> Copying combined files (this may take a moment)...")
    
    for triplet, prefix in all_combined_train:
        copy_triplets([triplet], "train", combined_path, prefix=prefix)
        
    for triplet, prefix in all_combined_test:
        copy_triplets([triplet], "test", combined_path, prefix=prefix)

    print(f"\n--- Success! All datasets generated in the '{OUTPUT_ROOT}' directory. ---")

if __name__ == "__main__":
    build_datasets()