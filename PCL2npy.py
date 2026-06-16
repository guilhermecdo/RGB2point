import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR = "data"          # Main output directory
TARGET_POINTS = 1024         # Fixed size required by your PointCloudNet model
TEST_SIZE = 0.2              # 20% of the data goes to the test set (80% to train)

def parse_xyz_file(file_path):
    """Parses a standard space-separated .xyz file."""
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    return np.array(points, dtype=np.float32)

def enforce_point_count(points, expected_points):
    """Resamples point clouds to meet a fixed sequence tensor size."""
    num_points = points.shape[0]
    if num_points == 0:
        return None
    if num_points > expected_points:
        indices = np.random.choice(num_points, expected_points, replace=False)
        points = points[indices]
    elif num_points < expected_points:
        indices = np.random.choice(num_points, expected_points, replace=True)
        points = points[indices]
    return points

def gather_valid_pairs(mission_path):
    img_dir = os.path.join(mission_path, "cartesian-images")
    pcl_dir = os.path.join(mission_path, "lidar-pcl")

    if not (os.path.isdir(img_dir) and os.path.isdir(pcl_dir)):
        print(f"Error: Could not find 'cartesian-images' or 'lidar-pcl' subfolders inside: {mission_path}")
        return []

    pcl_files = glob.glob(os.path.join(pcl_dir, "*.xyz"))
    print(f"Found {len(pcl_files)} total point cloud files.")

    valid_pairs = []
    for pcl_path in pcl_files:
        basename = os.path.basename(pcl_path)
        core_name = os.path.splitext(basename)[0]
        
        possible_img_names = [
            f"{core_name}.png", 
            f"blur-{core_name}.png", 
            f"gauss-blur-{core_name}.png", 
            f"median-blur-{core_name}.png"
        ]
        
        img_path = None
        for name in possible_img_names:
            test_path = os.path.join(img_dir, name)
            if os.path.isfile(test_path):
                img_path = test_path
                break

        if img_path:
            valid_pairs.append({
                "img": img_path,
                "pcl": pcl_path,
                "name": core_name
            })
            
    return valid_pairs

def save_split_data(pairs, stage, experiment_name):
    """Processes and saves the file pairs into the chosen stage subfolder."""
    dest_img_dir = os.path.join(OUTPUT_DIR, experiment_name, stage, "images")
    dest_pcl_dir = os.path.join(OUTPUT_DIR, experiment_name, stage, "pointclouds")
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_pcl_dir, exist_ok=True)

    for item in pairs:
        # 1. Process and save Point Cloud to .npy
        raw_points = parse_xyz_file(item["pcl"])
        processed_points = enforce_point_count(raw_points, TARGET_POINTS)
        if processed_points is None:
            continue
        
        np.save(os.path.join(dest_pcl_dir, f"{item['name']}.npy"), processed_points)

        # 2. Copy the matched image
        dest_img_path = os.path.join(dest_img_dir, f"{item['name']}.png")
        with open(item["img"], "rb") as sf, open(dest_img_path, "wb") as df:
            df.write(sf.read())

if __name__ == "__main__":
    print("--- Simplified Splitter Manual Dataset Builder ---")
    
    target_mission = "/home/guilherme/git/RGB2point/Didson-blur"
    chosen_name = "Didson-blur"
    
    # 1. Find all valid matched pairs
    pairs = gather_valid_pairs(target_mission)
    
    if not pairs:
        print("No matching image/point cloud pairs found. Exiting.")
    else:
        print(f"Total matched pairs available: {len(pairs)}")
        
        # 2. Perform the Train / Test split
        train_pairs, test_pairs = train_test_split(pairs, test_size=TEST_SIZE, random_state=42)
        
        # 3. Process and save each subfolder block
        print(f"Splitting data into subfolders... (Train: {len(train_pairs)} | Test: {len(test_pairs)})")
        save_split_data(train_pairs, stage="train", experiment_name=chosen_name)
        save_split_data(test_pairs, stage="test", experiment_name=chosen_name)
        
        print(f"\nSuccess! Data perfectly split and structured inside: ./{OUTPUT_DIR}/{chosen_name}/")