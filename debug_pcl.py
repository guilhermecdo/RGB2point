import os
import sys
import numpy as np
import open3d as o3d

def visualize_point_cloud(npy_path, save_snapshot=False, snapshot_path="pc_debug.png"):
    if not os.path.exists(npy_path):
        print(f"Error: File not found at '{npy_path}'")
        return

    # 1. Load the numpy point cloud
    print(f"Loading point cloud from: {npy_path}")
    points = np.load(npy_path)
    
    # Check shape assumptions
    print(f"Point cloud shape: {points.shape}")
    if len(points.shape) == 3:
        # If it has a batch dimension, take the first sample [N, 3]
        points = points[0]
        print(f"Extracted first sample from batch. New shape: {points.shape}")

    if points.shape[1] != 3:
        print(f"Error: Expected an array of shape (N, 3), but got {points.shape}")
        return

    # 2. Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Optional: Estimate normals so lighting looks good in 3D
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # 3. Visualization strategy
    if save_snapshot:
        # Headless rendering mode (Great for remote SSH/HPC setups without a GUI display)
        print(f"Rendering snapshot to: {snapshot_path}")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False) # Keep window hidden
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(snapshot_path)
        vis.destroy_window()
        print("Snapshot saved successfully.")
    else:
        # Open an interactive live window
        print("Opening interactive 3D visualization window...")
        print("-> Use your mouse to rotate, zoom, and pan.")
        print("-> Press 'Q' or 'Esc' to close the window.")
        
        # Draw geometry with a black background for clear visibility
        o3d.visualization.draw_geometries(
            [pcd], 
            window_name="Ground Truth Point Cloud Debugger",
            width=800, 
            height=600
        )

if __name__ == "__main__":
    # --- QUICK CONFIGURATION ---
    # Put the path to any of your generated .npy files here
    SAMPLE_PCL = "/home/guilherme/git/RGB2point/data/Didson-original/train/pointclouds/original_Didson-denoise_2-Ring-50.npy"
    
    # Set to True if running on an SSH server or inside WSL2 without a GUI server configured
    HEADLESS_MODE = False 
    
    visualize_point_cloud(SAMPLE_PCL, save_snapshot=HEADLESS_MODE, snapshot_path="debug_result.png")