from torch.utils.data import Dataset
import glob
import os
import torch
import torch.nn as nn  # <-- ADD THIS LINE
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image # (Make sure this is there too if using the updated loader)

class PCDataset(Dataset):
    def __init__(self, stage, transform=None):
        self.transform = transform
        self.stage = stage

        # Define your base directories for custom data
        # Assume a structure like: data/train/images/, data/train/pointclouds/
        self.base_dir = f"{stage}" 
        self.image_dir = os.path.join(self.base_dir, "images")
        self.pc_dir = os.path.join(self.base_dir, "pointclouds")

        # Gather all image files (assuming .png or .jpg)
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")) + 
                                  glob.glob(os.path.join(self.image_dir, "*.jpg")))

    def __len__(self):
        return len(self.image_files)

    def normalize_point_cloud(self, point_cloud):
        centroid = np.mean(point_cloud, axis=0)
        centered_point_cloud = point_cloud - centroid
        if self.stage == "train":
            np.random.shuffle(centered_point_cloud)
        return centered_point_cloud

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Add view dimension: [num_views, C, H, W] -> [1, C, H, W]
        images_tensor = image.unsqueeze(0) 

        # 2. Load Corresponding Ground Truth Point Cloud
        # Assumes the pointcloud file shares the same base name (e.g., sample01.npy)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        pc_path = os.path.join(self.pc_dir, f"{base_name}.npy")
        
        # Load your custom pointcloud (assuming a numpy array of shape [N, 3])
        pc = np.load(pc_path) 
        centroid = np.mean(pc, axis=0) # Save the real-world position
        pc = self.normalize_point_cloud(pc)

        # Return the centroid alongside the data
        return images_tensor, torch.as_tensor(pc, dtype=torch.float32), torch.as_tensor(centroid, dtype=torch.float32), base_name
    


def chamfer_distance(x, y, metric="l2", direction="bi"):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == "y_to_x":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist

def fscore(dist1, dist2, threshold=0.01):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2
    
class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, pred, target):
        # pred and target are expected to have shape (batch_size, 1024, 3)
        assert pred.shape == target.shape
        assert pred.shape[1] == 1024 and pred.shape[2] == 3

        batch_size = pred.shape[0]
        num_points = pred.shape[1]

        # Compute pairwise distances between all points
        diff = pred.unsqueeze(2) - target.unsqueeze(1)
        dist = torch.sum(diff**2, dim=-1)

        # Solve the assignment problem using Hungarian algorithm
        # Note: This is a simplified version and may not be the most efficient for large point clouds
        assignment = torch.zeros_like(dist)
        for b in range(batch_size):
            _, indices = torch.topk(dist[b], k=num_points, largest=False, dim=1)
            assignment[b] = torch.scatter(assignment[b], 1, indices, 1)

        # Compute the EMD
        emd = torch.sum(dist * assignment, dim=[1, 2]) / num_points

        return emd.mean()


import open3d as o3d
def export_to_ply(point_cloud, filename):
    """
    Export a point cloud to a PLY file.
    :param point_cloud: Numpy array of shape (num_points, 3) representing the point cloud.
    :param filename: String, the name of the file to save the point cloud to.
    """
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Write to a PLY file
    o3d.io.write_point_cloud(filename, pcd)


from torchvision import transforms
from PIL import Image
def predict(model, image_path, save_path):

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    input_tensor = transform(image)
    input_tensor = input_tensor.reshape(1,1,3,224,224)

    

    # Invoke the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)


    export_to_ply(output[0], save_path)
    print(f"Image from {image_path} saved to {save_path}")



def pytorch_chamfer_distance(pc1, pc2):
    """
    Computes Chamfer Distance using pure PyTorch operations.
    Accepts shapes (B, N, 3) or (N, 3).
    """
    # Ensure inputs are torch Tensors
    if not isinstance(pc1, torch.Tensor):
        pc1 = torch.from_numpy(pc1).float()
    if not isinstance(pc2, torch.Tensor):
        pc2 = torch.from_numpy(pc2).float()
        
    # Ensure inputs have a batch dimension [B, N, 3]
    if pc1.dim() == 2:
        pc1 = pc1.unsqueeze(0)
    if pc2.dim() == 2:
        pc2 = pc2.unsqueeze(0)

    # Compute pairwise squared distance matrix: (B, N, M)
    dist_matrix = torch.cdist(pc1, pc2, p=2) ** 2
    
    # Minimum distance from pc1 to pc2
    min_dist_pc1_to_pc2 = torch.min(dist_matrix, dim=2)[0]
    # Minimum distance from pc2 to pc1
    min_dist_pc2_to_pc1 = torch.min(dist_matrix, dim=1)[0]
    
    # Return scalar mean of bidirectional distances
    chamfer_loss = torch.mean(min_dist_pc1_to_pc2) + torch.mean(min_dist_pc2_to_pc1)
    return chamfer_loss


class SonarPhysicsLoss(nn.Module):
    def __init__(self, k_neighbors=8, lambda_smooth=1.0, lambda_incidence=0.5):
        super(SonarPhysicsLoss, self).__init__()
        self.k = k_neighbors
        self.lambda_smooth = lambda_smooth
        self.lambda_incidence = lambda_incidence

    def forward(self, pred_pc, centroids):
        """
        pred_pc: [B, N, 3] centered generated point cloud
        centroids: [B, 3] the original centers to restore absolute sonar range
        """
        batch_size, num_points, _ = pred_pc.shape
        
        # 1. Restore absolute coordinates relative to the sonar (assuming sonar is at 0,0,0)
        absolute_pc = pred_pc + centroids.unsqueeze(1)
        
        # Calculate the sound propagation vector (r) from sonar to each point
        # r_ij = P_ij / ||P_ij|| as defined in the paper's Equation 19
        ranges = torch.norm(absolute_pc, p=2, dim=2, keepdim=True)
        r_vectors = absolute_pc / (ranges + 1e-6)

        # 2. Local Planarity / Smoothness Loss (Paper's piecewise planar assumption)
        # Find K-nearest neighbors for every point
        dist_matrix = torch.cdist(pred_pc, pred_pc)
        _, nn_idx = torch.topk(dist_matrix, self.k, dim=2, largest=False)
        
        # Gather neighbor coordinates
        # Shape: [B, N, K, 3]
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, num_points, self.k)
        neighbors = pred_pc[batch_indices, nn_idx] 
        
        # Compute local center of neighbors
        local_mean = torch.mean(neighbors, dim=2)
        
        # Smoothness loss: points should not deviate wildly from their local neighborhood
        loss_smooth = torch.mean(torch.norm(pred_pc - local_mean, p=2, dim=2))

        # 3. Acoustic Incidence Prior (Simplified n * r constraint)
        # Approximate surface normal using the vector from the local mean to the point
        # For a smooth monotonic surface facing the sonar, the local normal should roughly 
        # oppose the propagation vector r.
        approx_normals = pred_pc - local_mean
        approx_normals = approx_normals / (torch.norm(approx_normals, p=2, dim=2, keepdim=True) + 1e-6)
        
        # We want the dot product (n . r) to be negative (meaning the surface faces the sonar)
        # If the dot product is positive, the surface is facing away and couldn't reflect sound!
        dot_product = torch.sum(approx_normals * r_vectors, dim=2)
        
        # Penalize positive dot products (surfaces facing away from the acoustic source)
        loss_incidence = torch.mean(torch.relu(dot_product))

        # Combine losses
        total_physics_loss = (self.lambda_smooth * loss_smooth) + (self.lambda_incidence * loss_incidence)
        
        return total_physics_loss