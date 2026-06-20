import torch.nn as nn
from torch.nn import MultiheadAttention
import torch
import timm
class PointCloudGeneratorWithAttention(nn.Module):
    def __init__(
        self, input_feature_dim, point_cloud_size, num_heads=16, dim_feedforward=2048
    ):
        super(PointCloudGeneratorWithAttention, self).__init__()
        print(f"input_feature_dim:{input_feature_dim}")
        print(f"dim_feedforward:{dim_feedforward}")
        print(f"point_cloud_size:{point_cloud_size*3}")
        self.self_attention = MultiheadAttention(
            embed_dim=input_feature_dim, num_heads=num_heads
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(input_feature_dim, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(
                dim_feedforward, point_cloud_size * 3
            ),  # Output layer for point cloud
        )
        self.point_cloud_size = point_cloud_size

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_feature_dim]
        # Transpose for the attention layer
        x = x.transpose(0, 1)  # Shape: [seq_length, batch_size, input_feature_dim]

        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)

        # Transpose back
        attn_output = attn_output.transpose(
            0, 1
        )  # Shape: [batch_size, seq_length, input_feature_dim]

        # Pass through the linear layers
        point_cloud = self.linear_layers(attn_output.flatten(start_dim=1))

        # Reshape to (batch_size, point_cloud_size, 3)
        point_cloud = point_cloud.view(-1, self.point_cloud_size, 3)
        return point_cloud


class PointCloudNet(nn.Module):
    # Add 'pose_dim' to your init. Defaulting to 6 for standard [x,y,z,r,p,y].
    def __init__(self, num_views, point_cloud_size, num_heads, dim_feedforward, pose_dim=6):
        super(PointCloudNet, self).__init__()
        
        # Load the pretrained ViT
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        num_features = self.vit.num_features

        # Aggregate features from different views
        out_features = 1024 * 4
        self.aggregator = nn.Linear(num_features, out_features)
        
        # --- NEW: Cross-Modal Pose Encoder ---
        pose_embed_dim = 1024
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, pose_embed_dim),
            nn.LeakyReLU(0.2)
        )
        
        # --- MODIFIED: Point Cloud Generator Input Size ---
        # It now receives the image features (out_features) PLUS the pose embedding (pose_embed_dim)
        combined_feature_dim = out_features + pose_embed_dim
        
        self.point_cloud_generator = PointCloudGeneratorWithAttention(
            input_feature_dim=combined_feature_dim,
            point_cloud_size=point_cloud_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )

    # Add 'pose' to the forward pass arguments
    def forward(self, x, pose):
        batch_size, num_views, C, H, W = x.shape

        x = x.view(batch_size * num_views, C, H, W)

        with torch.no_grad():
            features = self.vit(x)

        features = features.view(batch_size, num_views, -1)
        mean_features = torch.mean(features, dim=1)

        # 1. Image Features
        aggregated_features = self.aggregator(mean_features)
        
        # 2. Pose Features (Cross-Modal embedding)
        encoded_pose = self.pose_encoder(pose)
        
        # 3. Early Fusion: Concatenate Image and Pose features
        # Shape becomes [batch_size, out_features + pose_embed_dim]
        fused_features = torch.cat((aggregated_features, encoded_pose), dim=1)
        fused_features = fused_features.unsqueeze(1)

        # Generate point cloud using the fused context
        point_cloud = self.point_cloud_generator(fused_features)
        point_cloud = point_cloud.view(batch_size, -1, 3)

        return point_cloud