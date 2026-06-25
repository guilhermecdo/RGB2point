import torch.nn as nn
from torch.nn import MultiheadAttention
import torch
import timm

class PointCloudGeneratorWithAttention(nn.Module):
    def __init__(self, input_feature_dim, point_cloud_size, num_heads=16, dim_feedforward=2048):
        super(PointCloudGeneratorWithAttention, self).__init__()
        
        self.self_attention = MultiheadAttention(
            embed_dim=input_feature_dim, num_heads=num_heads
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(input_feature_dim, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_feedforward, point_cloud_size * 3), 
        )
        self.point_cloud_size = point_cloud_size

    # Notice this only takes 'x'. The fused features are passed here!
    def forward(self, x):
        # x shape: [batch_size, seq_length, input_feature_dim]
        x = x.transpose(0, 1)  

        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)  

        # Pass through the linear layers
        point_cloud = self.linear_layers(attn_output.flatten(start_dim=1))
        point_cloud = point_cloud.view(-1, self.point_cloud_size, 3)
        
        return point_cloud


class PointCloudNet(nn.Module):
    # Added pose_dim=16 to the initialization
    def __init__(self, num_views, point_cloud_size, num_heads, dim_feedforward, pose_dim=16):
        super(PointCloudNet, self).__init__()
        
        # Load the pretrained Vision Transformer model
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        num_features = self.vit.num_features

        # Image Feature Aggregator
        out_features = 1024 * 4
        self.aggregator = nn.Linear(num_features, out_features)
        
        # --- Cross-Modal Pose Encoder ---
        pose_embed_dim = 1024
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, pose_embed_dim),
            nn.LeakyReLU(0.2)
        )
        
        # The generator must expect the combined size (Image Features + Pose Embedding)
        combined_feature_dim = out_features + pose_embed_dim
        
        self.point_cloud_generator = PointCloudGeneratorWithAttention(
            input_feature_dim=combined_feature_dim,
            point_cloud_size=point_cloud_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )

    # This is the forward pass that requires 'pose'
    def forward(self, x, pose):
        batch_size, num_views, C, H, W = x.shape

        x = x.view(batch_size * num_views, C, H, W)

        with torch.no_grad():
            features = self.vit(x)

        features = features.view(batch_size, num_views, -1)
        mean_features = torch.mean(features, dim=1)

        # 1. Image Features
        aggregated_features = self.aggregator(mean_features)
        
        # 2. Pose Features
        encoded_pose = self.pose_encoder(pose)
        
        # 3. Cross-Modal Fusion
        fused_features = torch.cat((aggregated_features, encoded_pose), dim=1)
        fused_features = fused_features.unsqueeze(1)

        # 4. Generate point cloud using the fused context
        point_cloud = self.point_cloud_generator(fused_features)
        point_cloud = point_cloud.view(batch_size, -1, 3)

        return point_cloud