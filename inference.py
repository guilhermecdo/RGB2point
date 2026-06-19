
from model import PointCloudNet
from utils import predict
import torch



model_save_name = "last_model_Didson-original-PINN-TRANSFER-1000-epochs.pth"

model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
model.load_state_dict(torch.load(model_save_name)["model"])
model.eval()  

image_path = "/home/guilherme/git/RGB2point/data/Didson-original/train/images/_Didson_1-Cylinder-0.png"
save_path = "result/PINN-1000-last-Fix-original-Cylinder-0.ply"

predict(model, image_path, save_path)
