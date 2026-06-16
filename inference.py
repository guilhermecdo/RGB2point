
from model import PointCloudNet
from utils import predict
import torch



model_save_name = "last_model_Didson-blur-Transfer-DDO.pth"

model = PointCloudNet(num_views=1, point_cloud_size=1024, num_heads=4, dim_feedforward=2048)
model.load_state_dict(torch.load(model_save_name)["model"])
model.eval()  

image_path = "/home/guilherme/git/RGB2point/Didson-blur/cartesian-images/Didson_blur-1-Cylinder-0.png"
save_path = "result/Didson-blur-Cylinder-0.ply"

predict(model, image_path, save_path)
