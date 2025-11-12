import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
from main_model_14 import HiFuse_Small

net = HiFuse_Small(num_classes=7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
net.eval()


target_layers = [net.fu4.norm3]


origin_img = cv2.imread('./view-1.jpg')
rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)


trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224)
])
crop_img = trans(rgb_img)
net_input = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop_img).unsqueeze(0)


canvas_img = (crop_img * 255).byte().numpy().transpose(1, 2, 0)
canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)


cam = pytorch_grad_cam.GradCAMPlusPlus(model=net, target_layers=target_layers)
grayscale_cam = cam(net_input)
grayscale_cam = grayscale_cam[0, :]


src_img = np.float32(canvas_img) / 255
visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
cv2.imshow('feature map', visualization_img)
cv2.waitKey(0)