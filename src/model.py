# model.py - khởi tạo
# === src/model.py ===
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class HairPoseNet(nn.Module):
    def __init__(self, num_posture=2, num_hair=4):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # bỏ FC cuối
        self.fc_pose = nn.Linear(512, num_posture)
        self.fc_hair = nn.Linear(512, num_hair)

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return self.fc_pose(x), self.fc_hair(x)


