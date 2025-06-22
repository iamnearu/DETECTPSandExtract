import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DualInputHairPoseNet(nn.Module):
    def __init__(self, posture_classes=2, hair_classes=3):
        super().__init__()
        self.backbone_pose = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])
        self.backbone_hair = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])
        self.fc_pose = nn.Linear(512, posture_classes)
        self.fc_hair = nn.Linear(512, hair_classes)

    def forward(self, x_pose, x_hair):
        f_pose = self.backbone_pose(x_pose).view(x_pose.size(0), -1)
        f_hair = self.backbone_hair(x_hair).view(x_hair.size(0), -1)
        return self.fc_pose(f_pose), self.fc_hair(f_hair)

