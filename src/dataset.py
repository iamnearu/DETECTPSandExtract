import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class HairPoseDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.posture_classes = sorted(self.data['posture'].unique())
        self.hair_classes = sorted(self.data['hair_color'].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        posture = self.posture_classes.index(row['posture'])
        hair = self.hair_classes.index(row['hair_color'])
        return image, posture, hair
