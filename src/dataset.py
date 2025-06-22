import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class DualInputHairPoseDataset(Dataset):
    def __init__(self, bbox_csv, image_dir, transform=None):
        self.df = pd.read_csv(bbox_csv)
        self.image_dir = image_dir
        self.transform = transform
        self.grouped = self._group_by_image(self.df)

    def _group_by_image(self, df):
        groups = {}
        for _, row in df.iterrows():
            img = row['image']
            if img not in groups:
                groups[img] = {}
            if row['class_id'] in [0, 1]:
                groups[img]['posture'] = row
            elif row['class_id'] in [2, 3, 4]:
                groups[img]['hair'] = row
        return [v for v in groups.values() if 'posture' in v and 'hair' in v]

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        row = self.grouped[idx]
        image_path = os.path.join(self.image_dir, row['posture']['image'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        def crop_bbox(r):
            x, y = r['x_center'] * w, r['y_center'] * h
            bw, bh = r['width'] * w, r['height'] * h
            x1, y1 = max(int(x - bw / 2), 0), max(int(y - bh / 2), 0)
            x2, y2 = min(int(x + bw / 2), w), min(int(y + bh / 2), h)
            return image[y1:y2, x1:x2]

        person_img = crop_bbox(row['posture'])
        hair_img = crop_bbox(row['hair'])

        if self.transform:
            person_img = self.transform(person_img)
            hair_img = self.transform(hair_img)

        posture_label = int(row['posture']['class_id'])      # 0 or 1
        hair_label = int(row['hair']['class_id']) - 2        # 2,3,4 -> 0,1,2

        return person_img, hair_img, posture_label, hair_label

