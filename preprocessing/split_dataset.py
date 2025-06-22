import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === 1. Đọc file nhãn gốc sau augment
df = pd.read_csv(r"C:\Users\Iamnearu\Documents\XLA\DETECTPSandExtract\banhr\banhr\labels_clean.csv")

# === 2. Tạo group ảnh gốc (trước _aug hoặc trước .jpg)
df["group"] = df["image"].apply(lambda x: x.split("_aug")[0].split(".")[0])

# === 3. Lấy danh sách group duy nhất để chia theo nhóm
unique_groups = df["group"].unique()
train_groups, temp_groups = train_test_split(unique_groups, test_size=0.3, random_state=42)
val_groups, test_groups = train_test_split(temp_groups, test_size=0.5, random_state=42)

# === 4. Gán lại tập cho từng ảnh
df["split"] = df["group"].apply(lambda g:
    "train" if g in train_groups else (
    "val" if g in val_groups else "test"))

# === 5. Lưu từng tập
os.makedirs("data/splits", exist_ok=True)
df[df["split"] == "train"].drop(columns="split").to_csv("data/splits/train_bbox.csv", index=False)
df[df["split"] == "val"].drop(columns="split").to_csv("data/splits/val_bbox.csv", index=False)
df[df["split"] == "test"].drop(columns="split").to_csv("data/splits/test_bbox.csv", index=False)

# === 6. Thống kê
print(" Đã chia tập CHẶT CHẼ theo nhóm ảnh gốc:")
print(f"  - Train: {df['split'].value_counts().get('train', 0)} dòng nhãn")
print(f"  - Val:   {df['split'].value_counts().get('val', 0)} dòng nhãn")
print(f"  - Test:  {df['split'].value_counts().get('test', 0)} dòng nhãn")
