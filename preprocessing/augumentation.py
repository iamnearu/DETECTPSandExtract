import os
import cv2
import albumentations as A
import pandas as pd
from tqdm import tqdm

# === CẤU HÌNH ===
image_dir = r"/data/images"
label_path = r"C:\Users\Iamnearu\Documents\XLA\DETECTPSandExtract\banhr\banhr\labels_clean.csv"
augmented_count = 5

# Đọc nhãn gốc
df = pd.read_csv(label_path)

# Gom nhãn theo ảnh (mỗi ảnh có 2 dòng nhãn: posture + hair)
grouped = df.groupby("image")
output_labels = []

# Aug pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_id']))

# Thực hiện augment
for img_name, rows in tqdm(grouped, desc="Augmenting"):
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    if image is None or len(rows) != 2:
        continue

    bboxes = rows[["x_center", "y_center", "width", "height"]].values.tolist()
    class_ids = rows["class_id"].tolist()

    for i in range(augmented_count):
        try:
            aug = transform(image=image, bboxes=bboxes, class_id=class_ids)
            aug_img = aug["image"]
            aug_bboxes = aug["bboxes"]
            aug_classes = aug["class_id"]

            name, ext = os.path.splitext(img_name)
            aug_name = f"{name}_aug{i}{ext}"
            aug_path = os.path.join(image_dir, aug_name)
            cv2.imwrite(aug_path, aug_img)

            # Luôn ghi đúng 2 dòng nhãn cho ảnh augment
            for bbox, cls in zip(aug_bboxes, aug_classes):
                output_labels.append({
                    "image": aug_name,
                    "class_id": cls,
                    "x_center": bbox[0],
                    "y_center": bbox[1],
                    "width": bbox[2],
                    "height": bbox[3]
                })

        except Exception as e:
            print(f"⚠️ Lỗi augment ảnh {img_name}: {e}")

# Gộp lại nhãn mới và ghi đè
df_aug = pd.DataFrame(output_labels)
df_all = pd.concat([df, df_aug]).reset_index(drop=True)
df_all.to_csv(label_path, index=False)

print("\n✅ Đã tăng cường toàn bộ ảnh (không kiểm tra class_id)")
print(f"📊 Số ảnh augment: {len(df_aug['image'].unique())}")
print(f"📊 Tổng dòng nhãn sau gộp: {len(df_all)}")
