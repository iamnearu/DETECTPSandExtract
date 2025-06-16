import os
import cv2
import albumentations as A
from tqdm import tqdm

# Cấu hình
image_dir = "dataset/images"  # Đường dẫn tới thư mục ảnh
augmented_count = 5           # Mỗi ảnh tạo thêm mấy bản tăng cường

# Pipeline augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.ColorJitter(p=0.3),
    A.Resize(224, 224), 
    A.Normalize()
])

# Duyệt và augment
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(image_files):
    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    for i in range(augmented_count):
        augmented = transform(image=image)['image']
        aug_name = img_name.replace(".", f"_aug{i}.")
        aug_path = os.path.join(image_dir, aug_name)
        # Convert từ normalized về ảnh hiển thị được
        save_img = (augmented * 255).clip(0, 255).astype("uint8")
        cv2.imwrite(aug_path, save_img)
