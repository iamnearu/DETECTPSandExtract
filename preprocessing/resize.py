import os
import cv2
from tqdm import tqdm

# CẤU HÌNH
input_dir = r"/data/images"  # Thư mục chứa ảnh gốc và augment
output_dir = r"/data/images"  # Thư mục lưu ảnh đã resize
resize_size = (224, 224)  # Kích thước mong muốn

# Tạo thư mục lưu nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Duyệt qua các ảnh
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(image_files, desc="Resizing images"):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠ Không đọc được ảnh: {img_name}")
        continue

    resized_img = cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, resized_img)

print(f"\n✅ Đã resize {len(image_files)} ảnh về kích thước {resize_size} và lưu tại: {output_dir}")
