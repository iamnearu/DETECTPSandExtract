import os
import pandas as pd

# CẤU HÌNH
image_dir = r"/data/images"  # Thư mục chứa ảnh gốc
csv_path = r"C:\Users\Iamnearu\Documents\XLA\DETECTPSandExtract\banhr\banhr\labels_clean.csv"          # File nhãn đầu vào
output_csv_path = r"C:\Users\Iamnearu\Documents\XLA\DETECTPSandExtract\banhr\banhr\labels_clean.csv"  # File nhãn đầu ra (sau khi lọc)

# Đọc file nhãn
df = pd.read_csv(csv_path)

# Đếm số nhãn ứng với từng ảnh
label_counts = df['image'].value_counts()

# Ảnh hợp lệ: chỉ những ảnh có đúng 2 dòng nhãn (1 tóc, 1 pose)
valid_images = set(label_counts[label_counts == 2].index)

# Danh sách ảnh hiện có trong thư mục
all_images = set(os.listdir(image_dir))

# Lặp qua ảnh trong thư mục và xóa những ảnh KHÔNG hợp lệ
deleted_images = 0
for img_name in all_images:
    if img_name not in valid_images:
        img_path = os.path.join(image_dir, img_name)
        try:
            os.remove(img_path)
            deleted_images += 1
        except Exception as e:
            print(f" Không xóa được {img_name}: {e}")

# Lọc lại nhãn cho các ảnh hợp lệ
df_clean = df[df['image'].isin(valid_images)]
df_clean.to_csv(output_csv_path, index=False)

# Thống kê
print(" Đã xử lý xong!")
print(f" Ảnh hợp lệ giữ lại: {len(valid_images)}")
print(f" Ảnh đã xóa: {deleted_images}")
print(f" File nhãn mới lưu tại: {output_csv_path}")
