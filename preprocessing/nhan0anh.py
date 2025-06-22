import pandas as pd
import os

# Đọc file CSV nhãn
csv_path = r"C:\Users\Iamnearu\Documents\XLA\DETECTPSandExtract\banhr\banhr\labels_clean.csv"
image_dir = r"/data/images"
df = pd.read_csv(csv_path)

# Đếm số dòng nhãn ứng với mỗi ảnh
count_per_image = df['image'].value_counts()

# Lọc ra những ảnh có đúng 2 nhãn (ví dụ tóc + người)
valid_images = set(count_per_image[count_per_image == 2].index)

# Lặp qua thư mục ảnh và xóa những ảnh không nằm trong valid_images
deleted = 0
for fname in os.listdir(image_dir):
    if fname not in valid_images:
        os.remove(os.path.join(image_dir, fname))
        deleted += 1

print(f"✅ Đã xóa {deleted} ảnh không có đủ nhãn.")
