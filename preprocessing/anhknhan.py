import pandas as pd
import os

# Đường dẫn đến file CSV nhãn và thư mục chứa ảnh
csv_path = r"C:\Users\Iamnearu\Documents\XLA\DETECTPSandExtract\banhr\banhr\train\annotations_yolo.csv"
image_dir = r"/data/images"

# Đọc file nhãn
df = pd.read_csv(csv_path)

# Lấy danh sách file ảnh còn tồn tại trong thư mục
existing_images = set(os.listdir(image_dir))

# Lọc lại DataFrame chỉ giữ những dòng có ảnh thực sự còn tồn tại
df_clean = df[df['image'].isin(existing_images)]

# Ghi ra file mới
df_clean.to_csv(r"C:\Users\Iamnearu\Documents\XLA\DETECTPSandExtract\banhr\banhr\labels_clean.csv", index=False)

print(f"Giữ lại {len(df_clean)} dòng nhãn hợp lệ từ tổng số {len(df)}.")
