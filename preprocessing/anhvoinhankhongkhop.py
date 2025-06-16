import os
import pandas as pd

# === CẤU HÌNH ===
image_dir = r"C:\Users\FPT SHOP\OneDrive\Documents\XLA\DETECTPSandExtract\dataset\images"  # Thư mục chứa ảnh thật
csv_path = "annotations.csv"  # File annotation cần kiểm tra
output_csv_path = "annotations.csv"  # File xuất ra

# === BƯỚC 1: Đọc file CSV ===
df = pd.read_csv(csv_path)
print(f"🔍 Tổng số dòng trong annotation: {len(df)}")

# === BƯỚC 2: Lấy danh sách ảnh thực sự tồn tại ===
actual_images = set(os.listdir(image_dir))

# === BƯỚC 3: Giữ lại chỉ những ảnh có tồn tại ===
df_matched = df[df['image'].isin(actual_images)]
print(f"✅ Số ảnh có tồn tại thực tế: {len(df_matched)}")

# === BƯỚC 4: Lưu file mới ===
df_matched.to_csv(output_csv_path, index=False)
print(f"💾 Đã lưu annotation mới vào: {output_csv_path}")
