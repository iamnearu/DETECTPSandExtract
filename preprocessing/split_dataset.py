import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_csv = r"C:\Users\FPT SHOP\OneDrive\Documents\XLA\DETECTPSandExtract\dataset\annotations.csv"
df = pd.read_csv(input_csv)

# Chia train/val/test
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df[['posture', 'hair_color']], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[['posture', 'hair_color']], random_state=42)

# Lưu file
os.makedirs("data/splits", exist_ok=True)
train_df.to_csv("data/splits/train.csv", index=False)
val_df.to_csv("data/splits/val.csv", index=False)
test_df.to_csv("data/splits/test.csv", index=False)

print("✅ Đã chia dữ liệu thành train / val / test.")
