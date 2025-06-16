# === run_training.py ===
import yaml
from src.train import train_model

if __name__ == '__main__':
    # Đọc cấu hình từ config.yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_model(
        train_csv=config["data"]["train_csv"],
        val_csv=config["data"]["val_csv"],
        image_dir=config["data"]["image_dir"],
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["learning_rate"]
    )
