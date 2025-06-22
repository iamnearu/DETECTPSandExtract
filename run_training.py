from src.train import train_dual_input
import yaml

if __name__ == '__main__':
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_dual_input(
        train_csv=config["data"]["train_bbox_csv"],
        val_csv=config["data"]["val_bbox_csv"],
        image_dir=config["data"]["image_dir"],
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["learning_rate"]
    )
