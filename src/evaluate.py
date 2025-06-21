# evaluate.py - khởi tạo
    # === src/evaluate.py ===
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import torch
from torch.utils.data import DataLoader
from src.dataset import HairPoseDataset
from src.model import HairPoseNet
from torchvision import transforms


def plot_training_log(log_path="logs/training_log.csv"):
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['loss'], label='Loss')
    plt.plot(df['epoch'], df['f1_pose'], label='F1 Score (Posture)')
    plt.plot(df['epoch'], df['f1_hair'], label='F1 Score (Hair Color)')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training Metrics')
    plt.legend()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_metrics.png")
    plt.close()


def plot_confusion(csv_path, image_dir, model_path, save_dir="results"):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = HairPoseDataset(csv_path, image_dir, transform)
    loader = DataLoader(dataset, batch_size=32)

    model = HairPoseNet(num_posture=len(dataset.posture_classes), num_hair=len(dataset.hair_classes))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    y_true_pose, y_pred_pose = [], []
    y_true_hair, y_pred_hair = [], []

    with torch.no_grad():
        for images, postures, hairs in loader:
            out_pose, out_hair = model(images)
            y_true_pose.extend(postures.numpy())
            y_pred_pose.extend(torch.argmax(out_pose, dim=1).numpy())
            y_true_hair.extend(hairs.numpy())
            y_pred_hair.extend(torch.argmax(out_hair, dim=1).numpy())

    os.makedirs(save_dir, exist_ok=True)

    cm_pose = confusion_matrix(y_true_pose, y_pred_pose)
    disp_pose = ConfusionMatrixDisplay(cm_pose, display_labels=dataset.posture_classes)
    disp_pose.plot()
    plt.title("Confusion Matrix - Posture")
    plt.savefig(f"{save_dir}/confusion_pose.png")
    plt.close()

    cm_hair = confusion_matrix(y_true_hair, y_pred_hair)
    disp_hair = ConfusionMatrixDisplay(cm_hair, display_labels=dataset.hair_classes)
    disp_hair.plot()
    plt.title("Confusion Matrix - Hair Color")
    plt.savefig(f"{save_dir}/confusion_hair.png")
    plt.close()

    # In chi tiết báo cáo
    print("\n [POSTURE] Evaluation Report:")
    print(classification_report(y_true_pose, y_pred_pose, target_names=dataset.posture_classes))

    print("\n [HAIR COLOR] Evaluation Report:")
    print(classification_report(y_true_hair, y_pred_hair, target_names=dataset.hair_classes))
    acc_pose = accuracy_score(y_true_pose, y_pred_pose)
    acc_hair = accuracy_score(y_true_hair, y_pred_hair)

    print(f"\n Độ chính xác (Accuracy) Posture: {acc_pose:.2%}")
    print(f" Độ chính xác (Accuracy) Hair Color: {acc_hair:.2%}")


if __name__ == '__main__':
    plot_training_log()
    plot_confusion(
        csv_path="data/splits/test.csv",
        image_dir="data/images",
        model_path="models/model_best.pt"
    )
