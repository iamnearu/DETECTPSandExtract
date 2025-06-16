# train.py - khởi tạo
# === src/train.py ===
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import os
from src.dataset import HairPoseDataset
from src.model import HairPoseNet
from torchvision import transforms
import pandas as pd


def train_model(train_csv, val_csv, image_dir, num_epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_set = HairPoseDataset(train_csv, image_dir, transform)
    val_set = HairPoseDataset(val_csv, image_dir, transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = HairPoseNet(num_posture=len(train_set.posture_classes), num_hair=len(train_set.hair_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    os.makedirs("models", exist_ok=True)
    log = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, postures, hairs in train_loader:
            images, postures, hairs = images.to(device), postures.to(device), hairs.to(device)

            out_pose, out_hair = model(images)
            loss_pose = criterion(out_pose, postures)
            loss_hair = criterion(out_hair, hairs)
            loss = loss_pose + loss_hair

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        y_true_pose, y_pred_pose = [], []
        y_true_hair, y_pred_hair = [], []
        with torch.no_grad():
            for images, postures, hairs in val_loader:
                images = images.to(device)
                out_pose, out_hair = model(images)
                y_true_pose.extend(postures.tolist())
                y_pred_pose.extend(torch.argmax(out_pose, dim=1).cpu().tolist())
                y_true_hair.extend(hairs.tolist())
                y_pred_hair.extend(torch.argmax(out_hair, dim=1).cpu().tolist())

        f1_pose = f1_score(y_true_pose, y_pred_pose, average='macro')
        f1_hair = f1_score(y_true_hair, y_pred_hair, average='macro')
        avg_f1 = (f1_pose + f1_hair) / 2

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - F1(pose): {f1_pose:.4f} - F1(hair): {f1_hair:.4f}")
        log.append([epoch+1, total_loss, f1_pose, f1_hair])

        # Save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), "models/model_best.pt")

    # Save final model and log
    torch.save(model.state_dict(), "models/model_last.pt")
    pd.DataFrame(log, columns=["epoch", "loss", "f1_pose", "f1_hair"]).to_csv("logs/training_log.csv", index=False)
