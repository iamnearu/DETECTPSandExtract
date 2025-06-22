import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.dataset import DualInputHairPoseDataset
from src.model import DualInputHairPoseNet
from src.transforms import get_train_transforms, get_val_transforms
import pandas as pd
import os
from sklearn.metrics import f1_score

def train_dual_input(train_csv, val_csv, image_dir, num_epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = DualInputHairPoseDataset(train_csv, image_dir, transform=get_train_transforms())
    val_set = DualInputHairPoseDataset(val_csv, image_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = DualInputHairPoseNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    log = []
    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for person_img, hair_img, pose_label, hair_label in train_loader:
            person_img, hair_img = person_img.to(device), hair_img.to(device)
            pose_label, hair_label = pose_label.to(device), hair_label.to(device)
            out_pose, out_hair = model(person_img, hair_img)
            loss = criterion(out_pose, pose_label) + criterion(out_hair, hair_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        y_true_pose, y_pred_pose = [], []
        y_true_hair, y_pred_hair = [], []
        with torch.no_grad():
            for person_img, hair_img, pose_label, hair_label in val_loader:
                person_img, hair_img = person_img.to(device), hair_img.to(device)
                out_pose, out_hair = model(person_img, hair_img)
                y_true_pose.extend(pose_label.numpy())
                y_pred_pose.extend(torch.argmax(out_pose, dim=1).cpu().numpy())
                y_true_hair.extend(hair_label.numpy())
                y_pred_hair.extend(torch.argmax(out_hair, dim=1).cpu().numpy())

        f1_pose = f1_score(y_true_pose, y_pred_pose, average='macro')
        f1_hair = f1_score(y_true_hair, y_pred_hair, average='macro')
        avg_f1 = (f1_pose + f1_hair) / 2
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - F1(pose): {f1_pose:.4f} - F1(hair): {f1_hair:.4f}")
        log.append([epoch+1, total_loss, f1_pose, f1_hair])

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), "models/model_best.pt")

    pd.DataFrame(log, columns=["epoch", "loss", "f1_pose", "f1_hair"]).to_csv("logs/training_log.csv", index=False)
    torch.save(model.state_dict(), "models/model_last.pt")