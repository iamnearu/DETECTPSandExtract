# predict.py - khởi tạo
# === src/predict.py ===
import torch
import cv2
import sys
import os
from torchvision import transforms
from src.model import HairPoseNet
from src.dataset import HairPoseDataset


def predict(image_path, model_path="models/model_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # Tạo nhãn thủ công (do không có file csv ở bước này)
    posture_labels = ['sitting', 'standing']
    hair_labels = ['black', 'blonde', 'brown', 'other']

    # Load model
    model = HairPoseNet(num_posture=len(posture_labels), num_hair=len(hair_labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        out_pose, out_hair = model(image_tensor)
        pred_pose = posture_labels[torch.argmax(out_pose).item()]
        pred_hair = hair_labels[torch.argmax(out_hair).item()]

    print(f"🧍 Tư thế dự đoán: {pred_pose}")
    print(f"🎨 Màu tóc dự đoán: {pred_hair}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Vui lòng cung cấp đường dẫn tới ảnh cần dự đoán.")
        print("Cách dùng: python src/predict.py path/to/image.jpg")
    else:
        image_path = sys.argv[1]
        predict(image_path)