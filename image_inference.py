# === image_inference.py ===
import os
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from src.model import HairPoseNet

# === Cấu hình ===
IMAGE_DIR = r"C:\Users\FPT SHOP\OneDrive\Documents\XLA\DETECTPSandExtract\test"     # Thư mục chứa ảnh gốc
OUTPUT_DIR = "test/output"   # Thư mục chứa ảnh đã vẽ kết quả
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Model phân loại ===
model_path = "models/model_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HairPoseNet(num_posture=2, num_hair=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# === Model YOLO detect người ===
person_detector = YOLO("yolov8n.pt")

# === Nhãn ===
posture_labels = ["sitting", "standing"]
hair_labels = ["black", "blonde", "brown"]

# === Transform ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Xử lý từng ảnh ===
for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(image_path)
    if image is None:
        continue

    results = person_detector(image)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # class "person"
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        inp = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out_pose, out_hair = model(inp)
        pred_pose = posture_labels[torch.argmax(out_pose).item()]
        pred_hair = hair_labels[torch.argmax(out_hair).item()]
        label = f"{pred_pose} | {pred_hair}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 3, lineType=cv2.LINE_AA)

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, image)
    print(f"[✔] Processed: {filename} → {output_path}")

print("✅ Đã xử lý xong toàn bộ ảnh.")
