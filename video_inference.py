# === video_inference.py ===
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from src.model import HairPoseNet

# Load YOLOv8 person detector
person_detector = YOLO("yolov8n.pt")  # dùng model nhẹ để chạy nhanh

# Load model phân loại
model_path = "models/model_best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HairPoseNet(num_posture=2, num_hair=3)  # khớp đúng với model bạn đã train
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Nhãn
posture_labels = ["sitting", "standing"]
hair_labels = ["black", "blonde", "brown"]

# Transform ảnh crop
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Mở video
cap = cv2.VideoCapture(r"C:\Users\FPT SHOP\OneDrive\Documents\XLA\DETECTPSandExtract\11728100-uhd_2160_3840_24fps.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = person_detector(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # chỉ lấy class "person"
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        inp = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out_pose, out_hair = model(inp)
        pred_pose = posture_labels[torch.argmax(out_pose).item()]
        pred_hair = hair_labels[torch.argmax(out_hair).item()]
        label = f"{pred_pose} | {pred_hair}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,           # tăng kích thước chữ lên
            (0, 255, 0),   # màu xanh lá
            3,             # độ đậm nét hơn
            lineType=cv2.LINE_AA)


    out.write(frame)

cap.release()
out.release()
print("✅ Đã xuất video: output.mp4")
