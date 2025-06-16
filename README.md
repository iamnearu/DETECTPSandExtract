📌 Dự án: Hệ thống nhận diện người và trích xuất thông tin đặc trưng từ ảnh

🎯 Mục tiêu

Xây dựng hệ thống AI có khả năng:

Phát hiện người trong ảnh hoặc video

Trích xuất đặc trưng: dáng đứng (standing/sitting) và màu tóc (black, brown, blonde, other)

Dự án ứng dụng học sâu (Deep Learning) kết hợp với mô hình ResNet18, hoạt động tốt trên tập dữ liệu ảnh tự thu thập.

🧠 Kiến trúc hệ thống

Tiền xử lý dữ liệu:

Gán nhãn ảnh người: posture, hair_color

Resize về 224x224

Augmentation: lật, xoay, tăng sáng

Chia tập: train.csv, val.csv, test.csv

Mô hình học sâu:

Backbone: ResNet18 (pretrained)

2 nhánh phân loại: fc_pose và fc_hair

Huấn luyện với CrossEntropyLoss và Adam Optimizer

Triển khai:

Dự đoán ảnh bất kỳ bằng model_best.pt

Dự đoán trên video thông qua YOLOv8 + mô hình phân loại

📂 Cấu trúc thư mục

DETECTPSandExtract/
├── data/            # ảnh + nhãn
├── preprocessing/   # xử lý ảnh
├── src/             # code train/predict/evaluate
├── models/          # lưu model_best.pt
├── results/         # lưu biểu đồ, confusion matrix
├── run_training.py  # chạy huấn luyện
├── run_predict.py   # dự đoán ảnh mới
├── video_inference.py # chạy trên video
├── config.yaml
└── requirements.txt

▶️ Cài đặt & chạy

🔧 Cài thư viện cần thiết:

pip install -r requirements.txt

🏋️ Huấn luyện mô hình:

python run_training.py

Model sẽ lưu tại: models/model_best.pt

🧠 Dự đoán ảnh đơn:

python src/predict.py data/images/example.jpg

Kết quả: in tư thế và màu tóc ra màn hình.

🎥 Dự đoán trên video:

python video_inference.py

Kết quả lưu tại: output.mp4

📊 Đánh giá mô hình

python -m src.evaluate

Vẽ biểu đồ: results/training_metrics.png

Confusion matrix: results/confusion_pose.png, confusion_hair.png

F1-score và accuracy in ra terminal

📌 Gợi ý mở rộng

Bổ sung nhận diện giới tính, độ tuổi

Phân tích biểu cảm khuôn mặt

Tích hợp real-time qua webcam hoặc Streamlit app