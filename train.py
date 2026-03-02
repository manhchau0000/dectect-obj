from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # tự download nếu chưa có

model.train(
    data="mydataset.yaml",
    epochs=100,
    imgsz=320,
    batch=16,       # giảm xuống 8 hoặc 4 nếu báo lỗi hết RAM
    device="cpu",       # GPU, đổi thành "cpu" nếu không có GPU
)

