# YOLO Steel Counter

Project đếm thép tự động sử dụng YOLO và RTSP camera.

## 🚀 Quick Start

### 1. Clone project
```bash
git clone <URL_REPOSITORY>
cd <FOLDER_NAME>
```

### 2. Cài đặt
```bash
# Tạo virtual environment
python -m venv yolo_env
yolo_env\Scripts\activate  # Windows
# source yolo_env/bin/activate  # Linux/Mac

# Cài packages
pip install -r requirements.txt
```

### 3. Tải model
Model file `steel_model_v112.pt` không có trong Git (quá lớn).
- Tải từ: [Link Google Drive / Dropbox]
- Đặt vào thư mục gốc project

### 4. Chạy
```bash
python count_v5.py
```

## 📁 Cấu trúc project

```
├── count_v5.py              # Script chính
├── steel_model_v112.pt      # Model YOLO (không có trong Git)
├── custom_botsort.yaml      # Config tracker
├── mydataset.yaml           # Config dataset
├── requirements.txt         # Dependencies
├── SETUP_GUIDE.md          # Hướng dẫn chi tiết
└── OPTIMIZE_RTSP_LATENCY.md # Tối ưu độ trễ camera
```

## 📋 Requirements

- Python 3.10.11
- OpenCV
- Ultralytics (YOLOv8)
- PyTorch

Xem đầy đủ trong [requirements.txt](requirements.txt)

## 📷 Camera hỗ trợ

- Hikvision iDS-TCM403-BI
- Fidus AfidusVTW-LPR-2MIB
- Các camera RTSP khác

Xem hướng dẫn tối ưu trong [OPTIMIZE_RTSP_LATENCY.md](OPTIMIZE_RTSP_LATENCY.md)

## 🔧 Cấu hình

Sửa trong `count_v5.py`:

```python
# Thay IP camera
SOURCE = "rtsp://admin:admin@192.168.0.100/..."

# Thay model path nếu cần
MODEL_PATH = "steel_model_v112.pt"

# Vùng đếm (region)
ORIGINAL_REGION = [(420, 307), (969, 307), (969, 452), (420, 452)]
```

## 📚 Documentation

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Hướng dẫn cài đặt đầy đủ
- [OPTIMIZE_RTSP_LATENCY.md](OPTIMIZE_RTSP_LATENCY.md) - Giảm độ trễ camera

## ⚠️ Lưu ý

- Model file **KHÔNG** được commit vào Git (quá lớn)
- Virtual environment **KHÔNG** commit
- Dataset và video output **KHÔNG** commit
- Xem [.gitignore](.gitignore) để biết chi tiết

## 📝 License

[Thêm license của bạn ở đây]
