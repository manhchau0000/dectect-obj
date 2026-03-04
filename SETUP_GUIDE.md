# Hướng dẫn cài đặt project YOLO Steel Counter

## Bước 1: Clone project từ Git
```cmd
git clone <URL_REPOSITORY>
cd <TÊN_FOLDER_PROJECT>
```

## Bước 2: Cài đặt Python
1. Tải Python 3.10.11 từ https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
2. Khi cài, **NHỚ TICK** "Add Python to PATH"


## Bước 3: Tạo Virtual Environment
```cmd
python -m venv yolo_env
yolo_env\Scripts\activate
```

## Bước 4: Cài đặt các packages
```cmd
pip install -r requirements.txt
```

**Lưu ý:** Nếu cài torch bị lỗi hoặc muốn dùng GPU CUDA:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Bước 5: Tải model file (nếu chưa có)
Model file `steel_model_v112.pt` không được đưa lên Git vì quá lớn (~22MB).

**Cách 1:** Tải từ Google Drive / Dropbox / OneDrive
```cmd
# Đặt file vào thư mục gốc project
# Ví dụ: C:\path\to\project\steel_model_v112.pt
```

**Cách 2:** Train model mới (nếu có dataset)
```cmd
python train.py
```

**Cách 3:** Dùng model YOLO mặc định để test
```python
# Trong count_v5.py, tạm thời thay:
MODEL_PATH = "yolov8n.pt"  # Model nhỏ để test
```

## Bước 6: Chỉnh sửa config
Trong file `count_v5.py`, sửa lại URL camera RTSP:
```python
SOURCE = "rtsp://admin:admin@192.168.0.100/..."
```

## Bước 7: Chạy thử
```cmd
python count_v5.py
```

---

## Xử lý lỗi thường gặp:

### Lỗi "No module named 'cv2'"
```cmd
pip install opencv-python
```

### Lỗi "FileNotFoundError: steel_model_v112.pt"
Model file chưa có trong project. Xem **Bước 5** để tải model.

### Lỗi "CUDA not available" (nếu muốn dùng GPU)
- Cài NVIDIA GPU driver
- Cài CUDA Toolkit phù hợp với version PyTorch

### Lỗi kết nối camera
- Kiểm tra IP camera
- Kiểm tra username/password
- Thử ping camera trước

---

## Kiểm tra cài đặt thành công:
```cmd
python -c "import cv2; import ultralytics; print('OK')"
```

Nếu in ra "OK" là đã cài đặt thành công!

---

## Lưu ý khi làm việc với Git:

### Files không nên commit (đã có trong .gitignore):
- `yolo_env/` - Virtual environment
- `*.pt` (trừ model nhỏ < 10MB) - Model files quá lớn
- `runs/` - Kết quả training
- `*.mp4`, `*.avi` - Video outputs
- `*.csv` - Log files
- `__pycache__/`, `*.pyc` - Python cache

### Files nên commit:
- `*.py` - Code Python
- `*.yaml` - Config files
- `requirements.txt` - Dependencies
- `README.md`, `SETUP_GUIDE.md` - Documentation
- `.gitignore` - Git ignore rules

### Cập nhật code mới từ Git:
```cmd
git pull origin main
# Nếu có thêm dependencies mới:
pip install -r requirements.txt
```
