# Hướng dẫn cài đặt project

## Bước 1: Clone project từ Git

```cmd
git clone <URL_REPOSITORY>
cd <TÊN_FOLDER_PROJECT>
```

Hoặc tải .zip , giản nén , mở ternimal trong thư mục đã giải nén.

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

![1772616904414](images/SETUP_GUIDE/1772616904414.png)![alt text](image-1.png

## Bước 5: Tải model file (nếu chưa có)

Model file `steel_model_v112.pt` .

## Bước 6: Chỉnh sửa config

Trong file `count_v5.py`, sửa lại URL camera RTSP:

```python
SOURCE = "rtsp://admin:admin@192.168.0.100/..."
```

Trong file `count_v3.py`, sửa lại Đường dẫn video:

```python
SOURCE     = r"C:\Users\chaum\Videos\count steel\video001.mp4"
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
