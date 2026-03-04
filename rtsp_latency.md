# Cải thiện độ trễ RTSP với OpenCV

## Mục lục
- [Vấn đề gốc](#vấn-đề-gốc)
- [Giải pháp tổng quan](#giải-pháp-tổng-quan)
- [Cấu hình Camera](#cấu-hình-camera)
- [Cấu hình FFmpeg flags](#cấu-hình-ffmpeg-flags)
- [RTSPStream — Thread đọc frame riêng biệt](#rtspstream--thread-đọc-frame-riêng-biệt)
- [Code hoàn chỉnh](#code-hoàn-chỉnh)
- [Tổng hợp mức độ ảnh hưởng](#tổng-hợp-mức-độ-ảnh-hưởng)

---

## Vấn đề gốc

Khi dùng `cap.read()` trực tiếp trong vòng lặp AI, OpenCV và model chạy cùng 1 thread:

```
Thread chính:
[cap.read()] → [AI xử lý 200ms] → [cap.read()] → [AI xử lý 200ms] ...
                ↑ trong lúc AI bận, camera vẫn đẩy frame vào buffer
                ↑ buffer tích lũy 5-10 frame → đang xem quá khứ
```

Nguyên nhân chính gây trễ:

- `cap.read()` bị block trong khi AI đang xử lý
- Buffer mặc định của OpenCV quá lớn (thường 10+ frame)
- Frame cũ tích lũy trong queue → trễ tăng dần theo thời gian
- FFmpeg flags mặc định không tối ưu cho real-time

---

## Giải pháp tổng quan

```
Thread nền:   [read]→[read]→[read]→[read]→[read] (liên tục, chỉ giữ frame mới nhất)
Thread chính:        [AI 200ms]        [AI 200ms] (luôn lấy frame mới nhất)
```

---

## Cấu hình Camera

### Hikvision iDS-TCM403-BI

Vào **Configuration → Video/Audio → Video**, chỉnh Sub-Stream:

| Tham số | Trước | Sau | Lý do |
|---|---|---|---|
| I-Frame Interval | 50 | **25** | = FPS, giảm trễ decode ~1–2s |
| Bitrate Type | Constant | **Variable (VBR)** | Linh hoạt hơn, ít trễ |
| Bitrate | 2048 Kbps | **1024 Kbps** | 720P H.264 chỉ cần 1024 |
| Stream | Main (subtype=0) | **Sub (subtype=1)** | Nhẹ hơn ~40–60% |

> **I-Frame Interval = 50** là thủ phạm chính — camera phải tích lũy 50 frame (~2 giây) mới decode được đầy đủ.

URL sau khi đổi sang sub-stream:
```
rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif
```

### Fidus AfidusVTW-LPR-2MIB

Vào **Network → Streaming**:

| Tham số | Trước | Sau | Lý do |
|---|---|---|---|
| RTP H264 Payload Size | 8192 | **1444** | Tránh packet fragmentation trên LAN |
| Profile | Profile1 | **Profile2** (640×480) | Sub-stream nhẹ hơn |

URL Profile2:
```
rtsp://admin:admin@192.168.0.100/onvif-media/media.amp?streamprofile=Profile2&audio=0
```

---

## Cấu hình FFmpeg flags

```python
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"   # dùng tcp (ổn định) hoặc udp (LAN nhanh hơn)
    "fflags;nobuffer|"      # không buffer phía FFmpeg
    "flags;low_delay|"      # ưu tiên độ trễ thấp
    "framedrop;1|"          # bỏ frame nếu xử lý không kịp
    "probesize;32|"         # giảm thời gian probe stream lúc kết nối
    "analyzeduration;0|"    # không phân tích duration
    "max_delay;0"           # delay tối đa = 0
)

cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buffer chỉ giữ 1 frame
```

> Dùng `rtsp_transport;udp` thay `tcp` khi camera và máy tính cùng mạng LAN để giảm thêm ~20–50ms.

---

## RTSPStream — Thread đọc frame riêng biệt

### Giải thích từng phần

#### `__init__` — Khởi tạo

```python
def __init__(self, url):
    self.url = url
    self.frame = None             # frame mới nhất, ban đầu chưa có gì
    self.lock = threading.Lock()  # khóa để 2 thread không đọc/ghi frame cùng lúc
    self.running = False          # cờ điều khiển thread có chạy không
    self.cap = None
    self._connect()               # mở kết nối RTSP ngay khi tạo object
```

#### `_connect` — Mở / mở lại kết nối

```python
def _connect(self):
    if self.cap:
        self.cap.release()  # đóng kết nối cũ trước (tránh leak)
    self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
```

Tách riêng thành method để có thể gọi lại khi reconnect.

#### `start` — Khởi động thread nền

```python
def start(self):
    self.running = True
    self.thread = threading.Thread(target=self._reader, daemon=True)
    #                                              ↑ daemon=True: thread tự chết khi chương trình thoát
    self.thread.start()
```

#### `_reader` — Trái tim của class

```python
def _reader(self):
    while self.running:
        ret, frame = self.cap.read()  # đọc liên tục
        if not ret:
            print("[WARN] Mất kết nối, đang reconnect...")
            self._connect()           # mất stream → tự reconnect
            time.sleep(1)
            continue
        with self.lock:
            self.frame = frame        # GHI ĐÈ — không queue, chỉ giữ frame MỚI NHẤT
```

Điểm mấu chốt: `self.frame = frame` **ghi đè liên tục** — AI luôn nhận frame mới nhất, không bao giờ xử lý frame cũ tích lũy.

#### `read` — Thread chính gọi để lấy frame

```python
def read(self):
    with self.lock:  # khóa lại trong lúc copy
        return self.frame.copy() if self.frame is not None else None
        #              ↑ .copy() quan trọng: tránh thread nền ghi đè
        #                trong lúc AI đang dùng frame đó
```

Tại sao cần `lock`? Không có lock:

```
Thread nền:   đang ghi frame mới vào self.frame (nửa chừng)
Thread chính: đang đọc self.frame               → đọc được frame hỏng
```

#### `get_props`, `is_opened`, `stop` — Tiện ích

```python
def get_props(self):      # lấy width, height, fps của stream
def is_opened(self):      # kiểm tra kết nối còn sống không
def stop(self):
    self.running = False  # báo thread nền dừng lại
    self.cap.release()    # giải phóng tài nguyên
```

---

## Code hoàn chỉnh

```python
import cv2
import os
import threading
import time
from ultralytics import solutions

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "framedrop;1|"
    "probesize;32|"
    "analyzeduration;0|"
    "max_delay;0"
)

SOURCE    = "rtsp://user:pass@ip/stream?channel=1&subtype=1"
MODEL_PATH = "steel_model_v112.pt"

ORIGINAL_REGION = [(420, 307), (969, 307), (969, 452), (420, 452)]
RESIZE_WIDTH    = 640


class RTSPStream:
    def __init__(self, url):
        self.url     = url
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = False
        self.cap     = None
        self._connect()

    def _connect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        """Liên tục đọc, chỉ giữ frame MỚI NHẤT — tránh queue tích lũy"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Mất kết nối, đang reconnect...")
                self._connect()
                time.sleep(1)
                continue
            with self.lock:
                self.frame = frame  # ghi đè, không queue

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_props(self):
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.cap.get(cv2.CAP_PROP_FPS) or 30),
        )

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ── Khởi tạo ─────────────────────────────────────────────────────────────────
stream = RTSPStream(SOURCE)
assert stream.is_opened(), "Error reading video file"

w_orig, h_orig, fps = stream.get_props()
scale  = RESIZE_WIDTH / w_orig
w_new  = RESIZE_WIDTH
h_new  = int(h_orig * scale)

NEW_REGION = [(int(x * scale), int(y * scale)) for (x, y) in ORIGINAL_REGION]

writer = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (w_new, h_new)
)

counter = solutions.ObjectCounter(
    model=MODEL_PATH,
    region=NEW_REGION,
    show=False,
    line_width=2,
    show_conf=False,
    show_labels=False,
    conf=0.3,
    iou=0.5,
    max_hist=50,
    tracker="custom_botsort.yaml",
)

# Chờ frame đầu tiên trước khi bắt đầu xử lý
stream.start()
print("Đang chờ frame đầu tiên...")
while stream.read() is None:
    time.sleep(0.05)
print("Bắt đầu xử lý...")

total = 0

# ── Vòng lặp chính ───────────────────────────────────────────────────────────
while True:
    im0 = stream.read()
    if im0 is None:
        time.sleep(0.01)
        continue

    im0_small   = cv2.resize(im0, (w_new, h_new))
    results     = counter(im0_small)
    annotated   = results.plot_im

    in_c  = results.in_count
    out_c = results.out_count
    total = in_c + out_c

    cv2.putText(annotated, f"TOTAL: {total}", (w_new // 2 - 100, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    writer.write(annotated)
    cv2.imshow("Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.stop()
writer.release()
cv2.destroyAllWindows()
print(f"\nok, found: {total}")
```

---

## Tổng hợp mức độ ảnh hưởng

| Thay đổi | Giảm trễ |
|---|---|
| Sub-stream thay Main-stream | ~40–60% |
| I-Frame Interval 50 → 25 | ~20–30% |
| Thread đọc frame riêng (RTSPStream) | ~15–20% |
| FFmpeg flags đúng và đầy đủ | ~10–15% |
| UDP thay TCP (LAN) | ~5–10% |
| Payload Size 8192 → 1444 | ~5% |
| Bỏ resize khi sub-stream đã đúng kích thước | ~5% |

> Áp dụng đầy đủ tất cả thay đổi trên, độ trễ thực tế có thể xuống dưới **100–200ms** tùy camera và mạng.
