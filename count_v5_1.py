import cv2
import os
import threading
import time
from ultralytics import solutions

SOURCE = "rtsp://admin:admin@192.168.0.100/onvif-media/media.amp?streamprofile=Profile2&audio=0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|"
    "framedrop;1|probesize;32|analyzeduration;0|max_delay;0"
)

MODEL_PATH = r"steel_model_v112.pt"

# Region mặc định — sẽ được vẽ lại khi nhấn R
DEFAULT_REGION = [(420, 307), (969, 307), (969, 452), (420, 452)]


# ── RTSP Stream ───────────────────────────────────────────────
class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.cap = None
        self._connect()

    def _connect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("reconnecting...")
                self._connect()
                time.sleep(1)
                continue
            with self.lock:
                self.frame = frame

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


# ── Vẽ region bằng chuột ─────────────────────────────────────
class RegionDrawer:
    def __init__(self):
        self.points = []
        self.drawing = False
        self.cursor = (0, 0)
        self.done = False

    def mouse_cb(self, event, x, y, flags, param):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self.drawing:
            self.points.append((x, y))

    def start(self):
        self.points = []
        self.drawing = True
        self.done = False
        print(" Click để thêm điểm | ENTER để xác nhận | ESC để hủy")

    def confirm(self):
        if len(self.points) >= 3:
            self.drawing = False
            self.done = True
            print(f" Region mới: {self.points}")
            return self.points
        return None

    def cancel(self):
        self.drawing = False
        self.done = False
        self.points = []

    def draw_preview(self, frame):
        if not self.drawing:
            return frame
        pts = self.points + [self.cursor]
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i+1], (0, 255, 255), 1)
        for p in self.points:
            cv2.circle(frame, p, 5, (0, 255, 255), -1)
        cv2.putText(frame,
                    f"Click them diem ({len(self.points)}) | ENTER xac nhan | ESC huy",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame


# ── Tạo ObjectCounter ─────────────────────────────────────────
def make_counter(region):
    return solutions.ObjectCounter(
        model=MODEL_PATH,
        region=region,
        show=False,
        line_width=2,
        show_conf=False,
        show_labels=False,
        conf=0.3,
        iou=0.5,
        max_hist=50,
        tracker="custom_botsort.yaml",
    )


# ── Main ──────────────────────────────────────────────────────
stream = RTSPStream(SOURCE)
assert stream.is_opened(), "Không kết nối được camera"

w, h, fps = stream.get_props()

writer = cv2.VideoWriter(
    "steel_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (w, h)
)

current_region = list(DEFAULT_REGION)
counter = make_counter(current_region)
drawer  = RegionDrawer()

cv2.namedWindow("Steel Counter")
cv2.setMouseCallback("Steel Counter", drawer.mouse_cb)

stream.start()
while stream.read() is None:
    time.sleep(0.05)

total = 0
print(" Chạy... | R=vẽ lại region | Q=thoát")

while True:
    im0 = stream.read()
    if im0 is None:
        time.sleep(0.01)
        continue

    # Nếu đang vẽ region → hiển thị preview, không đếm
    if drawer.drawing:
        preview = im0.copy()
        preview = drawer.draw_preview(preview)
        # Vẽ region hiện tại mờ
        if len(current_region) >= 3:
            import numpy as np
            pts = np.array(current_region, np.int32)
            cv2.polylines(preview, [pts], True, (100, 100, 100), 1)
        cv2.imshow("Steel Counter", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:   # Enter → xác nhận
            new_region = drawer.confirm()
            if new_region:
                current_region = new_region
                counter = make_counter(current_region)
                total = 0
                print(f"🔄 Đã cập nhật region, reset đếm về 0")
        elif key == 27:  # ESC → hủy
            drawer.cancel()
            print(" Hủy vẽ region")
        continue

    # Detect + đếm bình thường
    results      = counter(im0)
    annotated_im = results.plot_im.copy()

    in_c  = results.in_count
    out_c = results.out_count
    total = in_c + out_c

    # Số đếm
    cv2.putText(annotated_im, f"TOTAL: {total}",
                (w // 2 - 100, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

    # Hướng dẫn
    cv2.putText(annotated_im, "R=ve lai region  Q=thoat",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    writer.write(annotated_im)
    cv2.imshow("Steel Counter", annotated_im)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        drawer.start()   # bắt đầu vẽ region mới

stream.stop()
writer.release()
cv2.destroyAllWindows()
print(f"\n✅ Tổng: {total} thanh thép")
print(f"   Region cuối: {current_region}")