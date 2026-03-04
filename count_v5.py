import cv2
import os
import threading
import time
from ultralytics import solutions


# ──────────────────────────────────────────────────────────────────────────────
# cam fidus

SOURCE = "rtsp://admin:admin@192.168.0.100/onvif-media/media.amp?streamprofile=Profile2&audio=0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;udp|"  
    "fflags;nobuffer|"
    "flags;low_delay|"
    "framedrop;1|"
    "probesize;32|"
    "analyzeduration;0|"
    "max_delay;0"
)

# ──────────────────────────────────────────────────────────────────────────────


# cam hik

# SOURCE = "rtsp://admin:bkict@2025@192.168.2.11:554/cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
#     "rtsp_transport;tcp|"
#     "fflags;nobuffer|"
#     "flags;low_delay|"
#     "framedrop;1|"
#     "probesize;32|"
#     "analyzeduration;0"
# )

MODEL_PATH = r"steel_model_v112.pt"

ORIGINAL_REGION = [(420, 307), (969, 307), (969, 452), (420, 452)]
RESIZE_WIDTH = 640


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
# ──────────────────────────────────────────────────────────────────────────────


stream = RTSPStream(SOURCE)
assert stream.is_opened(), "Error reading video file"

w_orig, h_orig, fps = stream.get_props()

scale  = RESIZE_WIDTH / w_orig
w_new  = RESIZE_WIDTH
h_new  = int(h_orig * scale)

NEW_REGION = [(int(x * scale), int(y * scale)) for (x, y) in ORIGINAL_REGION]

writer = cv2.VideoWriter(
    "steel_output_v5_fast.mp4",
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

stream.start()
while stream.read() is None:
    time.sleep(0.05)

total = 0

while True:
    im0 = stream.read()
    if im0 is None:
        time.sleep(0.01)
        continue

    im0_small = cv2.resize(im0, (w_new, h_new))

    results = counter(im0_small)

    annotated_im = results.plot_im

    in_c  = results.in_count
    out_c = results.out_count
    total = in_c + out_c

    cv2.putText(annotated_im, f"TOTAL: {total}", (w_new // 2 - 100, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    writer.write(annotated_im)
    cv2.imshow("press 'q' to exit", annotated_im)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.stop()
writer.release()
cv2.destroyAllWindows()

print(f"\nok, found: {total}")