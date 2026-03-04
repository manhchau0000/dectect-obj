import cv2
import threading
import time

class RTSPStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._connect()

    def _connect(self):
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|"
            "flags;low_delay|analyzeduration;0|probesize;32"
        )
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
                self._connect()  # tự reconnect nếu mất stream
                time.sleep(0.5)
                continue
            with self.lock:
                self.frame = frame  # chỉ giữ frame MỚI NHẤT

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


stream = RTSPStream("rtsp://admin:bkict@2025@192.168.2.11:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")
stream.start()

while True:
    frame = stream.read()
    if frame is None:
        continue

    # result = your_model(frame)
    cv2.imshow("Live", frame)

    if cv2.waitKey(1) == ord('q'):
        break

stream.stop()