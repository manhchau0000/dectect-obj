import cv2
import os
import threading
import time

# Cấu hình FFmpeg tối ưu trễ
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0"

SOURCE = "rtsp://admin:bkict@2025@192.168.2.11:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

class CameraStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Giảm buffer xuống tối thiểu
        self.status, self.frame = False, None
        self.stopped = False

    def start(self):
        # Chạy thread để liên tục đọc frame mới nhất
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            # Liên tục grab frame để dọn dẹp buffer của camera
            ret = self.cap.grab()
            if ret:
                self.status, self.frame = self.cap.retrieve()
            else:
                time.sleep(0.01)

    def get_frame(self):
        return self.status, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Bắt đầu stream
stream = CameraStream(SOURCE).start()

while True:
    ret, frame = stream.get_frame()
    
    if ret and frame is not None:
        cv2.imshow("Realtime Low Latency", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stream.stop()
        break

cv2.destroyAllWindows()