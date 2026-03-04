import cv2
import os
import threading
import time
from ultralytics import solutions

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0"

SOURCE = "rtsp://admin:bkict@2025@192.168.2.11:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
MODEL_PATH = r"steel_model_v112.pt"
ORIGINAL_REGION = [(420, 307), (969, 307), (969, 452), (420, 452)]
RESIZE_WIDTH = 640

class LowLatencyCounter:
    def __init__(self, src, model_path, region):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.w_orig = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h_orig = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_source = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        
        # Tính toán scale và vùng đếm mới
        self.scale = RESIZE_WIDTH / self.w_orig
        self.w_new = RESIZE_WIDTH
        self.h_new = int(self.h_orig * self.scale)
        self.new_region = [(int(x * self.scale), int(y * self.scale)) for (x, y) in region]

        # Khởi tạo Object Counter
        self.counter = solutions.ObjectCounter(
            model=model_path,
            region=self.new_region,
            show=False,
            line_width=2,
            conf=0.3,
            iou=0.5,
            tracker="custom_botsort.yaml"
        )

        self.frame = None
        self.stopped = False
        self.total_count = 0

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret = self.cap.grab() 
            if ret:
                # Retrieve chỉ khi cần thiết để giảm tải CPU
                _, self.frame = self.cap.retrieve()
            else:
                time.sleep(0.01)

    def run(self):
        writer = cv2.VideoWriter("steel_output_v5_fast.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), self.fps_source, (self.w_new, self.h_new))

        print("Đang bắt đầu xử lý luồng... Nhấn 'q' để thoát.")

        while not self.stopped:
            if self.frame is not None:
                # Tạo bản sao frame để tránh xung đột thread khi đang vẽ
                img = self.frame.copy()
                
                # 1. Resize khung hình
                im0_small = cv2.resize(img, (self.w_new, self.h_new))

                # 2. Đưa qua bộ đếm YOLO (SỬA LỖI Ở ĐÂY)
                # Hàm count() sẽ trả về ảnh đã được vẽ box và region
                processed_im = self.counter.count(im0_small)

                # 3. Lấy thông tin đếm
                self.total_count = self.counter.in_count + self.counter.out_count
                
                # 4. Hiển thị thông tin lên màn hình
                cv2.putText(processed_im, f"TOTAL: {self.total_count}", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Fast YOLO Counter (UDP + Threading)", processed_im)
                writer.write(processed_im)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stopped = True
                break
        
        self.cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"\nKết quả cuối cùng: {self.total_count}")

# Thực thi
if __name__ == "__main__":
    app = LowLatencyCounter(SOURCE, MODEL_PATH, ORIGINAL_REGION).start()
    app.run()