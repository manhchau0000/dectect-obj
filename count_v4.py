import cv2
import os
from ultralytics import solutions

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|nobuffer|probesize;32|analyzeduration;0"

SOURCE = "rtsp://admin:bkict@2025@192.168.2.11:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
MODEL_PATH = r"steel_model_v112.pt"

ORIGINAL_REGION = [(420, 307), (969, 307), (969, 452), (420, 452)]

# ORIGINAL_REGION = [(180, 509), (1749, 496), (1749, 912), (163, 886)]

RESIZE_WIDTH = 640  

cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
assert cap.isOpened(), "Error reading video file"

w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS) or 30)

scale = RESIZE_WIDTH / w_orig
w_new = RESIZE_WIDTH
h_new = int(h_orig * scale)

NEW_REGION = [(int(x * scale), int(y * scale)) for (x, y) in ORIGINAL_REGION]

writer = cv2.VideoWriter("steel_output_v5_fast.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_new, h_new))

counter = solutions.ObjectCounter(
    model=MODEL_PATH,
    region=NEW_REGION,
    show=False,
    line_width=2,
    show_conf=False,
    show_labels=False,
    conf=0.3,  # độ tin cậy tối thiểu 
    iou=0.5,     # Tăng IOU để tracker bám đuôi tốt hơn
    max_hist=50, #lưu lịch sử tracking lâu hơn để tránh mất ID khi đối tượng tạm thời biến mất hoặc bị che khuất,
    tracker="custom_botsort.yaml",   
)

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    im0_small = cv2.resize(im0, (w_new, h_new))

    results = counter(im0_small)

    annotated_im = results.plot_im

    in_c  = results.in_count
    out_c = results.out_count
    total = in_c + out_c

    cv2.putText(annotated_im, f"TOTAL: {total}", (w_new // 2 - 100, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    writer.write(annotated_im)
    cv2.imshow("Fast YOLO26 Counter (640px)", annotated_im)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"\nok, found: {total}")