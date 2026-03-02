import cv2
import numpy as np
from ultralytics import YOLO

# ── Cấu hình ─────────────────────────────────────────────────
SOURCE     = r"C:\Users\chaum\Videos\Training\1\lv_0_20231110153119_1.mp4"
MODEL_PATH = "runs/detect/train/weights/best.pt"
CONF       = 0.20

# Vùng polygon — chỉnh tọa độ cho khớp với video của bạn
REGION = np.array([[180, 509], [1749, 496], [1757, 912], [163, 886]], np.int32)
# ─────────────────────────────────────────────────────────────

def in_polygon(px, py, poly):
    """Kiểm tra điểm (px,py) có nằm trong polygon không."""
    result = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
    return result >= 0

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(SOURCE)
assert cap.isOpened(), f"Không mở được: {SOURCE}"

w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

writer = cv2.VideoWriter("steel_output.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))

total        = 0
tracked      = set()   # ID đã đếm rồi
prev_inside  = {}      # trạng thái frame trước

print("▶ Đang chạy... nhấn Q để dừng")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track
    results = model.track(frame, persist=True, verbose=False,
                          conf=CONF, iou=0.3, tracker="botsort.yaml")

    annotated = frame.copy()

    if (results[0].boxes is not None and
            results[0].boxes.id is not None):

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, obj_id, cf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            inside = in_polygon(cx, cy, REGION)

            # Đếm khi vật thể ĐI VÀO vùng lần đầu
            if inside and not prev_inside.get(obj_id, False):
                if obj_id not in tracked:
                    tracked.add(obj_id)
                    total += 1

            prev_inside[obj_id] = inside

            # Màu box: vàng=trong vùng, tím=ngoài vùng
            color = (0, 255, 128) if inside else (255, 50, 200)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{obj_id} {cf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1)

    # Vẽ vùng polygon
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [REGION], (180, 0, 200))
    cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
    cv2.polylines(annotated, [REGION], True, (255, 0, 255), 2)

    # Số đếm to ở giữa
    txt = str(total)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3.5, 5)
    cv2.putText(annotated, txt,
                (w // 2 - tw // 2, 120),
                cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 0), 5)

    writer.write(annotated)
    cv2.imshow("Steel Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"\n✅ Tổng: {total} steel  |  Video: steel_output.mp4")