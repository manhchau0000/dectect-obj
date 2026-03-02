import cv2
import numpy as np
from ultralytics import YOLO

SOURCE     = r"C:\Users\chaum\Videos\Training\1\lv_0_20231110153119_1.mp4"
MODEL_PATH = r"C:\Users\chaum\Downloads\best.pt"
CONF       = 0.20
REGION     = np.array([[180, 509], [1749, 496], [1757, 912], [163, 886]], np.int32)

GHOST_TTL       = 45    # tăng lên 45 frame cho tracker chậm
GHOST_DIST      = 100   # tăng lên 100px cho vật thể di chuyển nhanh
MAX_TRACKED_AGE = 3000

def in_polygon(px, py, poly):
    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(SOURCE)
assert cap.isOpened()

w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

writer = cv2.VideoWriter("steel_output.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

total       = 0
tracked     = {}   # obj_id -> frame_idx
prev_inside = {}   # obj_id -> bool
prev_cx     = {}   # obj_id -> (cx, cy)
ghost_zone  = []   # [(cx, cy, ttl, was_counted)]
frame_idx   = 0

def find_ghost(cx, cy):
    """Tìm ghost gần nhất — ưu tiên ghost gần VÀ còn TTL cao."""
    best_i, best_score = -1, float("inf")
    for i, (gx, gy, ttl, _) in enumerate(ghost_zone):
        dist = ((cx-gx)**2 + (cy-gy)**2)**0.5
        if dist < GHOST_DIST:
            # Score = dist / ttl — gần và còn mới thì ưu tiên hơn
            score = dist / (ttl + 1)
            if score < best_score:
                best_score = score
                best_i = i
    return best_i

print("▶ Đang chạy... nhấn Q để dừng")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Dọn RAM mỗi 1000 frame
    if frame_idx % 1000 == 0:
        tracked     = {k: v for k, v in tracked.items()
                       if frame_idx - v < MAX_TRACKED_AGE}
        # Dọn luôn prev_inside và prev_cx cho ID đã quá cũ
        old_ids = [k for k in prev_inside if k not in tracked
                   and k not in {id for id,_,_,_ in
                   [(0,0,0,0)]}]  # giữ nguyên, dọn qua lost_ids bên dưới

    # Giảm TTL ghost
    ghost_zone[:] = [(gx, gy, t-1, c)
                     for gx, gy, t, c in ghost_zone if t > 1]

    results = model.track(frame, persist=True, verbose=False,
                          conf=CONF, iou=0.3, tracker="botsort.yaml")

    annotated = frame.copy()
    current_ids = set()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, obj_id, cf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_ids.add(obj_id)

            inside = in_polygon(cx, cy, REGION)

            # ID mới → kiểm tra ghost
            if obj_id not in prev_inside:
                gi = find_ghost(cx, cy)
                if gi >= 0:
                    _, _, _, was_counted = ghost_zone.pop(gi)
                    if was_counted:
                        # Kế thừa: đánh dấu đã đếm, KHÔNG đếm lại
                        tracked[obj_id] = frame_idx
                        # Kế thừa luôn trạng thái inside trước đó
                        # để không trigger đếm ngay khi vào lại
                        prev_inside[obj_id] = inside

            # Đếm khi vừa vào vùng lần đầu
            if inside and not prev_inside.get(obj_id, False):
                if obj_id not in tracked:
                    tracked[obj_id] = frame_idx
                    total += 1
                    # Flash hiệu ứng khi đếm
                    cv2.circle(annotated, (cx, cy), 25, (0, 255, 0), 3)

            # Làm mới thời gian
            if obj_id in tracked:
                tracked[obj_id] = frame_idx

            prev_inside[obj_id] = inside
            prev_cx[obj_id]     = (cx, cy)

            # Vẽ box
            color = (0, 255, 128) if inside else (255, 50, 200)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"#{obj_id}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ID biến mất → lưu ghost
    for lid in set(prev_inside.keys()) - current_ids:
        if lid in prev_cx:
            ghost_zone.append((*prev_cx[lid], GHOST_TTL, lid in tracked))
        prev_inside.pop(lid, None)
        prev_cx.pop(lid, None)

    # Vẽ polygon
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [REGION], (180, 0, 200))
    cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
    cv2.polylines(annotated, [REGION], True, (255, 0, 255), 2)

    # Số đếm
    txt = str(total)
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3.5, 5)
    cv2.putText(annotated, txt, (w//2 - tw//2, 120),
                cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 0), 5)

    # Debug info nhỏ góc dưới
    cv2.putText(annotated,
                f"tracking:{len(current_ids)}  ghost:{len(ghost_zone)}",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

    writer.write(annotated)
    cv2.imshow("Steel Counter", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"\n✅ Tổng: {total} thanh thép")