import cv2
from ultralytics import solutions

SOURCE     = r"C:\Users\chaum\Videos\count steel\video001.mp4"
MODEL_PATH = r"steel_model_v112.pt"
ORIGINAL_REGION = [(180, 509), (1749, 496), (1749, 912), (163, 886)]
RESIZE_WIDTH = 640

cap = cv2.VideoCapture(SOURCE)
assert cap.isOpened(), "Error reading video file"

w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS) or 30)

scale  = RESIZE_WIDTH / w_orig
w_new  = RESIZE_WIDTH
h_new  = int(h_orig * scale)

NEW_REGION = [(int(x * scale), int(y * scale)) for (x, y) in ORIGINAL_REGION]

writer = cv2.VideoWriter("result count v3.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_new, h_new))

counter = solutions.ObjectCounter(
    model=MODEL_PATH,
    region=NEW_REGION,
    show=False,
    line_width=2,
    conf=0.3,
    iou=0.5,
    max_hist=50,
    tracker="custom_botsort.yaml",  
    show_conf=False,     
    show_labels=True,   
    show_in=False,       
    show_out=False, 
)

while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        break

    im0_small = cv2.resize(im0, (w_new, h_new))
    results   = counter(im0_small)

    annotated_im = results.plot_im
    in_c  = results.in_count
    out_c = results.out_count
    total = in_c + out_c

    cv2.putText(annotated_im, f"TOTAL: {total}", (w_new // 2 - 100, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(annotated_im,
                    "BKICT", 
                    (20, h_new - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,           
                    (255, 255, 255), 
                    2)         

    writer.write(annotated_im)
    cv2.imshow("Fast YOLO Counter", annotated_im)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"\nok, found: {total}")