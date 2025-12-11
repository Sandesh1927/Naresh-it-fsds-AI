import os
import random
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- CONFIG ----------
VIDEO_PATH = r"C:\Users\sande\OneDrive\Desktop\A_VS_CODE\Yolo\vecteezy_busy-traffic-on-the-highway_6434705.mp4"
WEIGHTS_PATH = "weights/yolov8n.pt"
COCO_TXT = r"C:\Users\sande\OneDrive\Desktop\A_VS_CODE\Ultralytics_Yolo\coco.txt"
# ----------------------------

# load class names
if not os.path.isfile(COCO_TXT):
    print(f"Class file not found: {COCO_TXT}")
    class_list = [str(i) for i in range(80)]
else:
    with open(COCO_TXT, "r", encoding="utf-8", errors="ignore") as f:
        class_list = [ln.strip() for ln in f.read().splitlines() if ln.strip() != ""]

# generate colors
detection_colors = []
for _ in range(len(class_list)):
    detection_colors.append(tuple([random.randint(0, 255) for _ in range(3)][::-1]))  # BGR

# load YOLO model (make sure path exists)
if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(f"YOLO weights not found: {WEIGHTS_PATH}")
model = YOLO(WEIGHTS_PATH, "v8")

# helper to try open video with a given backend
def try_open(path_or_index, backend=None):
    if backend is None:
        cap = cv2.VideoCapture(path_or_index)
    else:
        cap = cv2.VideoCapture(path_or_index, backend)
    return cap

# Try to open the video file; print diagnostics
print("Trying to open video:", VIDEO_PATH)
if not os.path.exists(VIDEO_PATH):
    print("-> File does not exist at the path above. Check the path, OneDrive sync, or filename.")
else:
    print("-> File exists. Size (bytes):", os.path.getsize(VIDEO_PATH))

# Attempt opening with default first
cap = try_open(VIDEO_PATH)

# If failed, try with FFmpeg backend (Windows)
if not cap.isOpened():
    print("Default open failed. Trying with FFMPEG backend...")
    try:
        cap = try_open(VIDEO_PATH, cv2.CAP_FFMPEG)
    except Exception as e:
        print("CAP_FFMPEG attempt error:", e)

# If still failed, try CAP_ANY
if not cap.isOpened():
    print("Trying CAP_ANY backend...")
    try:
        cap = try_open(VIDEO_PATH, cv2.CAP_ANY)
    except Exception as e:
        print("CAP_ANY attempt error:", e)

# If still not opened, try camera indices 0..2 as a last resort
if not cap.isOpened():
    print("All file attempts failed. Trying local camera indices 0..2 as fallback.")
    for idx in range(3):
        c = try_open(idx, cv2.CAP_DSHOW)  # use DirectShow for cameras on Windows
        if c.isOpened():
            print(f"Opened camera index {idx} as fallback.")
            cap = c
            break
        else:
            print(f"Camera index {idx} not available.")
    else:
        print("No camera available and could not open video file. Exiting.")
        exit(1)

print("Capture opened successfully.")

frame_wid = 640
frame_hyt = 480

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # optional resize - uncomment if you want smaller frames
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # YOLO predict: pass the frame directly (Ultralytics accepts np.ndarray)
    results = model.predict(source=[frame], conf=0.45, save=False)

    # results[0] is the result for this frame
    r = results[0]
    # Convert to numpy to check if any boxes present
    try:
        DP = r.numpy()
    except Exception:
        DP = None

    if DP is not None and len(DP) != 0 and len(r.boxes) > 0:
        for i, box in enumerate(r.boxes):
            clsID = int(box.cls.numpy()[0])
            conf = float(box.conf.numpy()[0])
            bb = box.xyxy.numpy()[0]  # [x1,y1,x2,y2]

            x1, y1, x2, y2 = map(int, bb.tolist())

            color = detection_colors[clsID % len(detection_colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"{class_list[clsID] if clsID < len(class_list) else clsID} {round(conf*100, 1)}%"
            cv2.putText(frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("ObjectDetection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
