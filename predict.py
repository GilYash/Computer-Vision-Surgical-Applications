import cv2
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# Configuration 
# ────────────────────────────────────────────────
WEIGHTS_PATH = Path('/home/student/HW1/weights.pt')
VIDEO_PATH   = Path('/datashare/HW1/ood_video_data/surg_1.mp4')
FRAME_INDEX  = 150
CONF_THRESH  = 0.5

CLASS_NAMES = ['Empty', 'Tweezers', 'Needle Driver']
COLORS      = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
model = YOLO(str(WEIGHTS_PATH))

# ────────────────────────────────────────────────
# Read the single frame
# ────────────────────────────────────────────────
cap = cv2.VideoCapture(str(VIDEO_PATH))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if FRAME_INDEX >= total:
    cap.release()
    raise ValueError(f"Frame index {FRAME_INDEX} exceeds total frames ({total})")

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to read frame")

# ────────────────────────────────────────────────
# Run YOLO prediction on that frame
# ────────────────────────────────────────────────
results = model.predict(frame, imgsz=640, conf=CONF_THRESH)[0]

# ────────────────────────────────────────────────
# Draw all bounding boxes + labels directly onto the BGR image
# ────────────────────────────────────────────────
for box in results.boxes:
    cls  = int(box.cls)
    conf = float(box.conf)
    xc, yc, bw, bh = box.xywh[0]
    x1 = int(xc - bw/2);  y1 = int(yc - bh/2)
    x2 = int(xc + bw/2);  y2 = int(yc + bh/2)

    color = COLORS[cls % len(COLORS)]
    label = f"{CLASS_NAMES[cls]} ({conf:.2f})"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ────────────────────────────────────────────────
# Convert BGR → RGB and show inline with Matplotlib
# ────────────────────────────────────────────────
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(rgb_frame)
plt.axis('off')
plt.title(f"Frame {FRAME_INDEX} with YOLO Predictions")
plt.show()
