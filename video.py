import cv2
import os
from ultralytics import YOLO
from pathlib import Path

def annotate_video(
    video_path,
    weights_path,
    output_path,
    class_names,
    colors,
    confidence_threshold=0.6,
    imgsz=640
):
    # Load model
    model = YOLO(weights_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames at {fps} FPS...")

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(output_path.parent, exist_ok=True)
    out_vid = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Annotate frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=imgsz, conf=confidence_threshold)[0]

        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            xc, yc, bw, bh = box.xywh[0]
            x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
            x2, y2 = int(xc + bw / 2), int(yc + bh / 2)

            color = colors[cls % len(colors)]
            label = f"{class_names[cls]} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_vid.write(frame)

    cap.release()
    out_vid.release()
    print(f"Saved annotated video to: {output_path}")

if __name__ == "__main__":
    # Paths
    VIDEO_PATH = Path('/datashare/HW1/ood_video_data/surg_1.mp4')
    WEIGHTS_PATH = Path('/home/student/HW1/best.pt')
    OUTPUT_PATH = Path('/home/student/HW1/ood_preds/annotated_ood.mp4')

    # Class names and colors
    CLASS_NAMES = ['Empty', 'Tweezers', 'Needle Driver']
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    # Run annotation
    annotate_video(
        video_path=VIDEO_PATH,
        weights_path=WEIGHTS_PATH,
        output_path=OUTPUT_PATH,
        class_names=CLASS_NAMES,
        colors=COLORS,
        confidence_threshold=0.6
    )
