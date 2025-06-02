# Computer-Vision-Surgical-Applications


# 🧠 Surgical Instrument Detection with Semi-Supervised Learning

This repository contains code and results from our HW1 project on semi-supervised object detection for surgical videos. We use YOLOv8 to detect instruments in leg suturing procedures under both in-distribution (ID) and out-of-distribution (OOD) settings.

---

## 📌 Project Overview

In this project, we tackled the challenge of object detection in a low-resource medical domain using a small set of labeled images and unlabeled surgical videos. By leveraging pseudo-labeling and strong data augmentations, we trained a YOLOv8 model capable of robustly detecting surgical tools such as Tweezers and Needle Drivers. Our pipeline demonstrates the effectiveness of combining real and synthetic labels in semi-supervised learning settings.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/GilYash/surgical-instrument-detection.git
cd surgical-instrument-detection
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Requires Python 3.10+ and the Ultralytics YOLOv8 library

---

## 🧪 Running Inference

### 🔍 Predict on a single image

```bash
python predict.py
```

- Extracts a specific frame from the OOD video
- Runs prediction using the YOLOv8 model
- Saves annotated output image with bounding boxes

### 🎬 Predict on an entire video

```bash
python video.py
```

- Reads the full OOD video using OpenCV
- Annotates predictions with bounding boxes
- Saves the video as `annotated_ood.mp4`

---

## 📦 Files Included

- `predict.py` — Predict and visualize on one image
- `video.py` — Annotate full video
- `requirements.txt` — Python dependencies
- `weights.pt` — Trained YOLO model weights


---

## 📥 Final Model Weights

Download the final trained model weights:  
📦 **[weights.pt](weights.pt)**

---

## 🎥 OOD Video Demo

Watch the predictions from our model on the OOD surgical video:  
🎬 **[Demo Video (Google Drive)](https://drive.google.com/file/d/19sJoSm_KpbdMjYa0CrBvwYHXBQmuBhe6/view?usp=sharing)**

---

## 👥 Collaborators

- **Yarden** – [@yarden077](https://github.com/yarden077)

---

## 📚 Acknowledgments

Thanks to the Ultralytics team for YOLOv8 and to the course staff for guidance and feedback throughout the project.
