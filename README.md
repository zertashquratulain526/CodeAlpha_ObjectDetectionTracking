# 🎯 CodeAlpha_ObjectDetectionTracking

**CodeAlpha Artificial Intelligence Internship – Task 4: Object Detection and Tracking**

Real-time object **detection** with a pre-trained **YOLOv8** model and object **tracking** with the **SORT** algorithm. Every object gets a bounding box, a class label, a confidence score and a **unique tracking ID** that stays the same while the object moves through the video.

## Pipeline

```
Webcam / video file (OpenCV)
        │  frame by frame
        ▼
YOLOv8 (pre-trained on COCO, 80 classes)  →  bounding boxes + class + confidence
        ▼
SORT tracker  (Kalman filter + Hungarian algorithm on IoU)  →  same object = same ID
        ▼
Draw boxes, labels, IDs, FPS  →  show in a window and/or save to a video file
```

### What is SORT?
*Simple Online and Realtime Tracking* works in three steps for each frame:
1. **Predict** – a Kalman filter predicts where each tracked object should be now.
2. **Associate** – detections are matched with the predicted boxes using the Hungarian algorithm; the cost is the overlap (IoU) between boxes. Matching is done per class.
3. **Update** – matched tracks are corrected, unmatched detections start new tracks, and tracks that are lost for 20 frames are removed.

`sort_tracker.py` contains my own implementation (only numpy + scipy needed).

## Project structure

```
CodeAlpha_ObjectDetectionTracking/
├── detect_track.py      # main program: video input, YOLO detection, drawing, output
├── sort_tracker.py      # SORT tracker (Kalman filter + Hungarian matching)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1) live webcam  (press  q  to quit)
python detect_track.py

# 2) a video file, and save the result
python detect_track.py --source my_video.mp4 --output result.mp4

# 3) only track some classes
python detect_track.py --classes person car

# 4) no window (e.g. on a server / Google Colab)
python detect_track.py --source my_video.mp4 --output result.mp4 --no-display
```

The YOLO weights (`yolov8n.pt`, ~6 MB) are downloaded automatically the first time you run the program.

### Options

| Option | Meaning | Default |
|---|---|---|
| `--source` | `0` for webcam or path of a video file | `0` |
| `--model` | `yolov8n.pt` (fast) … `yolov8s.pt`, `yolov8m.pt` (more accurate) | `yolov8n.pt` |
| `--conf` | minimum detection confidence | `0.35` |
| `--classes` | only keep these class names | all |
| `--output` | save the annotated video | – |
| `--no-display` | do not open a window | off |
| `--max-frames` | stop after N frames | no limit |

At the end the program prints how many **unique** objects of each class were tracked.

## Technologies

Python · OpenCV · Ultralytics YOLOv8 · NumPy · SciPy
