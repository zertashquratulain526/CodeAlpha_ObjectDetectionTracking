"""
detect_track.py
---------------
CodeAlpha AI Internship - Task 4: Object Detection and Tracking

Pipeline for every video frame (webcam or video file):
    OpenCV frame  ->  YOLOv8 detection  ->  SORT tracking  ->  boxes + labels + track IDs on screen

Examples
--------
    python detect_track.py                       # default webcam (press  q  to quit)
    python detect_track.py --source video.mp4    # video file
    python detect_track.py --source video.mp4 --output result.mp4 --no-display
    python detect_track.py --classes person car  # track only some classes
"""

import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from sort_tracker import Sort


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection (YOLOv8) and tracking (SORT)")
    p.add_argument("--source", default="0", help="0 = webcam, or path to a video file (default: 0)")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (n=fastest ... x=most accurate)")
    p.add_argument("--conf", type=float, default=0.35, help="minimum detection confidence")
    p.add_argument("--classes", nargs="*", default=None, help="only keep these class names, e.g. person car")
    p.add_argument("--output", default=None, help="save the annotated video to this file (e.g. result.mp4)")
    p.add_argument("--no-display", action="store_true", help="do not open a window (useful on servers)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    return p.parse_args()


def color_for_id(track_id: int):
    """A fixed, bright colour per track id (BGR)."""
    rng = np.random.RandomState(track_id * 7919)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_track(frame, box, track_id, label, score):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = color_for_id(track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"ID {track_id} {label} {score:.2f}"
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(y1 - th - base - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    # ------------------------------------------------------------ video input (OpenCV)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

    # ------------------------------------------------------------ model + tracker
    model = YOLO(args.model)                       # pre-trained on the COCO dataset (80 classes)
    names = model.names
    allowed = None
    if args.classes:
        lookup = {v.lower(): k for k, v in names.items()}
        unknown = [c for c in args.classes if c.lower() not in lookup]
        if unknown:
            raise SystemExit(f"Unknown class name(s): {unknown}. Available: {sorted(lookup)}")
        allowed = [lookup[c.lower()] for c in args.classes]

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    seen_ids = defaultdict(set)                    # class name -> set of track ids ever seen

    frame_no, t_prev, fps = 0, time.time(), 0.0
    print("Running... press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        # -------------------------------------------------------- detection (YOLO)
        result = model.predict(frame, conf=args.conf, classes=allowed, verbose=False)[0]
        boxes = result.boxes
        if len(boxes):
            dets = np.hstack([
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy()[:, None],
                boxes.cls.cpu().numpy()[:, None],
            ])
        else:
            dets = np.empty((0, 6))

        # -------------------------------------------------------- tracking (SORT)
        tracks = tracker.update(dets)

        # -------------------------------------------------------- drawing
        for x1, y1, x2, y2, tid, cid, score in tracks:
            label = names[int(cid)]
            seen_ids[label].add(int(tid))
            draw_track(frame, (x1, y1, x2, y2), int(tid), label, score)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6)) if frame_no > 1 else 0.0
        t_prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  Tracked now: {len(tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Object Detection & Tracking (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frame_no >= args.max_frames:
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {args.output}")
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_no} frames. Unique objects tracked:")
    for label, ids in sorted(seen_ids.items()):
        print(f"  {label:15s} {len(ids)}")


if __name__ == "__main__":
    main()
