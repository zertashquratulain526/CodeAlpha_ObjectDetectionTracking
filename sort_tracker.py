"""
sort_tracker.py
---------------
A compact implementation of the SORT algorithm
(Simple Online and Realtime Tracking, Bewley et al. 2016).

For every video frame SORT does three things:
  1. PREDICT  – a Kalman filter predicts where every existing track will be now
  2. ASSOCIATE – detections are matched to the predicted boxes with the
                 Hungarian algorithm using IoU (overlap) as the cost
  3. UPDATE   – matched tracks are corrected, unmatched detections start new
                 tracks, tracks that are not seen for a while are deleted

Only numpy and scipy are needed (no filterpy).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between every box in A (N,4) and every box in B (M,4); boxes are x1,y1,x2,y2."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] -> [cx, cy, scale(area), aspect ratio] as a column vector."""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    return np.array([cx, cy, w * h, w / float(h + 1e-9)]).reshape(4, 1)


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Kalman state -> [x1,y1,x2,y2]."""
    w = np.sqrt(max(x[2, 0] * x[3, 0], 1e-9))
    h = x[2, 0] / w
    return np.array([x[0, 0] - w / 2.0, x[1, 0] - h / 2.0, x[0, 0] + w / 2.0, x[1, 0] + h / 2.0])


# ----------------------------------------------------------------------------
# one tracked object
# ----------------------------------------------------------------------------

class KalmanBoxTracker:
    """Kalman filter for one object. State: [cx, cy, s, r, d_cx, d_cy, d_s]."""

    _next_id = 1

    def __init__(self, bbox: np.ndarray, cls_id: int = 0, score: float = 0.0):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0   # constant-velocity model
        self.H = np.eye(4, 7)                              # we only measure cx, cy, s, r

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0                           # velocity is unknown at the start
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)

        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.cls_id = cls_id
        self.score = score
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        if self.x[6, 0] + self.x[2, 0] <= 0:               # area must not become negative
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_bbox(self.x)

    def update(self, bbox: np.ndarray, cls_id: int, score: float):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls_id, self.score = cls_id, score

        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def bbox(self) -> np.ndarray:
        return x_to_bbox(self.x)


# ----------------------------------------------------------------------------
# the SORT tracker
# ----------------------------------------------------------------------------

class Sort:
    def __init__(self, max_age: int = 20, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        max_age       : frames a track survives without a matching detection
        min_hits      : detections needed before a new track is shown (removes flicker)
        iou_threshold : minimum overlap to match a detection with a track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list = []
        self.frame_count = 0
        KalmanBoxTracker._next_id = 1

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        detections : array (N, 6) -> x1, y1, x2, y2, score, class_id   (N may be 0)
        returns    : array (M, 7) -> x1, y1, x2, y2, track_id, class_id, score
        """
        self.frame_count += 1
        detections = np.asarray(detections, dtype=float).reshape(-1, 6)

        # 1) predict new positions of the existing tracks
        predicted = []
        for trk in list(self.tracks):
            box = trk.predict()
            if np.any(np.isnan(box)):
                self.tracks.remove(trk)
            else:
                predicted.append(box)
        predicted = np.array(predicted).reshape(-1, 4)

        # 2) associate detections <-> tracks (Hungarian algorithm on IoU)
        matches, unmatched_dets = self._associate(detections, predicted)

        # 3) update matched tracks, create tracks for the rest
        for det_idx, trk_idx in matches:
            d = detections[det_idx]
            self.tracks[trk_idx].update(d[:4], int(d[5]), d[4])
        for det_idx in unmatched_dets:
            d = detections[det_idx]
            self.tracks.append(KalmanBoxTracker(d[:4], int(d[5]), d[4]))

        # 4) collect output and delete dead tracks
        results = []
        for trk in self.tracks:
            confirmed = trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            if trk.time_since_update < 1 and confirmed:
                results.append([*trk.bbox, trk.id, trk.cls_id, trk.score])
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(results).reshape(-1, 7)

    def _associate(self, detections: np.ndarray, predicted: np.ndarray):
        if len(predicted) == 0:
            return [], list(range(len(detections)))
        if len(detections) == 0:
            return [], []

        iou = iou_matrix(detections[:, :4], predicted)
        # an object can only be matched with a track of the same class
        track_classes = np.array([t.cls_id for t in self.tracks])
        iou[detections[:, 5][:, None] != track_classes[None, :]] = 0.0

        rows, cols = linear_sum_assignment(-iou)
        matches, matched_dets = [], set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_threshold:
                matches.append((r, c))
                matched_dets.add(r)
        unmatched = [i for i in range(len(detections)) if i not in matched_dets]
        return matches, unmatched
