"""
Visual Re-Identification Module — HSV Histogram Fingerprinting
===============================================================
Stores a visual fingerprint (color histogram + aspect ratio + area)
of the tracked target, and compares it against candidate detections
to re-identify the target after occlusion or ID switch.

Features:
    - HSV color histogram with temporal averaging (10-frame buffer)
    - Aspect ratio and area comparison for shape consistency
    - Weighted multi-feature scoring (60% color + 20% shape + 20% size)
    - Bhattacharyya distance metric for histogram comparison
    - CPU-only — runs in <0.5ms per comparison

Author: ByIbos
License: MIT
"""

import cv2
import numpy as np


class TargetReID:
    """
    Visual Re-Identification using color histogram fingerprinting.

    When a tracked target is lost (e.g., its tracking ID changes after
    occlusion), this module compares the visual appearance of new
    detections against the stored fingerprint to find the same target.

    The fingerprint consists of:
        - HSV color histogram (30×32 bins, temporally averaged)
        - Aspect ratio (width/height)
        - Average bounding box area

    Args:
        similarity_threshold: Minimum combined score (0-1) required
            to consider a detection as a match (default: 0.55).

    Example:
        >>> reid = TargetReID(similarity_threshold=0.55)
        >>> reid.update_fingerprint(frame, (x1, y1, x2, y2))
        >>> score = reid.compare(frame, (x1_new, y1_new, x2_new, y2_new))
        >>> if score > 0.55:
        ...     print("Same target found!")
    """

    def __init__(self, similarity_threshold=0.55):
        self.similarity_threshold = similarity_threshold

        # Stored fingerprint
        self.hist = None           # HSV color histogram
        self.aspect_ratio = None   # Width / height ratio
        self.avg_area = None       # Average bounding box area
        self.template = None       # Last cropped image (for template matching)

        # Histogram buffer for temporal averaging
        self.hist_buffer = []
        self.max_hist_buffer = 10

    def update_fingerprint(self, frame, box):
        """
        Update the stored fingerprint with the current target crop.

        Should be called every frame while the target is being tracked
        (locked state). The histogram is averaged over the last 10 frames
        for stability.

        Args:
            frame: Full video frame (BGR numpy array).
            box: Bounding box as (x1, y1, x2, y2) — pixel coordinates.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]

        # Boundary clamping
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        crop = frame[y1:y2, x1:x2]
        self.template = crop.copy()

        # HSV histogram — captures color distribution
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        self.hist_buffer.append(hist)
        if len(self.hist_buffer) > self.max_hist_buffer:
            self.hist_buffer.pop(0)

        # Temporal average for stability
        self.hist = np.mean(self.hist_buffer, axis=0).astype(np.float32)

        # Geometric features
        self.aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
        self.avg_area = (x2 - x1) * (y2 - y1)

    def compare(self, frame, box):
        """
        Compare a candidate bounding box against the stored fingerprint.

        Uses a weighted combination of:
            - HSV histogram similarity (Bhattacharyya distance) — 60%
            - Aspect ratio similarity — 20%
            - Area ratio similarity — 20%

        Args:
            frame: Full video frame (BGR numpy array).
            box: Candidate bounding box as (x1, y1, x2, y2).

        Returns:
            float: Similarity score in [0.0, 1.0] (1.0 = identical).
        """
        if self.hist is None:
            return 0.0

        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0

        crop = frame[y1:y2, x1:x2]

        # HSV histogram of candidate
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # Bhattacharyya distance (lower = more similar)
        hist_score = 1.0 - cv2.compareHist(self.hist, hist, cv2.HISTCMP_BHATTACHARYYA)

        # Aspect ratio similarity
        ar = (x2 - x1) / max(1, (y2 - y1))
        ar_diff = abs(ar - self.aspect_ratio) / max(self.aspect_ratio, 0.1)
        ar_score = max(0, 1.0 - ar_diff)

        # Area similarity
        area = (x2 - x1) * (y2 - y1)
        area_ratio = min(area, self.avg_area) / max(area, self.avg_area, 1)

        # Weighted combination
        score = hist_score * 0.6 + ar_score * 0.2 + area_ratio * 0.2
        return score

    def find_best_match(self, frame, boxes, track_ids, exclude_id=None):
        """
        Find the detection that best matches the stored fingerprint.

        Iterates through all provided bounding boxes, computes the
        similarity score for each, and returns the best match if it
        exceeds the similarity threshold.

        Args:
            frame: Full video frame (BGR numpy array).
            boxes: (N, 4) numpy array of bounding boxes.
            track_ids: (N,) numpy array of tracker-assigned IDs.
            exclude_id: Optional ID to skip (e.g., the current target ID).

        Returns:
            tuple: (best_id, best_score) if a match is found,
            or (None, 0.0) if no match exceeds the threshold.
        """
        if self.hist is None or boxes is None or len(boxes) == 0:
            return None, 0.0

        best_id = None
        best_score = 0.0

        for box, tid in zip(boxes, track_ids):
            tid = int(tid)
            if tid == exclude_id:
                continue

            score = self.compare(frame, box)
            if score > best_score:
                best_score = score
                best_id = tid

        if best_score >= self.similarity_threshold:
            return best_id, best_score

        return None, best_score

    def reset(self):
        """Clear the stored fingerprint and all internal buffers."""
        self.hist = None
        self.aspect_ratio = None
        self.avg_area = None
        self.template = None
        self.hist_buffer = []
