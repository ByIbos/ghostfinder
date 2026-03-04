"""
Template Tracker — ROI-Focused Visual Search Fallback
======================================================
When the primary detector (YOLO, SSD, etc.) fails to detect the target,
this module searches for the last known visual template within a
constrained Region of Interest (ROI) using OpenCV's matchTemplate.

Features:
    - ROI-constrained search (uses predicted center to limit search area)
    - TM_CCOEFF_NORMED matching for illumination robustness
    - Configurable match threshold and search margin
    - Returns match center, bounding box, score, and search region
    - CPU-only — runs in ~1-3ms per search

Author: ByIbos
License: MIT
"""

import cv2
import numpy as np


class TemplateTracker:
    """
    Fallback tracker using OpenCV template matching.

    Stores the last known visual appearance of the target and searches
    for it within a constrained region when the primary detector fails.
    The search region is centered on the predicted position (from a
    Kalman filter or last known location) to minimize false positives.

    Args:
        match_threshold: Minimum correlation score (0-1) to accept
            a match. Lower = more lenient (default: 0.45).
        search_margin: Extra pixel margin around the predicted center
            to define the search region (default: 150).

    Example:
        >>> tracker = TemplateTracker(match_threshold=0.45)
        >>> tracker.update_template(frame, (x1, y1, x2, y2))
        >>> # When detector fails:
        >>> result = tracker.search(frame, predicted_center=(320, 240))
        >>> if result:
        ...     print(f"Found at {result['center']} with score {result['score']:.2f}")
    """

    def __init__(self, match_threshold=0.45, search_margin=150):
        self.match_threshold = match_threshold
        self.search_margin = search_margin

        self.template = None          # Cropped target image (BGR)
        self.template_gray = None     # Grayscale version for matching
        self.last_center = None       # Last known center (x, y)
        self.last_size = None         # Last known size (w, h)
        self.last_score = 0.0         # Most recent match score
        self.active = False           # Whether template tracking is active

    def update_template(self, frame, box):
        """
        Update the stored template with the current target crop.

        Should be called every frame while the target is visible and
        being tracked by the primary detector.

        Args:
            frame: Full video frame (BGR numpy array).
            box: Bounding box as (x1, y1, x2, y2) — pixel coordinates.
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]

        # Boundary clamping
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        tw, th = x2 - x1, y2 - y1
        if tw < 15 or th < 15:
            return

        self.template = frame[y1:y2, x1:x2].copy()
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.last_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.last_size = (tw, th)

    def search(self, frame, predicted_center=None):
        """
        Search for the template within the frame.

        The search is constrained to a region centered on `predicted_center`
        (or the last known center) with a margin of `search_margin` pixels.
        This dramatically reduces false positives compared to full-frame search.

        Args:
            frame: Full video frame (BGR numpy array).
            predicted_center: Optional (x, y) tuple from Kalman filter
                or motion model. If None, uses last known center.

        Returns:
            dict or None: If found, returns:
                {
                    'center': (cx, cy),          # Match center
                    'box': (x1, y1, x2, y2),     # Bounding box
                    'score': float,              # Correlation score (0-1)
                    'search_region': (x1, y1, x2, y2)  # Where we searched
                }
                Returns None if no match above threshold.
        """
        if self.template_gray is None:
            self.active = False
            return None

        fh, fw = frame.shape[:2]
        th, tw = self.template_gray.shape[:2]

        # Determine search center
        if predicted_center is not None:
            cx, cy = predicted_center
        elif self.last_center is not None:
            cx, cy = self.last_center
        else:
            cx, cy = fw // 2, fh // 2

        margin = self.search_margin
        s_x1 = max(0, cx - margin - tw)
        s_y1 = max(0, cy - margin - th)
        s_x2 = min(fw, cx + margin + tw)
        s_y2 = min(fh, cy + margin + th)

        # Validate search region
        if (s_x2 - s_x1) < tw or (s_y2 - s_y1) < th:
            self.active = False
            self.last_score = 0.0
            return None

        search_region = frame[s_y1:s_y2, s_x1:s_x2]
        search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(search_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        self.last_score = max_val

        if max_val >= self.match_threshold:
            # Convert from search-region coordinates to frame coordinates
            match_x = s_x1 + max_loc[0]
            match_y = s_y1 + max_loc[1]

            center = (match_x + tw // 2, match_y + th // 2)
            box = (match_x, match_y, match_x + tw, match_y + th)

            self.last_center = center
            self.active = True

            return {
                'center': center,
                'box': box,
                'score': max_val,
                'search_region': (s_x1, s_y1, s_x2, s_y2)
            }

        self.active = False
        return None

    def reset(self):
        """Clear all stored data and deactivate the tracker."""
        self.template = None
        self.template_gray = None
        self.last_center = None
        self.last_size = None
        self.last_score = 0.0
        self.active = False
