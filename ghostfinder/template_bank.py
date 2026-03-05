"""
Template Bank — Multi-Angle Visual Template Storage
====================================================
Stores multiple templates captured from different angles, scales, or
lighting conditions. When searching, all templates are matched and the
best overall result is returned (rather than relying on a single template).

Features:
    - Configurable bank size (default: 5 templates max)
    - Per-template scoring with angle/scale tags
    - Best-of-N matching strategy
    - Automatic worst-template replacement when bank is full
    - CPU-only — compatible with edge devices

Author: ByIbos
License: MIT
"""

import cv2
import numpy as np


class TemplateBank:
    """
    Multi-template storage and matching system.

    Instead of keeping a single template, this bank stores up to `max_size`
    templates captured at different times. During search, ALL templates
    are tried and the best match is returned. This dramatically improves
    robustness against scale changes, rotation, and lighting variation.

    Args:
        max_size: Maximum number of templates to store (default: 5).
        match_threshold: Minimum correlation score to accept (default: 0.40).
        search_margin: Pixel margin around predicted center (default: 150).

    Example:
        >>> bank = TemplateBank(max_size=5)
        >>> # While tracking, add templates periodically:
        >>> bank.add_template(frame, (x1, y1, x2, y2), tag="front")
        >>> bank.add_template(frame, (x1, y1, x2, y2), tag="side")
        >>> # When detector fails:
        >>> result = bank.search(frame, predicted_center=(320, 240))
        >>> if result:
        ...     print(f"Found with template '{result['tag']}' (score: {result['score']:.2f})")
    """

    def __init__(self, max_size=5, match_threshold=0.40, search_margin=150):
        self.max_size = max_size
        self.match_threshold = match_threshold
        self.search_margin = search_margin
        self.templates = []  # List of dicts: {gray, color, size, tag, score_history}

    def add_template(self, frame, box, tag="default"):
        """
        Add a new template to the bank.

        If the bank is full, the template with the lowest historical
        match score is replaced.

        Args:
            frame: Full video frame (BGR numpy array).
            box: Bounding box as (x1, y1, x2, y2).
            tag: Optional label for this template (e.g., "front", "45deg").
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        tw, th = x2 - x1, y2 - y1
        if tw < 15 or th < 15:
            return

        crop = frame[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        entry = {
            'gray': gray,
            'color': crop,
            'size': (tw, th),
            'tag': tag,
            'score_history': [],
        }

        if len(self.templates) < self.max_size:
            self.templates.append(entry)
        else:
            # Replace the weakest template
            worst_idx = self._find_worst_template()
            self.templates[worst_idx] = entry

    def _find_worst_template(self):
        """Find the index of the template with the lowest average score."""
        worst_idx = 0
        worst_avg = float('inf')
        for i, t in enumerate(self.templates):
            if not t['score_history']:
                return i  # No history = never matched = worst
            avg = sum(t['score_history']) / len(t['score_history'])
            if avg < worst_avg:
                worst_avg = avg
                worst_idx = i
        return worst_idx

    def search(self, frame, predicted_center=None):
        """
        Search for the target using all stored templates.

        Each template is matched within the ROI, and the best overall
        result is returned.

        Args:
            frame: Full video frame (BGR numpy array).
            predicted_center: Optional (x, y) from Kalman prediction.

        Returns:
            dict or None: If found:
                {
                    'center': (cx, cy),
                    'box': (x1, y1, x2, y2),
                    'score': float,
                    'tag': str,
                    'template_idx': int,
                    'search_region': (x1, y1, x2, y2)
                }
        """
        if not self.templates:
            return None

        fh, fw = frame.shape[:2]

        # Determine search center
        if predicted_center is not None:
            cx, cy = int(predicted_center[0]), int(predicted_center[1])
        else:
            cx, cy = fw // 2, fh // 2

        best_result = None
        best_score = 0.0
        best_idx = -1

        for i, tmpl in enumerate(self.templates):
            tw, th = tmpl['size']
            margin = self.search_margin

            s_x1 = max(0, cx - margin - tw)
            s_y1 = max(0, cy - margin - th)
            s_x2 = min(fw, cx + margin + tw)
            s_y2 = min(fh, cy + margin + th)

            if (s_x2 - s_x1) < tw or (s_y2 - s_y1) < th:
                continue

            search_gray = cv2.cvtColor(
                frame[s_y1:s_y2, s_x1:s_x2], cv2.COLOR_BGR2GRAY
            )

            result = cv2.matchTemplate(
                search_gray, tmpl['gray'], cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Update score history
            tmpl['score_history'].append(max_val)
            if len(tmpl['score_history']) > 20:
                tmpl['score_history'].pop(0)

            if max_val > best_score:
                best_score = max_val
                best_idx = i
                match_x = s_x1 + max_loc[0]
                match_y = s_y1 + max_loc[1]
                best_result = {
                    'center': (match_x + tw // 2, match_y + th // 2),
                    'box': (match_x, match_y, match_x + tw, match_y + th),
                    'score': max_val,
                    'tag': tmpl['tag'],
                    'template_idx': i,
                    'search_region': (s_x1, s_y1, s_x2, s_y2),
                }

        if best_score >= self.match_threshold and best_result:
            return best_result

        return None

    def count(self):
        """Return the number of stored templates."""
        return len(self.templates)

    def reset(self):
        """Clear all stored templates."""
        self.templates.clear()
