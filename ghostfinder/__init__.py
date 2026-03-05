"""
ghostfinder — Lightweight Visual Re-Identification & Template Tracker
=================================================================
CPU-friendly visual Re-ID using HSV color histograms and
ROI-focused template matching. Designed for edge devices
(Jetson Nano, Raspberry Pi) and real-time tracking pipelines.

Quick Start:
    from ghostfinder import TargetReID, TemplateTracker

    reid = TargetReID(similarity_threshold=0.55)
    tracker = TemplateTracker(match_threshold=0.45)
"""

from ghostfinder.reid_module import TargetReID
from ghostfinder.template_tracker import TemplateTracker
from ghostfinder.template_bank import TemplateBank

__version__ = "1.1.0"
__author__ = "ByIbos"
__all__ = ["TargetReID", "TemplateTracker", "TemplateBank"]
