<div align="center">

# 🔍 ghostfinder

**Lightweight Visual Re-Identification & Template Tracker**

*CPU-friendly Re-ID using color histograms and template matching — fast enough for edge devices.*

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/Dependency-OpenCV-red?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Speed](https://img.shields.io/badge/Speed-<0.5ms_per_match-brightgreen)

</div>

---

## ✨ Why ghostfinder?

| Feature | ghostfinder | torchreid | deep-person-reid |
|---|:---:|:---:|:---:|
| CPU-only (no GPU needed) | ✅ | ❌ | ❌ |
| Edge device ready (RPi, Jetson) | ✅ | ❌ | ❌ |
| < 0.5ms per comparison | ✅ | ❌ | ❌ |
| Template matching fallback | ✅ | ❌ | ❌ |
| Multi-feature scoring | ✅ | ✅ | ✅ |
| ROI-constrained search | ✅ | ❌ | ❌ |
| No training data needed | ✅ | ❌ | ❌ |
| Lightweight (~300 lines total) | ✅ | ❌ | ❌ |

> Deep learning Re-ID solutions are powerful but require GPUs and large models. **ghostfinder** is designed for real-time systems on resource-constrained hardware — where you need to re-identify a target in under 1ms using only CPU.

---

## 📦 Installation

```bash
pip install ghostfinder
```

Or install from source:
```bash
git clone https://github.com/ByIbos/ghostfinder.git
cd ghostfinder
pip install -e .
```

---

## 🚀 Quick Start

### Visual Re-Identification
```python
import cv2
from ghostfinder import TargetReID

reid = TargetReID(similarity_threshold=0.55)

# While tracking the target:
reid.update_fingerprint(frame, (x1, y1, x2, y2))

# When target is lost and new detections appear:
best_id, score = reid.find_best_match(frame, all_boxes, all_ids)
if best_id is not None:
    print(f"Target re-identified as ID {best_id} (score: {score:.2f})")
```

### Template Matching Fallback
```python
from ghostfinder import TemplateTracker

tracker = TemplateTracker(match_threshold=0.45, search_margin=150)

# While detector is working:
tracker.update_template(frame, (x1, y1, x2, y2))

# When YOLO/detector fails to find the target:
result = tracker.search(frame, predicted_center=(320, 240))
if result:
    center = result['center']
    score = result['score']
    print(f"Template found at {center} (confidence: {score:.0%})")
```

### Combined Pipeline (Re-ID + Template)
```python
from ghostfinder import TargetReID, TemplateTracker

reid = TargetReID()
template = TemplateTracker()

def handle_target_lost(frame, boxes, ids, kalman_prediction):
    """
    3-layer recovery strategy:
    1. Try Re-ID (color fingerprint matching)
    2. Try Template Matching (visual search)
    3. Use Kalman prediction (coast on momentum)
    """
    # Layer 1: Re-ID
    match_id, score = reid.find_best_match(frame, boxes, ids)
    if match_id is not None:
        return ("REID", match_id, score)

    # Layer 2: Template
    result = template.search(frame, predicted_center=kalman_prediction)
    if result:
        return ("TEMPLATE", result['center'], result['score'])

    # Layer 3: Kalman (handled externally)
    return ("PREDICT", kalman_prediction, 0.0)
```

---

## 🔧 API Reference

### `TargetReID(similarity_threshold=0.55)`

A visual fingerprinting system that stores the target's color histogram,
aspect ratio, and area for comparison against new detections.

#### Methods

| Method | Returns | Description |
|---|---|---|
| `update_fingerprint(frame, box)` | None | Store/update fingerprint from current crop |
| `compare(frame, box)` | float | Compare one box against fingerprint (0-1) |
| `find_best_match(frame, boxes, ids)` | (id, score) | Find best matching detection |
| `reset()` | None | Clear all stored data |

#### Scoring Breakdown

| Component | Weight | Metric |
|---|---|---|
| Color similarity | 60% | HSV histogram Bhattacharyya distance |
| Shape similarity | 20% | Aspect ratio difference |
| Size similarity | 20% | Area ratio (min/max) |

---

### `TemplateTracker(match_threshold=0.45, search_margin=150)`

A fallback tracker that stores the last known visual appearance and
searches for it using `cv2.matchTemplate` within a constrained ROI.

#### Methods

| Method | Returns | Description |
|---|---|---|
| `update_template(frame, box)` | None | Store current target appearance |
| `search(frame, predicted_center)` | dict or None | Search for template in frame |
| `reset()` | None | Clear all stored data |

#### Search Result Dictionary

```python
{
    'center': (cx, cy),                    # Match center point
    'box': (x1, y1, x2, y2),              # Bounding box
    'score': 0.72,                         # Correlation score (0-1)
    'search_region': (sx1, sy1, sx2, sy2)  # ROI that was searched
}
```

---

## 📐 How It Works

```
┌──────────────────────────────────────────────────┐
│                ghostfinder Pipeline                    │
│                                                  │
│  ┌─── TRACKING PHASE ─────────────────────────┐  │
│  │  Target visible → update fingerprint        │  │
│  │    • HSV histogram (30×32 bins)             │  │
│  │    • Aspect ratio (w/h)                     │  │
│  │    • Area (w×h)                             │  │
│  │    • Template crop (grayscale)              │  │
│  └─────────────────────────────────────────────┘  │
│                      │                            │
│                  Target Lost!                     │
│                      │                            │
│  ┌─── RECOVERY PHASE ─────────────────────────┐  │
│  │                                             │  │
│  │  Layer 1: Re-ID (TargetReID)                │  │
│  │    Compare all detections vs fingerprint     │  │
│  │    Score = 0.6×color + 0.2×shape + 0.2×size │  │
│  │    Match if score > threshold (0.55)         │  │
│  │                                             │  │
│  │  Layer 2: Template (TemplateTracker)         │  │
│  │    Search ROI = predicted_center ± margin    │  │
│  │    cv2.matchTemplate (TM_CCOEFF_NORMED)     │  │
│  │    Match if correlation > threshold (0.45)   │  │
│  │                                             │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## ⚡ Performance

Benchmarked on a Raspberry Pi 4B (no GPU):

| Operation | Time | Notes |
|---|---|---|
| `update_fingerprint()` | ~0.3ms | Per frame |
| `compare()` (single box) | ~0.4ms | Per candidate |
| `find_best_match()` (10 boxes) | ~3ms | Worst case |
| `search()` (template) | ~1-3ms | Depends on ROI size |

---

## 🎯 Use Cases

- **🚁 Drone Tracking** — Re-acquire target after occlusion
- **📹 Surveillance** — Track individuals across camera blind spots
- **🤖 Robot Vision** — Persistent object following
- **🏭 Industrial** — Track parts on conveyor belts
- **🎯 Sports Analytics** — Player tracking through crowds

---

## 📜 License

MIT License — use it anywhere, commercially or personally.

---

<div align="center">
<i>Built with ❤️ by <a href="https://github.com/ByIbos">ByIbos</a> — extracted from the <a href="https://github.com/ByIbos/Auto-ReID-Drone-Tracker">Auto-ReID Drone Tracker</a> project.</i>
</div>
