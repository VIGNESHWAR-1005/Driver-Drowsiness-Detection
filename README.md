# Driver Drouzyness — Eye Lid + Yawn Detection
Driver Drowsiness Detection System using Computer Vision and Facial Landmark Analysis to monitor eye closure in real time and alert drivers when signs of fatigue are detected, improving road safety and accident prevention.
### Python 3.12 · OpenCV 4.13 · MediaPipe 0.10.35 · No Dataset · No API

---

## Two Detection Systems

### 1. EAR — Eye Aspect Ratio (Drowsiness)
```
EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)

Open eye  ~ 0.25-0.40
Closed eye ~ 0.05-0.18
Threshold  = 0.22 (default)
```
Fires alarm when eyes stay closed for **2+ seconds**.

### 2. MAR — Mouth Aspect Ratio (Yawning)
```
MAR = (|M2-M8| + |M3-M7| + |M4-M6|) / (2 * |M1-M5|)

Mouth closed ~ 0.1-0.3
Mouth open   ~ 0.4-0.6
Yawning      ~ 0.6-1.0+
Threshold    = 0.65 (default)
```
Fires alarm when mouth is wide open for **8+ consecutive frames** (~0.25s).
3-second cooldown prevents repeated triggers from one yawn.

---

## Quick Start

```bash
# Windows (double-click)
run.bat

# Or manually
pip install -r requirements.txt
python app.py
```

First run downloads `face_landmarker.task` (~6 MB) automatically.

---

## What You See on Screen

| Overlay | Meaning |
|---|---|
| Green eye contour | Eyes open |
| Red eye contour | Eyes closed (EAR below threshold) |
| Blue mouth contour | Mouth closed / normal |
| Red mouth contour | Mouth open / yawning |
| `L:0.31 R:0.29` | Per-eye EAR above each eye |
| `MAR:0.68` | Mouth opening ratio below mouth |
| `EAR` value top-left | Average EAR score |
| `MAR` value top-left | Current MAR score |
| Bottom bar 1 | DROWSY % — fills as eyes stay closed |
| Bottom bar 2 | YAWN % — fills as mouth opens |
| **WAKE UP!** red banner | Eye-close alarm |
| **YAWNING! STAY ALERT** amber banner | Yawn alarm |

---

## Configuration (top of app.py)

| Variable | Default | Effect |
|---|---|---|
| `EAR_THRESHOLD` | `0.22` | Eye closure sensitivity |
| `ALERT_DELAY_SEC` | `2.0` | Seconds before drowsy alarm |
| `MAR_THRESHOLD` | `0.65` | Yawn detection sensitivity |
| `YAWN_FRAMES` | `8` | Frames to confirm yawn (~0.27s) |
| `CAMERA_INDEX` | `0` | Which webcam to use |

### Tuning MAR for your face
- Run the app and watch the `MAR` value in the top-left
- Normal talking/expression ~ 0.2-0.4
- Wide yawn ~ 0.7-1.0+
- Set `MAR_THRESHOLD` just above your normal speech range

---

## Controls

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit |
