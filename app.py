import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import math
import threading
import sys
import os
import urllib.request

# =============================================================
#  CONFIGURATION  ← edit values here
# =============================================================
# --- Eye drowsiness ---
EAR_THRESHOLD    = 0.22   # EAR below this = eyes closed
ALERT_DELAY_SEC  = 2.0    # seconds eyes must stay closed before alarm fires

# --- Yawn detection ---
MAR_THRESHOLD    = 0.65   # MAR above this = yawning (lower = more sensitive)
YAWN_FRAMES      = 8      # consecutive frames above MAR threshold before alarm
                          # (at ~30fps, 8 frames ≈ 0.27s — prevents false triggers)

# --- General ---
ALARM_FREQ_HZ    = 880    # beep pitch Hz (Windows only)
CAMERA_INDEX     = 0      # 0 = default webcam, try 1 or 2 if wrong camera
WINDOW_TITLE     = "Driver Drouzyness"
# =============================================================

# ── MediaPipe landmark indices ──────────────────────────────

# Eye: 6 points [outer, top-outer, top-inner, inner, bot-inner, bot-outer]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_IDX = [
    61,   # M1 — left corner
    40,   # M2 — top-left outer
    12,    # M3 — top centre
    270,  # M4 — top-right outer
    291,  # M5 — right corner
    318,  # M6 — bot-right outer
    14,   # M7 — bot centre
    88,   # M8 — bot-left outer
]

# Model
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
GREEN    = (0, 200, 150)
RED      = (0,  50, 255)
AMBER    = (0, 170, 255)
BLUE     = (220, 120,  0)   # mouth contour colour (BGR)
WHITE    = (255, 255, 255)
PANEL_BG = (20,  28,  42)


# =============================================================
#  MODEL DOWNLOAD
# =============================================================
def ensure_model():
    if os.path.exists(MODEL_FILE):
        return
    print("[INFO] Downloading face_landmarker.task (~6 MB) — one-time setup...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("[INFO] Model saved:", MODEL_FILE)
    except Exception as exc:
        print("[ERROR] Download failed:", exc)
        print("        Get it from:", MODEL_URL)
        print("        Save as:    ", MODEL_FILE)
        sys.exit(1)


# =============================================================
#  EAR — Eye Aspect Ratio
#  Formula: (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
# =============================================================
def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def compute_ear(landmarks, indices, fw, fh):
    """
    Returns (ear_value, list_of_6_pixel_points).
    Open eye  ~ 0.25-0.40
    Closed eye ~ 0.05-0.18
    """
    pts = [(int(landmarks[i].x * fw), int(landmarks[i].y * fh))
           for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    horiz = _dist(p1, p4)
    if horiz == 0:
        return 0.0, pts
    ear = (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horiz)
    return ear, pts


# =============================================================
#  MAR — Mouth Aspect Ratio
#
#  Uses 8 mouth landmarks:
#    M1 (left corner) ────────────────── M5 (right corner)
#    M2 (top-left)    M3 (top-ctr)  M4 (top-right)
#    M8 (bot-left)    M7 (bot-ctr)  M6 (bot-right)
#
#  MAR = (|M2-M8| + |M3-M7| + |M4-M6|) / (2 * |M1-M5|)
#
#  Mouth closed ~ 0.1-0.3
#  Mouth open   ~ 0.4-0.6
#  Yawning      ~ 0.6-1.0+
# =============================================================
def compute_mar(landmarks, fw, fh):
    """
    Returns (mar_value, list_of_8_pixel_points).
    """
    pts = [(int(landmarks[i].x * fw), int(landmarks[i].y * fh))
           for i in MOUTH_IDX]
    m1, m2, m3, m4, m5, m6, m7, m8 = pts

    # three vertical distances across the mouth opening
    vert1 = _dist(m2, m8)   # left pair
    vert2 = _dist(m3, m7)   # centre pair
    vert3 = _dist(m4, m6)   # right pair

    # horizontal mouth width
    horiz = _dist(m1, m5)

    if horiz == 0:
        return 0.0, pts

    mar = (vert1 + vert2 + vert3) / (2.0 * horiz)
    return mar, pts
# =============================================================
#  ALARM — background thread (non-blocking)
# =============================================================
_alarm_running = False
_alarm_thread  = None
def _alarm_loop():
    global _alarm_running
    while _alarm_running:
        _beep()
        time.sleep(0.85)
def _beep():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(ALARM_FREQ_HZ, 380)
        elif sys.platform == "darwin":
            os.system("afplay /System/Library/Sounds/Sosumi.aiff >/dev/null 2>&1 &")
        else:
            ret = os.system(
                f"speaker-test -t sine -f {ALARM_FREQ_HZ} -l 1 -P 1 >/dev/null 2>&1")
            if ret != 0:
                print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def start_alarm():
    global _alarm_running, _alarm_thread
    if not _alarm_running:
        _alarm_running = True
        _alarm_thread = threading.Thread(target=_alarm_loop, daemon=True)
        _alarm_thread.start()


def stop_alarm():
    global _alarm_running
    _alarm_running = False


# =============================================================
#  DRAWING HELPERS
# =============================================================
def draw_contour(frame, pts, color, thickness=1):
    """Draw a closed polygon through a list of pixel points."""
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [arr], True, color, thickness, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def draw_ear_tag(frame, pts, ear, side):
    cx = sum(p[0] for p in pts) // len(pts)
    cy = min(p[1] for p in pts) - 8
    col = RED if ear < EAR_THRESHOLD else GREEN
    cv2.putText(frame, f"{side}:{ear:.3f}", (cx - 30, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

def draw_mar_tag(frame, pts, mar):
    cx = sum(p[0] for p in pts) // len(pts)
    cy = max(p[1] for p in pts) + 16   # below the mouth
    col = RED if mar > MAR_THRESHOLD else BLUE
    cv2.putText(frame, f"MAR:{mar:.3f}", (cx - 35, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


def draw_hud(frame, avg_ear, mar, closed_secs, yawn_frames,
             alert_type, total_alerts, fps):
    """
    Draw HUD overlay on the camera frame.
    alert_type: None | 'drowsy' | 'yawn'
    """
    fh, fw = frame.shape[:2]

    # ── top-left info panel ──────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (230, 100), PANEL_BG, -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    ear_col = (RED   if avg_ear < EAR_THRESHOLD else
               AMBER if avg_ear < EAR_THRESHOLD + 0.04 else GREEN)
    mar_col = RED if mar > MAR_THRESHOLD else BLUE

    cv2.putText(frame, f"EAR     {avg_ear:.3f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, ear_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"MAR     {mar:.3f}",     (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, mar_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS     {fps:02d}",      (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140,160,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ALERTS  {total_alerts}", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA)

    # ── top-right driver status ──────────────────────────
    if alert_type == 'drowsy':
        stxt, scol = "DROWSY!", RED
    elif alert_type == 'yawn':
        stxt, scol = "YAWNING!", AMBER
    elif avg_ear < EAR_THRESHOLD + 0.04:
        stxt, scol = "CAUTION", AMBER
    else:
        stxt, scol = "ALERT",   GREEN

    sx = fw - len(stxt) * 11 - 10
    cv2.putText(frame, stxt, (sx, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, scol, 1, cv2.LINE_AA)

    # ── EAR drowsiness progress bar (bottom) ────────────
    bx, by  = 10, fh - 40
    bw, bh  = fw - 20, 9
    ear_ratio = min(closed_secs / ALERT_DELAY_SEC, 1.0)

    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (30,40,55), -1)
    fw2 = int(bw * ear_ratio)
    if fw2 > 0:
        bc = RED if ear_ratio > 0.8 else AMBER if ear_ratio > 0.5 else GREEN
        cv2.rectangle(frame, (bx, by), (bx+fw2, by+bh), bc, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (60,80,100), 1)
    cv2.putText(frame, f"DROWSY {int(ear_ratio*100)}%", (bx+4, by-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140,160,180), 1, cv2.LINE_AA)

    # ── MAR yawn progress bar (bottom, below EAR bar) ───
    by2 = fh - 22
    mar_ratio = min(mar / MAR_THRESHOLD, 1.0)   # fills as mouth opens

    cv2.rectangle(frame, (bx, by2), (bx+bw, by2+bh), (30,40,55), -1)
    fw3 = int(bw * mar_ratio)
    if fw3 > 0:
        bc2 = RED if mar_ratio >= 1.0 else AMBER if mar_ratio > 0.7 else BLUE
        cv2.rectangle(frame, (bx, by2), (bx+fw3, by2+bh), bc2, -1)
    cv2.rectangle(frame, (bx, by2), (bx+bw, by2+bh), (60,80,100), 1)
    cv2.putText(frame, f"YAWN   {int(mar_ratio*100)}%", (bx+4, by2-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140,160,180), 1, cv2.LINE_AA)

    # ── closed-eyes timer ───────────────────────────────
    if closed_secs > 0.1:
        cv2.putText(frame, f"{closed_secs:.1f}s", (fw-70, fh-46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    RED if alert_type == 'drowsy' else AMBER, 1, cv2.LINE_AA)

    # ── ALERT banner ─────────────────────────────────────
    if alert_type == 'drowsy':
        _draw_banner(frame, fw, fh, "WAKE  UP !", (0,0,180), (0,50,255))
    elif alert_type == 'yawn':
        _draw_banner(frame, fw, fh, "YAWNING! STAY ALERT", (0,80,160), (0,140,255))


def _draw_banner(frame, fw, fh, text, tint_bgr, text_bgr):
    """Draw a full-frame tinted alert banner."""
    al = frame.copy()
    cv2.rectangle(al, (0, 0), (fw, fh), tint_bgr, -1)
    cv2.addWeighted(al, 0.18, frame, 0.82, 0, frame)

    font, sc, th = cv2.FONT_HERSHEY_DUPLEX, 1.5, 2
    (tw, tth), _ = cv2.getTextSize(text, font, sc, th)
    tx, ty = (fw-tw)//2, (fh+tth)//2
    cv2.putText(frame, text, (tx+2, ty+2), font, sc, (0,0,0),    th+2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx,   ty),   font, sc, text_bgr,   th,   cv2.LINE_AA)


# =============================================================
#  MAIN
# =============================================================
def main():
    print("=" * 60)
    print("  Driver Drouzyness — Eye Lid + Yawn Detection System")
    print("  EAR (drowsiness) + MAR (yawning) | No Dataset | No API")
    print("=" * 60)
    print(f"  EAR Threshold  : {EAR_THRESHOLD}  (eyes closed below this)")
    print(f"  Alert Delay    : {ALERT_DELAY_SEC}s (eye-closed duration)")
    print(f"  MAR Threshold  : {MAR_THRESHOLD}  (yawn detected above this)")
    print(f"  Yawn Frames    : {YAWN_FRAMES}   (frames to confirm yawn)")
    print(f"  Camera Index   : {CAMERA_INDEX}")
    print("  Press  Q  or  ESC  to quit")
    print("=" * 60)

    # ── 1. Download model ──────────────────────────────────
    ensure_model()

    # ── 2. Build FaceLandmarker ───────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    landmarker_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.55,
        min_face_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(landmarker_opts)
    print("[INFO] FaceLandmarker ready.")

    # ── 3. Open camera ─────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        print("        Change CAMERA_INDEX at the top of app.py and retry.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 620)
    closed_start    = None     # time.time() when eyes first closed
    eye_alerting    = False    # currently in eye-close alarm

    # Yawn state
    yawn_frame_cnt  = 0        # consecutive frames where MAR > threshold
    yawn_alerting   = False    # currently in yawn alarm
    yawn_cooldown   = 0.0      # time.time() — ignore yawns until after this

    total_alerts    = 0
    fps_buf         = []
    frame_ts_ms     = 0        # synthetic timestamp for MediaPipe VIDEO mode

    print("[INFO] Camera open. Face the camera to begin.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        t_now = time.time()
        fh, fw = frame.shape[:2]

        # Mirror so left/right matches natural view
        frame = cv2.flip(frame, 1)

        # ── 5. Run MediaPipe detection ─────────────────────
        frame_ts_ms += 33    # must increase each frame
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, frame_ts_ms)

        closed_secs = 0.0
        avg_ear     = 0.30
        mar         = 0.0
        alert_type  = None    # None | 'drowsy' | 'yawn'

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ── EAR: both eyes ──────────────────────────────
            l_ear, l_pts = compute_ear(lm, LEFT_EYE,  fw, fh)
            r_ear, r_pts = compute_ear(lm, RIGHT_EYE, fw, fh)
            avg_ear = (l_ear + r_ear) / 2.0

            # ── MAR: mouth ──────────────────────────────────
            mar, m_pts = compute_mar(lm, fw, fh)

            # ── Draw eye contours ───────────────────────────
            draw_contour(frame, l_pts,
                         RED if l_ear < EAR_THRESHOLD else GREEN)
            draw_contour(frame, r_pts,
                         RED if r_ear < EAR_THRESHOLD else GREEN)

            # ── Draw mouth contour ──────────────────────────
            mouth_col = RED if mar > MAR_THRESHOLD else BLUE
            draw_contour(frame, m_pts, mouth_col)

            # ── Draw labels ─────────────────────────────────
            draw_ear_tag(frame, l_pts, l_ear, "L")
            draw_ear_tag(frame, r_pts, r_ear, "R")
            draw_mar_tag(frame, m_pts, mar)

            # ═══════════════════════════════════════════════
            #  DROWSINESS LOGIC  (EAR-based)
            # ═══════════════════════════════════════════════
            if avg_ear < EAR_THRESHOLD:
                if closed_start is None:
                    closed_start = t_now
                closed_secs = t_now - closed_start

                if closed_secs >= ALERT_DELAY_SEC and not eye_alerting:
                    eye_alerting   = True
                    total_alerts  += 1
                    start_alarm()
                    print(f"[DROWSY] Alert #{total_alerts} — "
                          f"eyes closed {closed_secs:.1f}s")
            else:
                if eye_alerting:
                    eye_alerting = False
                    stop_alarm()
                    print("[INFO]   Eyes reopened — drowsy alert cleared")
                closed_start = None

            if mar > MAR_THRESHOLD:
                yawn_frame_cnt += 1

                if (yawn_frame_cnt >= YAWN_FRAMES
                        and not yawn_alerting
                        and t_now > yawn_cooldown):
                    yawn_alerting  = True
                    total_alerts  += 1
                    start_alarm()
                    yawn_cooldown  = t_now + 3.0   # 3-second cooldown
                    print(f"[YAWN]   Alert #{total_alerts} — "
                          f"MAR={mar:.3f}  frames={yawn_frame_cnt}")
            else:
                # mouth closed again — end yawn alert
                if yawn_alerting:
                    yawn_alerting  = False
                    yawn_frame_cnt = 0
                    if not eye_alerting:
                        stop_alarm()
                    print("[INFO]   Mouth closed — yawn alert cleared")
                else:
                    yawn_frame_cnt = max(0, yawn_frame_cnt - 1)

            # ── Decide which alert type to show ─────────────
            if eye_alerting:
                alert_type = 'drowsy'
            elif yawn_alerting:
                alert_type = 'yawn'

        else:
            # No face — clear all alerts
            if eye_alerting or yawn_alerting:
                eye_alerting  = False
                yawn_alerting = False
                stop_alarm()
            closed_start   = None
            yawn_frame_cnt = 0

        # ── 6. FPS ─────────────────────────────────────────
        fps_buf = [t for t in fps_buf + [t_now] if t_now - t <= 1.0]
        fps     = len(fps_buf)

        # ── 7. Draw HUD and show ────────────────────────────
        draw_hud(frame, avg_ear, mar, closed_secs,
                 yawn_frame_cnt, alert_type, total_alerts, fps)
        cv2.imshow(WINDOW_TITLE, frame)

        # ── 8. Key handler ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    # ── Cleanup ────────────────────────────────────────────
    stop_alarm()
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
