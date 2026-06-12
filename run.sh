#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Driver Drouzyness — Linux / macOS Launcher
# ═══════════════════════════════════════════════════════

echo ""
echo "  ====================================================="
echo "   Driver Drouzyness - Eye Lid Detection System"
echo "  ====================================================="
echo ""

# Find python3
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python not found. Install Python 3.12 and retry."
    exit 1
fi

echo "[INFO] Using Python: $PYTHON ($($PYTHON --version 2>&1))"

# Check / install deps
$PYTHON -c "import cv2, mediapipe, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing dependencies..."
    $PYTHON -m pip install -r requirements.txt
fi

echo "[INFO] Starting Driver Drouzyness..."
echo "[INFO] Press Q or ESC inside the camera window to quit."
echo ""

$PYTHON app.py
