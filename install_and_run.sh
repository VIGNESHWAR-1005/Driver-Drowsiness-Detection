#!/bin/bash
echo "============================================"
echo "  DRIVER DROUZYNESS — Setup and Launch"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Install Python 3.7.4."
    exit 1
fi

echo "[1/3] Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install packages. Check internet connection."
    exit 1
fi

echo ""
echo "[2/3] Dependencies installed."
echo ""
echo "[3/3] Launching Driver Drouzyness..."
echo ""
echo "Controls inside the window:"
echo "  Q or ESC  =  Quit"
echo "  R         =  Reset session"
echo "  +         =  Raise EAR threshold"
echo "  -         =  Lower EAR threshold"
echo ""

python3 app.py
