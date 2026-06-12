@echo off
title Driver Drouzyness
echo.
echo  =====================================================
echo   Driver Drouzyness - Eye Lid Detection System
echo  =====================================================
echo.
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH.
    pause & exit /b 1
)
python -c "import cv2, mediapipe, numpy" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing required packages...
    pip install -r requirements.txt
)
echo [INFO] Starting app — press Q or ESC in the camera window to quit.
echo.
python app.py
if %ERRORLEVEL% neq 0 ( echo. & echo [ERROR] Check messages above. & pause )
