@echo off
title Driver Drouzyness — Setup and Run
color 0A

echo ============================================
echo   DRIVER DROUZYNESS — Setup and Launch
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.7.4 from python.org
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install packages. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [2/3] Dependencies installed successfully.
echo.
echo [3/3] Launching Driver Drouzyness...
echo.
echo Controls inside the window:
echo   Q or ESC  =  Quit
echo   R         =  Reset session
echo   +         =  Raise EAR threshold
echo   -         =  Lower EAR threshold
echo.

python app.py

pause
