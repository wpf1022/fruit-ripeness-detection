@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Fruit Ripeness YOLOv8 Detection Demo
echo   URL: http://127.0.0.1:5001
echo ============================================
python detect_app.py
pause
