@echo off
title Credit Risk Dashboard Launcher
echo ===================================================
echo   Starting Credit Risk Assessment Dashboard Server
echo ===================================================
echo.
cd /d "C:\Users\Administrator\.gemini\antigravity\scratch\credit-risk-assessment"

:: Check if Python exists
if not exist "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe" (
    echo [ERROR] Python 3.13 not found at the expected location.
    echo Please make sure Python is installed.
    pause
    exit /b 1
)

echo Starting Streamlit...
echo [NOTE] Please KEEP this window open while using the dashboard.
echo.

"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run src/app.py

echo.
echo Dashboard stopped.
pause
