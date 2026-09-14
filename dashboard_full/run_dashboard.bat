@echo off
cd /d "%~dp0\.."
echo.
echo ============================================
echo ECG NAS FPGA Dashboard
echo ============================================
echo.
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)
echo Starting: http://127.0.0.1:8000
echo.
python -m uvicorn dashboard_full.main:app --reload --host 127.0.0.1 --port 8000
pause
