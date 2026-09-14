@echo off
cd /d "%~dp0\.."

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo =========================================
echo ECG DATASET CHECK
echo =========================================
echo.

python -c "from pathlib import Path; p=Path('datasets'); h=list(p.glob('*.hea')); d=list(p.glob('*.dat')); a=list(p.glob('*.atr')); print('datasets =', p.resolve()); print('HEA =', len(h)); print('DAT =', len(d)); print('ATR =', len(a)); print('First records =', [x.stem for x in h[:10]])"

echo.
echo If HEA/DAT/ATR are not zero, start dashboard and open:
echo http://127.0.0.1:8000/api/debug/dataset
echo.
pause
