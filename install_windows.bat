@echo off
echo ==========================================================
echo    INSTALLATION LEISHMANIA SCREENER - CLIENT OPTIMISE
echo    Developpe par BOUNAB SOUFYANE
echo ==========================================================
echo Installation des dependances Python...
python -m pip install --upgrade pip
python -m pip install flask flask-socketio opencv-python torch ultralytics numpy psutil
echo.
echo Installation terminee!
echo Double-cliquez sur start_windows.bat pour lancer l'application
pause