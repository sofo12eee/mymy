@echo off
echo ==========================================================
echo    LEISHMANIA SCREENER - CLIENT OPTIMISE IP WEBCAM
echo    Latence reduite 4s vers moins de 500ms
echo    Developpe par BOUNAB SOUFYANE
echo ==========================================================
echo Demarrage de l'application...
echo Interface web: http://localhost:5000
echo.
echo Optimisations actives:
echo - Buffer video reduit pour IP Webcam
echo - Detection selective 1 frame sur 3
echo - Gestion erreurs amelioree
echo - Timeouts configures
echo.
python start_client.py
pause