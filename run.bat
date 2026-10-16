@echo off
rem Runs the acoustic image link. Double-click to start.
rem Before running: amp on, USB-6341 plugged in (AO0 -> amp, AI0 <- RX transducer).

cd /d "%~dp0"
echo Acoustic image link - press Ctrl+C to stop
echo.
python "Fih image.py"
echo.
if errorlevel 1 (
    echo Script exited with an error - see the messages above.
) else (
    echo Done.
)
pause
