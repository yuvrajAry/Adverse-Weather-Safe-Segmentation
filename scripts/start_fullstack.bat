@echo off
echo IDDAW Full Stack Application Startup
echo ====================================
echo.

echo Starting Backend API Server...
start "IDDAW Backend" cmd /k "cd /d \"%~dp0..\project\" && python start_backend.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo Starting Frontend Development Server...
start "IDDAW Frontend" cmd /k "cd /d \"%~dp0..\frontend\" && npm run dev"

echo.
echo Both servers are starting...
echo.
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Please wait a moment for both servers to fully start.
echo Then open http://localhost:5173 in your browser.
echo.
echo Press any key to exit this window...
pause > nul
