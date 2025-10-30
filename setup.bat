@echo off
REM AW-SafeSeg (IDDAW) - Complete Setup Script
REM This script sets up both backend and frontend dependencies

echo ========================================
echo AW-SafeSeg Setup Script
echo ========================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

REM Check Node.js installation
echo [2/5] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    pause
    exit /b 1
)
node --version
npm --version
echo.

REM Install Python dependencies
echo [3/5] Installing Python dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo Python dependencies installed successfully!
echo.

REM Install frontend dependencies
echo [4/5] Installing frontend dependencies...
echo This may take a few minutes...
cd project\frontend
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies
    cd ..\..
    pause
    exit /b 1
)
cd ..\..
echo Frontend dependencies installed successfully!
echo.

REM Create necessary directories
echo [5/5] Creating necessary directories...
if not exist "project\uploads" mkdir project\uploads
if not exist "project\results" mkdir project\results
echo Directories created successfully!
echo.

REM Setup complete
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Ensure model checkpoints are in project\ckpts\
echo 2. Configure project\frontend\.env file
echo 3. Run: scripts\start_fullstack.bat
echo.
echo For detailed instructions, see README.md
echo.
pause
