@echo off
title Academic Research Assistant v2.0 Launcher

echo ============================================================
echo  Academic Research Assistant v2.0 - Windows Launcher
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if we're in a virtual environment
python -c "import sys; print('Virtual Environment: ' + ('Yes' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No'))"

echo.
echo Checking system requirements...
echo ----------------------------------------

REM Run the Python launcher
python launch.py

echo.
echo Application finished.
pause