@echo off
echo Setting up DeepTrace Pro (Minimal Version)...

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install minimal requirements
echo Installing minimal requirements...
pip install -r requirements-minimal.txt

echo.
echo Setup complete! Project size: ~416 MB
echo.
echo To run the application:
echo 1. Activate venv: .venv\Scripts\activate.bat
echo 2. Start backend: uvicorn app.api.server:app --host 0.0.0.0 --port 8000
echo 3. Start frontend: streamlit run app/main.py
echo.
echo Or simply run: run.bat 