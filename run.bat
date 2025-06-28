@echo off
echo Starting DeepTrace Pro...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start backend in background
echo Starting backend server...
start "DeepTrace Backend" cmd /k "uvicorn app.api.server:app --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting frontend...
streamlit run app/main.py 