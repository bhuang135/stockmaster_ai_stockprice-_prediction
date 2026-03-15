@echo off

:: Activate virtual environment
call .venv\Scripts\activate

:: Open browser
start http://127.0.0.1:8050/

:: Run Dash app
python stock_master_app_fixed.py

pause