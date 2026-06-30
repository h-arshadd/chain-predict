@echo off
REM ============================================================
REM run_pipeline.bat
REM Runs the Binance AND Bybit incremental data pipelines, back to back.
REM Meant to be triggered repeatedly by Windows Task Scheduler
REM (e.g. every 1 minute). Each pipeline only fetches whatever is
REM newer than that exchange's last stored timestamp, so each run
REM is fast/cheap once both initial backfills are done.
REM ============================================================

REM --- EDIT THESE TWO PATHS FOR YOUR MACHINE ---
SET PROJECT_DIR=C:\Users\PMYLS\chain-predict
SET PYTHON_EXE=C:\Users\PMYLS\chain-predict\venv\Scripts\python.exe
REM ----------------------------------------------

cd /d "%PROJECT_DIR%"

"%PYTHON_EXE%" -m crypto_pipeline.data.binance.main
"%PYTHON_EXE%" -m crypto_pipeline.data.bybit.main

REM Exit cleanly so Task Scheduler doesn't flag it as hung
exit /b 0