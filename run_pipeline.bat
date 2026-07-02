@echo off
REM ============================================================
REM run_pipeline.bat
REM Runs the Binance AND Bybit incremental data pipelines, back to back.
REM Meant to be triggered repeatedly by Windows Task Scheduler
REM (e.g. every 1 minute). Each pipeline only fetches whatever is
REM newer than that exchange's last stored timestamp, so each run
REM is fast/cheap once both initial backfills are done.
REM ============================================================

REM --- REQUIRE ENVIRONMENT VARIABLES TO BE SET ---
if not defined PROJECT_DIR (
    echo ERROR: PROJECT_DIR environment variable not set
    exit /b 1
)

if not defined PYTHON_EXE (
    echo ERROR: PYTHON_EXE environment variable not set
    exit /b 1
)

REM Validate that python executable exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found at %PYTHON_EXE%
    exit /b 1
)

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not change to directory %PROJECT_DIR%
    exit /b 1
)

echo Running Binance pipeline...
"%PYTHON_EXE%" -m crypto_pipeline.data.binance.main
if errorlevel 1 (
    echo ERROR: Binance pipeline failed
    exit /b 1
)

echo Running Bybit pipeline...
"%PYTHON_EXE%" -m crypto_pipeline.data.bybit.main
if errorlevel 1 (
    echo ERROR: Bybit pipeline failed
    exit /b 1
)

echo Both pipelines completed successfully
exit /b 0