@echo off
REM ============================================================
REM run_execution.bat
REM Runs the Execution Module. Meant to be triggered repeatedly by
REM Windows Task Scheduler (e.g. every 1-5 minutes), same as
REM run_simulator.bat / run_pipeline.bat. Each run picks up wherever
REM it left off (saved state in execution.positions) and processes
REM whatever new 1-minute candles have arrived since then -- placing
REM REAL orders on Bybit when a signal fires, appending closed trades
REM to each pair's *_trades ledger table, and refreshing its row in
REM execution.stats once it has closed trades to compute from.
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

echo Running Execution Module...
"%PYTHON_EXE%" -m crypto_pipeline.execution.main
if errorlevel 1 (
    echo ERROR: Execution Module failed
    exit /b 1
)

echo Execution run completed successfully
exit /b 0