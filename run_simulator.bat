@echo off
REM ============================================================
REM run_simulator.bat
REM Runs the Simulator Module. Meant to be triggered repeatedly by
REM Windows Task Scheduler (e.g. every 1-5 minutes), same as
REM run_pipeline.bat. Each run picks up wherever it left off (saved
REM state in simulator.positions) and processes whatever new
REM 1-minute candles have arrived since then -- appending new rows to
REM each strategy's *_trades ledger table and refreshing its row in
REM simulator.stats, never rebuilding anything from scratch.
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

echo Running Simulator Module...
"%PYTHON_EXE%" -m crypto_pipeline.simulator.main
if errorlevel 1 (
    echo ERROR: Simulator Module failed
    exit /b 1
)

echo Simulator run completed successfully
exit /b 0