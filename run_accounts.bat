@echo off
REM ============================================================
REM run_accounts.bat
REM Runs the Accounts Module. Meant to be triggered on a schedule by
REM Windows Task Scheduler (e.g. every 15-60 minutes), same as
REM run_execution.bat / run_simulator.bat / run_pipeline.bat. Each run
REM registers/updates the account in accounts.api_keys (first run only
REM -- never overwrites a stored key on later runs), then rebuilds
REM accounts.history and accounts.stats from scratch by pulling this
REM account's fill history LIVE from Bybit for every (exchange, symbol)
REM combo currently in execution.config. Places no orders itself.
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

echo Running Accounts Module...
"%PYTHON_EXE%" -m crypto_pipeline.accounts.run_accounts
if errorlevel 1 (
    echo ERROR: Accounts Module failed
    exit /b 1
)

echo Accounts run completed successfully
exit /b 0