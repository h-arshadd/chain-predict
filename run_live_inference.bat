@echo off
REM ============================================================
REM run_live_inference.bat
REM Runs live ML inference. Meant to be triggered repeatedly by
REM Windows Task Scheduler (e.g. every 5 minutes), same as
REM run_execution.bat / run_simulator.bat / run_pipeline.bat.
REM
REM Each run picks up wherever it left off per run_id (the latest
REM datetime already saved in ml.model_signals) and predicts on
REM whatever new candles have landed since then for every trained
REM regression/classification run_id in ml.run_configs, writing
REM fresh 1/0/-1 signals back into ml.model_signals -- the same
REM table the Strategy Builder reads an ML model's signal series
REM from (crypto_pipeline/strategy_builder/assemble.py).
REM
REM First run for a given run_id (nothing saved yet) backfills from
REM that model's own training start_date instead of just today
REM forward -- same script, no separate "backfill mode" to remember
REM to run once by hand.
REM
REM Most runs will find zero new closed candles (e.g. a 5-minute
REM schedule against an hourly-timeframe model) and write nothing --
REM that is expected, not a failure.
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

echo Running live ML inference for every trained model...
"%PYTHON_EXE%" -m crypto_pipeline.ml.inference.live_inference --all
if errorlevel 1 (
    echo ERROR: Live inference failed for at least one run_id -- see log above
    exit /b 1
)

echo Live inference run completed successfully
exit /b 0