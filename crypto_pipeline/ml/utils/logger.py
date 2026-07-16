# crypto_pipeline/ml/utils/logger.py

"""
logger.py
---------
Centralized logging utility (PDF heading 12).

Every module in this codebase already does `logger = logging.getLogger(__name__)`
and calls logger.info()/logger.error() throughout each stage -- dataset
loading, feature selection, preprocessing, training, evaluation, signal
generation, model saving. Those are child loggers of the root logger and
propagate up to it by default, so configuring the root logger ONCE, here,
is enough to centralize logging for the whole pipeline. No other module
needs its own logging.basicConfig() or handlers.

Usage (called once, at the top of a pipeline run):

    from crypto_pipeline.ml.utils.logger import setup_logging

    log_path = setup_logging(run_id="20260716_142530_random_forest")
    logger.info("Regression pipeline starting: run_id=...")

This gives every subsequent logger.info()/logger.error() call in this
process both a console line and a line in logs/{run_id}.log, until
setup_logging() is called again with a different run_id.
"""

import logging
import os

LOG_DIR = "logs"

_console_configured = False


def setup_logging(run_id: str = None, log_dir: str = LOG_DIR, level: int = logging.INFO) -> str:
    """
    Configure the root logger with a console handler (added once per
    process) and a per-run file handler (replaced on every call, so each
    run_id gets its own clean log file rather than one growing file
    shared across runs).

    Args:
        run_id: names the log file as logs/{run_id}.log. Defaults to
            "run" if not given.
        log_dir: directory log files are written to, created if missing.
        level: logging level applied to the root logger and both handlers.

    Returns:
        str: path to this run's log file.
    """
    global _console_configured

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id or 'run'}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    if not _console_configured:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        _console_configured = True

    # Drop any file handler left over from a previous run so this run's
    # log file only contains this run's own messages, not a prior run's.
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_path