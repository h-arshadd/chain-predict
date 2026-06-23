"""
main_binance.py
---------------
Entry point for the Binance data pipeline.
Handles:
    - Setting up logging
    - Loading config from config_binance.yml
    - Establishing PostgreSQL connection
    - Calling the data downloader for all Binance symbols
"""

import logging
import yaml
from pathlib import Path

from crypto_pipeline.utils.db_utils import get_db_connection
from crypto_pipeline.data.data_downloader import download_data
from crypto_pipeline.data.binance import exchange_binance


# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging():
    """
    Configure logging for the entire pipeline.
    Logs to both console and a log file.
    """
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            # Print logs to console
            logging.StreamHandler(),
            # Also save logs to a file
            logging.FileHandler("binance_pipeline.log")
        ]
    )


# ── Config loader ──────────────────────────────────────────────────────────────

def load_config() -> dict:
    """
    Load and return the Binance config from config_binance.yml.

    Returns:
        Dictionary of config parameters
    """
    # Get the directory where this file lives
    config_path = Path(__file__).parent / "config_binance.yml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """
    Main entry point for the Binance data pipeline.
    
    Flow:
        1. Set up logging
        2. Load config
        3. Connect to PostgreSQL
        4. Run data downloader for all symbols and timeframes
        5. Close DB connection
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Starting Binance data pipeline")
    logger.info("=" * 60)

    config = load_config()
    logger.info(f"Config loaded. Symbols: {config['symbols']} | Timeframes: {config['time_horizons']}")

    conn = get_db_connection()

    try:
        download_data(config=config, exchange_fetcher=exchange_binance, conn=conn)
        logger.info("Binance data pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()