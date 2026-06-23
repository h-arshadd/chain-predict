"""
main_bybit.py
-------------
Entry point for the Bybit data pipeline.
Handles:
    - Setting up logging
    - Loading config from config_bybit.yml
    - Establishing PostgreSQL connection
    - Calling the data downloader for all Bybit symbols
"""

import logging
import yaml
from pathlib import Path

from crypto_pipeline.utils.db_utils import get_db_connection
from crypto_pipeline.data.data_downloader import download_data
from crypto_pipeline.data.bybit import exchange_bybit


class BybitPipeline:
    """
    Entry point for the Bybit data pipeline.
    Handles logging setup, config loading, DB connection, and orchestrating the download.
    """

    # ── Logging setup ──────────────────────────────────────────────────────────

    def setup_logging(self):
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
                logging.FileHandler("bybit_pipeline.log")
            ]
        )

    # ── Config loader ──────────────────────────────────────────────────────────

    def load_config(self) -> dict:
        """
        Load and return the Bybit config from config_bybit.yml.

        Returns:
            Dictionary of config parameters
        """
        # Get the directory where this file lives
        config_path = Path(__file__).parent / "config_bybit.yml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    # ── Main ───────────────────────────────────────────────────────────────────

    def run(self):
        """
        Main entry point for the Bybit data pipeline.

        Flow:
            1. Set up logging
            2. Load config
            3. Connect to PostgreSQL
            4. Run data downloader for all symbols and timeframes
            5. Close DB connection
        """
        self.setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("Starting Bybit data pipeline")
        logger.info("=" * 60)

        config = self.load_config()
        logger.info(f"Config loaded. Symbols: {config['symbols']} | Timeframes: {config['time_horizons']}")

        conn = get_db_connection()

        try:
            download_data(config=config, exchange_fetcher=exchange_bybit, conn=conn)
            logger.info("Bybit data pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise

        finally:
            conn.close()
            logger.info("Database connection closed.")


def main():
    BybitPipeline().run()


if __name__ == "__main__":
    main()
