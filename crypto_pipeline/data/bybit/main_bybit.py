import logging
from pathlib import Path

from crypto_pipeline.utils.db_utils import get_db_connection
from crypto_pipeline.utils.pipeline_utils import setup_logging, load_config
from crypto_pipeline.data.data_downloader import DataDownloader
from crypto_pipeline.data.bybit.exchange_bybit import BybitExchange


def main():
    config = load_config(Path(__file__).parent / "config_bybit.yml")
    setup_logging(config["exchange"])
    logger = logging.getLogger(__name__)

    logger.info("Starting Bybit data pipeline")

    conn = get_db_connection()

    try:
        downloader = DataDownloader(config=config, exchange_fetcher=BybitExchange(), conn=conn)
        downloader.download()

        for symbol in config["symbols"]:
            downloader.get_data(exchange=config["exchange"], symbol=symbol, timeframe="5m")

        logger.info("Bybit data pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()