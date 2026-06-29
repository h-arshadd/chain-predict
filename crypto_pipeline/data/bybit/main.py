import logging
from pathlib import Path

from crypto_pipeline.utils.db_utils import get_db_connection
from crypto_pipeline.utils.pipeline_utils import initialize_pipeline
from crypto_pipeline.data.data_downloader import DataDownloader
from crypto_pipeline.data.bybit.exchange_bybit import BybitExchange


def main():
    config = initialize_pipeline(Path(__file__).parent / "config_bybit.yml")
    logger = logging.getLogger(__name__)

    logger.info("Starting Bybit data pipeline")

    conn = get_db_connection()

    try:
        downloader = DataDownloader(config=config, exchange_fetcher=BybitExchange(), conn=conn)
        downloader.download()

        for symbol in config["symbols"]:
            downloader.get_data(
                exchange=config["exchange"],
                symbol=symbol,
                start_date=config["start_date"],
                end_date=config["end_date"]
            )

        logger.info("Bybit data pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

    finally:
        conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    main()