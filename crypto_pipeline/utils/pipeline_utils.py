import logging
import time
import yaml
from datetime import datetime
from pathlib import Path

from crypto_pipeline.utils.db_utils import get_db_connection
from crypto_pipeline.data.data_downloader import DataDownloader

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def setup_logging(exchange_name):
    logging.Formatter.converter = time.gmtime  # log timestamps in UTC

    LOG_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s UTC | {exchange_name.upper()} | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / f"{exchange_name}_pipeline.log")
        ],
    )


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_config_dates(config):
    """
    Convert config["start_date"] and config["end_date"] from YAML strings
    into real datetime objects. end_date stays as the string "now" if set
    to that, since the pipeline resolves "now" at call time.
    """
    config["start_date"] = datetime.strptime(config["start_date"], "%Y-%m-%d")

    if config["end_date"] != "now":
        config["end_date"] = datetime.strptime(config["end_date"], "%Y-%m-%d")

    return config


def initialize_pipeline(config_path):
    """
    Load the config, parse its dates, and set up logging — the three steps
    every exchange's main() needs before it can start its pipeline run.
    Returns the ready-to-use config dict.
    """
    config = load_config(config_path)
    config = parse_config_dates(config)
    setup_logging(config["exchange"])
    return config


def run_pipeline(config_path, exchange_fetcher_cls):
    """
    Full pipeline run for one exchange: load config, set up logging, open a
    DB connection, run download(), close the connection.

    This is the single shared body every exchange's main.py calls — binance
    and bybit were doing the exact same setup/teardown around download(),
    just pointed at a different config file and exchange fetcher class.

    config_path: path to that exchange's config_*.yml
    exchange_fetcher_cls: the exchange's fetcher class itself (e.g.
        BinanceExchange, not BinanceExchange()) — instantiated here so each
        run gets its own fresh instance.
    """
    config = initialize_pipeline(config_path)
    logger = logging.getLogger(__name__)

    exchange_name = config["exchange"]
    logger.info(f"Starting {exchange_name} data pipeline")

    conn = get_db_connection()

    try:
        downloader = DataDownloader(config=config, exchange_fetcher=exchange_fetcher_cls(), conn=conn)
        downloader.download()

        logger.info(f"{exchange_name} data pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

    finally:
        conn.close()
        logger.info("Database connection closed.")