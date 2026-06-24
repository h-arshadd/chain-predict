import logging
import time
import yaml
from pathlib import Path

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
        ]
    )


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)