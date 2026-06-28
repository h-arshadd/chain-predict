import logging
import time
import yaml
from datetime import datetime
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