from pathlib import Path

from crypto_pipeline.utils.pipeline_utils import run_pipeline
from crypto_pipeline.data.bybit.exchange_bybit import BybitExchange


def main():
    run_pipeline(Path(__file__).parent / "config_bybit.yml", BybitExchange)


if __name__ == "__main__":
    main()