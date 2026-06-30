from pathlib import Path

from crypto_pipeline.utils.pipeline_utils import run_pipeline
from crypto_pipeline.data.binance.exchange_binance import BinanceExchange


def main():
    run_pipeline(Path(__file__).parent / "config_binance.yml", BinanceExchange)


if __name__ == "__main__":
    main()