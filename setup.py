"""
setup.py
--------
Makes crypto_pipeline an installable Python package.
Run: pip install -e .
This allows imports like:
    from crypto_pipeline.data.binance import exchange_binance
to work from anywhere in the project.
"""

from setuptools import setup, find_packages

setup(
    name="crypto_pipeline",
    version="0.1.0",
    description="Historical OHLCV data pipeline for Binance and Bybit",
    packages=find_packages(),
    install_requires=[
        "python-binance",
        "pybit",
        "psycopg2-binary",
        "pyyaml",
        "python-dotenv",
        "pandas",
        "numpy",
    ],
)