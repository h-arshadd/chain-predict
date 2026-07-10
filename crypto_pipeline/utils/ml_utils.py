# crypto_pipeline/utils/ml_utils.py

"""
ml_utils.py
-----------
Utility functions for ML module.
"""

import logging
import yaml


def setup_logging(name: str = "ml_module"):
    """Setup logging for ML module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{name}.log")
        ]
    )


def load_config_yaml(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_target_config(target_config: dict, model_type: str) -> bool:
    """Validate target configuration matches model type."""
    
    if not target_config:
        raise ValueError("Target config is empty")
    
    if model_type == "regression":
        valid_types = ["return", "log_return"]
        target_type = target_config.get("type", "return")
        if target_type not in valid_types:
            raise ValueError(f"Invalid regression target type: {target_type}")
    
    elif model_type == "classification":
        valid_types = ["binary", "threshold"]
        target_type = target_config.get("type", "binary")
        if target_type not in valid_types:
            raise ValueError(f"Invalid classification target type: {target_type}")
    
    return True


def validate_data_config(data_config: dict) -> bool:
    """Validate market data configuration."""
    
    required_fields = ["symbol", "exchange", "timeframe", "start_date", "end_date"]
    
    for field in required_fields:
        if field not in data_config:
            raise ValueError(f"Missing required field in data config: {field}")
    
    valid_exchanges = ["binance", "bybit"]
    if data_config["exchange"].lower() not in valid_exchanges:
        raise ValueError(f"Invalid exchange: {data_config['exchange']}")
    
    return True