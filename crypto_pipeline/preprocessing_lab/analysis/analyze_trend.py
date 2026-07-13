#!/usr/bin/env python3
"""
analyze_trend.py
----------------
Main entry point for trend preservation analysis.

Usage:
    python analyze_trend.py
"""

import os
import sys
import yaml
import numpy as np
from crypto_pipeline.preprocessing_lab.analysis.trend_preservation import (
    analyze_all_methods
)


def generate_trend_report(results):
    """Generate human-readable trend report."""
    report = []
    report.append("=" * 80)
    report.append("TREND PRESERVATION ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    for method_name in sorted(results.get("methods", {}).keys()):
        method_data = results["methods"][method_name]
        if "error" in method_data:
            continue
        
        report.append(f"\n{method_name}")
        report.append("-" * 80)
        
        scores = []
        for col_name, col_data in method_data.get("columns", {}).items():
            combined = col_data.get("combined_score", {})
            score = combined.get("combined_score", 0)
            interp = combined.get("interpretation", "")
            
            if not np.isnan(score):
                scores.append(score)
            
            report.append(f"  {col_name:<25} {score:.2f}  ({interp})")
        
        avg = np.mean(scores) if scores else 0
        report.append(f"\n  Average: {avg:.2f}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def load_config(config_path: str = None) -> dict:
    """Load preprocessing config."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "..", "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: Config not found at {config_path}")
        return None
    
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load config
    config = load_config()
    
    if config and "data" in config and "feature_columns" in config["data"]:
        feature_columns = config["data"]["feature_columns"]
    else:
        print("Using default feature columns...")
        feature_columns = [
            "open", "high", "low", "close", "volume",
            "ind_EMA", "ind_RSI_14", "ind_MACD_12_26_9"
        ]
    
    output_dir = os.path.join(base_dir, "..", "outputs")
    json_output = os.path.join(base_dir, "outputs", "trend_preservation_results.json")
    report_output = os.path.join(base_dir, "outputs", "trend_preservation_report.txt")
    
    print("=" * 80)
    print("TREND PRESERVATION ANALYSIS")
    print("=" * 80)
    print("")
    
    try:
        results = analyze_all_methods(
            output_dir,
            feature_columns,
            output_file=json_output
        )
        
        # Generate and save report
        report = generate_trend_report(results)
        print(report)
        
        with open(report_output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {report_output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()