#!/usr/bin/env python3
"""
analyze_stationarity.py
-----------------------
Main entry point for stationarity analysis.

This script:
1. Loads all preprocessed outputs from preprocessing_lab/outputs/
2. Compares each method's stationarity using ADF, KPSS, and ACF tests
3. Evaluates trend preservation
4. Generates a comprehensive report comparing all methods

Usage:
    python analyze_stationarity.py [--output OUTPUT_FILE]

Example output structure:
{
  "methods": {
    "none": {
      "columns": {
        "close": {
          "adf_test": {...},
          "kpss_test": {...},
          "combined_verdict": {...},
          "acf_analysis": {...},
          "trend_preservation": {...}
        }
      }
    },
    "standard_scaler": {...},
    ...
  }
}
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from crypto_pipeline.preprocessing_lab.analysis.stationarity_analysis import (
    StationarityAnalyzer,
    analyze_preprocessing_methods
)


def load_config(config_path: str = None) -> dict:
    """Load preprocessing config to get feature column names."""
    if config_path is None:
        # Look for config.yaml in preprocessing_lab
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Will analyze all numeric columns except datetime/target")
        return None
    
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return None


def generate_summary_report(results: dict) -> str:
    """
    Generate a human-readable summary report from analysis results.
    
    Args:
        results: Dictionary from analyze_preprocessing_methods()
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("STATIONARITY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    n_methods = results.get("n_methods_analyzed", 0)
    report.append(f"Methods Analyzed: {n_methods}")
    report.append("")
    
    # Summary table for each column
    all_columns = set()
    for method_name, method_data in results.get("methods", {}).items():
        if "columns" in method_data:
            all_columns.update(method_data["columns"].keys())
    
    for col_name in sorted(all_columns):
        report.append("-" * 80)
        report.append(f"COLUMN: {col_name}")
        report.append("-" * 80)
        report.append("")
        
        # Create comparison table
        report.append(f"{'Method':<30} {'ADF':<15} {'KPSS':<15} {'Verdict':<20} {'Trend Pres.':<12}")
        report.append("-" * 92)
        
        for method_name in sorted(results.get("methods", {}).keys()):
            method_data = results["methods"][method_name]
            
            if "error" in method_data or col_name not in method_data.get("columns", {}):
                report.append(f"{method_name:<30} {'ERROR':<15} {'':<15} {'':<20} {'':<12}")
                continue
            
            col_analysis = method_data["columns"][col_name]
            
            # Get test results
            adf_stat = col_analysis.get("adf_test", {}).get("stationary", False)
            kpss_stat = col_analysis.get("kpss_test", {}).get("stationary", False)
            verdict = col_analysis.get("combined_verdict", {}).get("verdict", "UNKNOWN")
            trend_pres = col_analysis.get("trend_preservation", {}).get("preservation_score", 0)
            
            adf_str = "✓ STAT" if adf_stat else "✗ NON-S"
            kpss_str = "✓ STAT" if kpss_stat else "✗ NON-S"
            trend_str = f"{trend_pres:.2f}"
            
            report.append(
                f"{method_name:<30} {adf_str:<15} {kpss_str:<15} {verdict:<20} {trend_str:<12}"
            )
        
        report.append("")
    
    # Overall recommendations
    report.append("=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    
    # Find best methods by criteria
    stationarity_scores = {}
    trend_preservation_scores = {}
    
    for method_name in sorted(results.get("methods", {}).keys()):
        method_data = results["methods"][method_name]
        if "error" in method_data:
            continue
        
        n_stationary = 0
        n_total = 0
        avg_trend_pres = 0
        n_trends = 0
        
        for col_analysis in method_data.get("columns", {}).values():
            verdict = col_analysis.get("combined_verdict", {})
            if verdict.get("verdict") in ["DEFINITELY STATIONARY", "LIKELY STATIONARY (weak trend)"]:
                n_stationary += 1
            n_total += 1
            
            trend_pres = col_analysis.get("trend_preservation", {}).get("preservation_score", 0)
            if not np.isnan(trend_pres):
                avg_trend_pres += trend_pres
                n_trends += 1
        
        stationarity_scores[method_name] = n_stationary / n_total if n_total > 0 else 0
        trend_preservation_scores[method_name] = avg_trend_pres / n_trends if n_trends > 0 else 0
    
    # Sort by stationarity
    best_stationary = sorted(
        stationarity_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    report.append("Best for Stationarity:")
    for method, score in best_stationary:
        report.append(f"  {method:<30} {score:.2%}")
    report.append("")
    
    # Sort by trend preservation
    best_trends = sorted(
        trend_preservation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    report.append("Best for Trend Preservation:")
    for method, score in best_trends:
        report.append(f"  {method:<30} {score:.2%}")
    report.append("")
    
    # Balanced recommendations (high stationarity + high trend preservation)
    report.append("Best Balanced (Stationarity + Trend Preservation):")
    balanced_scores = {}
    for method in stationarity_scores.keys():
        # Harmonic mean: emphasizes both criteria being good
        stat_score = stationarity_scores[method]
        trend_score = trend_preservation_scores[method]
        if stat_score > 0 and trend_score > 0:
            harmonic_mean = 2 * (stat_score * trend_score) / (stat_score + trend_score)
        else:
            harmonic_mean = 0
        balanced_scores[method] = harmonic_mean
    
    best_balanced = sorted(
        balanced_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    for method, score in best_balanced:
        report.append(f"  {method:<30} {score:.2%}")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze stationarity of all preprocessing methods"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for JSON results (default: stationarity_results.json)"
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Output file for text report (default: stationarity_report.txt)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: preprocessing_lab/config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Determine output paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    json_output = args.output or os.path.join(base_dir, "outputs", "stationarity_results.json")
    report_output = args.report or os.path.join(base_dir, "outputs", "stationarity_report.txt")
    
    # Load config to get feature columns
    config = load_config(args.config)
    
    if config and "data" in config and "feature_columns" in config["data"]:
        feature_columns = config["data"]["feature_columns"]
    else:
        feature_columns = None
    
    # Run analysis
    print("=" * 80)
    print("STATIONARITY ANALYSIS")
    print("=" * 80)
    print("")
    
    output_dir = os.path.join(base_dir, "..", "outputs")
    
    try:
        results = analyze_preprocessing_methods(
            output_dir,
            feature_columns,
            output_file=json_output
        )
        
        # Generate and save report
        report = generate_summary_report(results)
        print("")
        print(report)
        
        os.makedirs(os.path.dirname(report_output) or ".", exist_ok=True)
        with open(report_output, "w", encoding="utf-8") as f:            
            f.write(report)
        print(f"\nReport saved to: {report_output}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()