"""
trend_preservation.py
----------------------
Analyzes how well preprocessing methods preserve meaningful market trends.

Trend preservation = how closely does the transformed series match the 
original series in terms of:
  1. Direction of changes (up/down movements)
  2. Magnitude of swings (peak/trough patterns)
  3. Correlation with original
  4. Overall shape/pattern

Why it matters:
- Price prediction models need trend info to work
- If preprocessing destroys trends, model can't learn price movements
- Scalers preserve trends better than differencing methods
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import json
import os


class TrendPreservationAnalyzer:
    """
    Analyzes how well preprocessing methods preserve trends in financial data.
    """
    
    @staticmethod
    def correlation_metric(original: pd.Series, transformed: pd.Series) -> Dict:
        """
        Raw correlation between original and transformed series.
        
        High correlation (>0.9) = preprocessing preserves linear relationship
        Low correlation (<0.3) = series is completely transformed
        
        Args:
            original: Original untransformed series
            transformed: Preprocessed series
            
        Returns:
            Dict with correlation metrics
        """
        mask = original.notna() & transformed.notna()
        orig_clean = original[mask]
        trans_clean = transformed[mask]
        
        if len(orig_clean) < 2:
            return {
                "correlation": np.nan,
                "interpretation": "Insufficient data"
            }
        
        corr = orig_clean.corr(trans_clean)
        
        if np.isnan(corr):
            return {
                "correlation": np.nan,
                "interpretation": "Cannot compute (constant series?)"
            }
        
        if corr > 0.95:
            interp = "Perfect preservation (nearly identical)"
        elif corr > 0.8:
            interp = "Excellent preservation"
        elif corr > 0.5:
            interp = "Good preservation"
        elif corr > 0:
            interp = "Moderate preservation"
        else:
            interp = "Poor preservation (inverted/transformed)"
        
        return {
            "correlation": float(corr),
            "interpretation": interp
        }
    
    @staticmethod
    def direction_agreement(original: pd.Series, transformed: pd.Series) -> Dict:
        """
        Percentage of price movements where direction matches (up or down).
        
        How it works:
        1. Compute changes: close[t] - close[t-1]
        2. For each change, check if sign (up/down) matches between original & transformed
        3. Report % of matching directions
        
        High agreement (>0.8) = trends follow same pattern
        Low agreement (<0.5) = trends are inverted or independent
        
        Args:
            original: Original series
            transformed: Preprocessed series
            
        Returns:
            Dict with direction metrics
        """
        mask = original.notna() & transformed.notna()
        orig_clean = original[mask]
        trans_clean = transformed[mask]
        
        if len(orig_clean) < 2:
            return {
                "direction_agreement": np.nan,
                "n_changes": 0,
                "interpretation": "Insufficient data"
            }
        
        # Compute first differences (changes)
        orig_diff = orig_clean.diff().fillna(0)
        trans_diff = trans_clean.diff().fillna(0)
        
        # Get signs (-1, 0, +1)
        orig_sign = np.sign(orig_diff)
        trans_sign = np.sign(trans_diff)
        
        # Only count where both have non-zero change
        # (if one is constant, can't compare direction)
        both_nonzero = (orig_sign != 0) & (trans_sign != 0)
        
        if both_nonzero.sum() == 0:
            return {
                "direction_agreement": 0.5,
                "n_changes": 0,
                "interpretation": "No significant changes to compare"
            }
        
        # % where signs match
        matches = (orig_sign[both_nonzero] == trans_sign[both_nonzero]).sum()
        agreement = matches / both_nonzero.sum()
        
        if agreement > 0.9:
            interp = "Excellent - trends perfectly aligned"
        elif agreement > 0.75:
            interp = "Good - most trends match"
        elif agreement > 0.6:
            interp = "Moderate - trends somewhat similar"
        elif agreement > 0.5:
            interp = "Poor - mostly random agreement"
        else:
            interp = "Very poor - trends inverted"
        
        return {
            "direction_agreement": float(agreement),
            "n_changes": int(both_nonzero.sum()),
            "interpretation": interp
        }
    
    @staticmethod
    def volatility_preservation(original: pd.Series, transformed: pd.Series, 
                               window: int = None) -> Dict:
        """
        Measures if peak/trough patterns are preserved.
        
        How it works:
        1. Compute rolling range (max - min) over windows
        2. Correlate rolling ranges between original and transformed
        3. High correlation = volatility patterns preserved
        
        Example:
        - Original: price swings from 100 to 110 (range=10)
        - If transformed: swings from 0.0 to 0.1 (similar range pattern)
        - -> High volatility preservation
        
        Args:
            original: Original series
            transformed: Preprocessed series
            window: Window size for rolling range (default: series_length / 20)
            
        Returns:
            Dict with volatility metrics
        """
        mask = original.notna() & transformed.notna()
        orig_clean = original[mask]
        trans_clean = transformed[mask]
        
        if len(orig_clean) < 5:
            return {
                "volatility_preservation": np.nan,
                "interpretation": "Insufficient data"
            }
        
        if window is None:
            window = max(5, len(orig_clean) // 20)
        
        # Compute rolling ranges
        orig_rolling = orig_clean.rolling(window).max() - orig_clean.rolling(window).min()
        trans_rolling = trans_clean.rolling(window).max() - trans_clean.rolling(window).min()
        
        # Remove NaN from rolling computation
        mask2 = orig_rolling.notna() & trans_rolling.notna()
        
        if mask2.sum() < 2:
            return {
                "volatility_preservation": np.nan,
                "interpretation": "Window too large for series length"
            }
        
        # Correlate ranges
        vol_corr = orig_rolling[mask2].corr(trans_rolling[mask2])
        
        if np.isnan(vol_corr):
            return {
                "volatility_preservation": np.nan,
                "interpretation": "Cannot compute (constant ranges?)"
            }
        
        if vol_corr > 0.9:
            interp = "Excellent - volatility patterns match"
        elif vol_corr > 0.7:
            interp = "Good - most volatility preserved"
        elif vol_corr > 0.4:
            interp = "Moderate - some volatility pattern"
        else:
            interp = "Poor - volatility patterns lost"
        
        return {
            "volatility_preservation": float(vol_corr),
            "window": window,
            "interpretation": interp
        }
    
    @staticmethod
    def extrema_preservation(original: pd.Series, transformed: pd.Series,
                            prominence_pct: float = 5.0) -> Dict:
        """
        Measures if local peaks and troughs are preserved.
        
        How it works:
        1. Find peaks (local maxima) and troughs (local minima)
        2. Check if same indices are peaks/troughs in transformed
        3. Report % of preserved extrema
        
        A peak is a value higher than neighbors by prominence_pct%.
        
        Example:
        - Original has peaks at indices [10, 50, 100]
        - If transformed also has peaks at [10, 50, 100]
        - -> Good extrema preservation
        
        Args:
            original: Original series
            transformed: Preprocessed series
            prominence_pct: Min % above neighbors to count as extremum
            
        Returns:
            Dict with extrema metrics
        """
        mask = original.notna() & transformed.notna()
        orig_clean = original[mask].values
        trans_clean = transformed[mask].values
        
        if len(orig_clean) < 3:
            return {
                "extrema_preservation": np.nan,
                "n_peaks_original": 0,
                "interpretation": "Insufficient data"
            }
        
        def find_extrema(series):
            peaks = []
            troughs = []
            for i in range(1, len(series) - 1):
                threshold = np.abs(series[i] * prominence_pct / 100)
                if series[i] > series[i-1] + threshold and series[i] > series[i+1] + threshold:
                    peaks.append(i)
                elif series[i] < series[i-1] - threshold and series[i] < series[i+1] - threshold:
                    troughs.append(i)
            return set(peaks), set(troughs)
        
        orig_peaks, orig_troughs = find_extrema(orig_clean)
        trans_peaks, trans_troughs = find_extrema(trans_clean)
        
        all_extrema = orig_peaks | orig_troughs
        
        if len(all_extrema) == 0:
            return {
                "extrema_preservation": 1.0,
                "n_extrema": 0,
                "interpretation": "No extrema found (series too smooth)"
            }
        
        trans_extrema = trans_peaks | trans_troughs
        preserved = len(all_extrema & trans_extrema) / len(all_extrema)
        
        if preserved > 0.8:
            interp = "Excellent - extrema well preserved"
        elif preserved > 0.6:
            interp = "Good - most extrema preserved"
        elif preserved > 0.4:
            interp = "Moderate - some extrema preserved"
        else:
            interp = "Poor - extrema lost"
        
        return {
            "extrema_preservation": float(preserved),
            "n_extrema_original": len(all_extrema),
            "n_extrema_preserved": len(all_extrema & trans_extrema),
            "interpretation": interp
        }
    
    @staticmethod
    def combined_trend_score(original: pd.Series, transformed: pd.Series) -> Dict:
        """
        Combines multiple trend metrics into single 0-1 score.
        
        Score components:
        1. Correlation (0-1)
        2. Direction agreement (0-1)  
        3. Volatility preservation (0-1)
        4. Extrema preservation (0-1)
        
        Combined score = simple average of above
        """
        corr_metric = TrendPreservationAnalyzer.correlation_metric(original, transformed)
        corr = corr_metric.get("correlation", 0.5)
        corr = np.nan_to_num(corr, nan=0.5)  # Treat NaN as neutral
        corr_score = (corr + 1) / 2  # Scale -1..1 to 0..1
        
        dir_metric = TrendPreservationAnalyzer.direction_agreement(original, transformed)
        dir_score = dir_metric.get("direction_agreement", 0.5)
        dir_score = np.nan_to_num(dir_score, nan=0.5)
        
        vol_metric = TrendPreservationAnalyzer.volatility_preservation(original, transformed)
        vol_score = vol_metric.get("volatility_preservation", 0.5)
        vol_score = np.nan_to_num(vol_score, nan=0.5)
        vol_score = (vol_score + 1) / 2  # Scale -1..1 to 0..1
        
        extrema_metric = TrendPreservationAnalyzer.extrema_preservation(original, transformed)
        extrema_score = extrema_metric.get("extrema_preservation", 0.5)
        extrema_score = np.nan_to_num(extrema_score, nan=0.5)
        
        combined = (corr_score + dir_score + vol_score + extrema_score) / 4
        
        if combined > 0.85:
            interp = "Excellent trend preservation"
        elif combined > 0.7:
            interp = "Good trend preservation"
        elif combined > 0.5:
            interp = "Moderate trend preservation"
        elif combined > 0.3:
            interp = "Poor trend preservation"
        else:
            interp = "Very poor trend preservation"
        
        return {
            "combined_score": float(combined),
            "score_components": {
                "correlation": float(corr_score),
                "direction_agreement": float(dir_score),
                "volatility_preservation": float(vol_score),
                "extrema_preservation": float(extrema_score)
            },
            "interpretation": interp
        }
    
    def analyze_column(self, original: pd.Series, transformed: pd.Series, 
                      column_name: str = "") -> Dict:
        """
        Run full trend preservation analysis on one column.
        
        Args:
            original: Original series
            transformed: Preprocessed series
            column_name: Column name for reporting
            
        Returns:
            Complete analysis dict
        """
        return {
            "column": column_name,
            "correlation": self.correlation_metric(original, transformed),
            "direction_agreement": self.direction_agreement(original, transformed),
            "volatility_preservation": self.volatility_preservation(original, transformed),
            "extrema_preservation": self.extrema_preservation(original, transformed),
            "combined_score": self.combined_trend_score(original, transformed)
        }


def analyze_all_methods(
    base_output_dir: str,
    feature_columns: List[str],
    output_file: str = None
) -> Dict:
    """
    Analyze trend preservation for all preprocessing methods.
    
    Args:
        base_output_dir: Path to preprocessing_lab/outputs
        feature_columns: List of feature columns to analyze
        output_file: Optional path to save JSON results
        
    Returns:
        Dict with results for each method
    """
    analyzer = TrendPreservationAnalyzer()
    
    # Load original (untransformed) data
    none_path = os.path.join(base_output_dir, "none", "transformed.csv")
    if not os.path.exists(none_path):
        raise FileNotFoundError(f"Original data not found at: {none_path}")
    
    original_df = pd.read_csv(none_path)
    
    method_dirs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d))
    ]
    
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "base_output_dir": base_output_dir,
        "n_methods": len(method_dirs),
        "methods": {}
    }
    
    for method_name in sorted(method_dirs):
        method_dir = os.path.join(base_output_dir, method_name)
        data_path = os.path.join(method_dir, "transformed.csv")
        
        if not os.path.exists(data_path):
            results["methods"][method_name] = {"error": "No transformed.csv"}
            continue
        
        print(f"Analyzing {method_name}...", end=" ", flush=True)
        
        try:
            transformed_df = pd.read_csv(data_path)
            
            method_results = {"columns": {}}
            for col in feature_columns:
                if col not in original_df.columns or col not in transformed_df.columns:
                    continue
                
                method_results["columns"][col] = analyzer.analyze_column(
                    original_df[col],
                    transformed_df[col],
                    col
                )
            
            results["methods"][method_name] = method_results
            print("✓")
        except Exception as e:
            results["methods"][method_name] = {"error": str(e)}
            print(f"✗ ({e})")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    return results