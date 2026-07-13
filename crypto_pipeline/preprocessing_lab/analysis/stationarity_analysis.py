"""
stationarity_analysis.py
------------------------
Comprehensive stationarity analysis module for evaluating preprocessing methods.

Evaluates each preprocessing method using:
1. Augmented Dickey-Fuller (ADF) test
2. KPSS test  
3. Autocorrelation analysis
4. Trend preservation assessment

The goal is to quantify how different preprocessing methods affect:
- Stationarity (do they make the series stationary?)
- Trend preservation (do they keep meaningful market trends?)
- Memory/autocorrelation structure (does the series retain useful patterns?)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import warnings
import json
import os

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - install it for full stationarity analysis")


class StationarityAnalyzer:
    """
    Analyzes stationarity of preprocessed time series data.
    
    Provides methods for:
    - ADF (Augmented Dickey-Fuller) test
    - KPSS test
    - Autocorrelation function (ACF) analysis
    - Trend preservation evaluation
    - Comprehensive reporting
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels is required for stationarity analysis. "
                "Install it with: pip install statsmodels"
            )
    
    @staticmethod
    def adf_test(series: pd.Series, name: str = "", maxlag: int = "AIC") -> Dict:
        """
        Perform Augmented Dickey-Fuller test.
        
        H0 (null hypothesis): The series has a unit root (is non-stationary)
        H1 (alternative): The series is stationary
        
        A series is considered stationary if we REJECT the null hypothesis
        (p-value < 0.05).
        
        Args:
            series: Time series to test
            name: Name for reporting
            maxlag: Max lags to use ('AIC' for automatic selection or int)
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values for the test
        clean_series = series.dropna()
        
        if len(clean_series) < 2:
            return {
                "test": "ADF",
                "statistic": np.nan,
                "p_value": np.nan,
                "n_lags": np.nan,
                "n_obs": len(clean_series),
                "stationary": False,
                "reason": "Insufficient data after NaN removal"
            }
        
        try:
            result = adfuller(clean_series, maxlag=maxlag, autolag='AIC')
            
            adf_stat, p_value, n_lags, n_obs, critical_values, ic_best = result
            
            # Reject null hypothesis if p_value < 0.05 => series is stationary
            is_stationary = p_value < 0.05
            
            return {
                "test": "ADF",
                "statistic": float(adf_stat),
                "p_value": float(p_value),
                "n_lags": int(n_lags),
                "n_obs": int(n_obs),
                "critical_values": {
                    "1%": float(critical_values.get("1%", np.nan)),
                    "5%": float(critical_values.get("5%", np.nan)),
                    "10%": float(critical_values.get("10%", np.nan))
                },
                "stationary": bool(is_stationary),
                "interpretation": (
                    "Stationary (reject H0)" if is_stationary 
                    else "Non-stationary (fail to reject H0)"
                )
            }
        except Exception as e:
            return {
                "test": "ADF",
                "error": str(e),
                "stationary": False
            }
    
    @staticmethod
    def kpss_test(series: pd.Series, name: str = "", regression: str = "c") -> Dict:
        """
        Perform KPSS test (Kwiatkowski-Phillips-Schmidt-Shin).
        
        H0 (null hypothesis): The series is stationary
        H1 (alternative): The series has a unit root (is non-stationary)
        
        This is the OPPOSITE of ADF: we want to FAIL TO REJECT H0 here
        (high p-value) to conclude stationarity.
        
        Args:
            series: Time series to test
            name: Name for reporting
            regression: 'c' for constant, 'ct' for constant+trend
            
        Returns:
            Dictionary with test results
        """
        clean_series = series.dropna()
        
        if len(clean_series) < 2:
            return {
                "test": "KPSS",
                "statistic": np.nan,
                "p_value": np.nan,
                "n_obs": len(clean_series),
                "stationary": False,
                "reason": "Insufficient data after NaN removal"
            }
        
        try:
            result = kpss(clean_series, regression=regression, nlags="auto")
            
            kpss_stat, p_value, n_lags, critical_values = result
            
            # Fail to reject H0 if p_value > 0.05 => series is stationary
            # (This is opposite to ADF logic)
            is_stationary = p_value > 0.05
            
            return {
                "test": "KPSS",
                "statistic": float(kpss_stat),
                "p_value": float(p_value),
                "n_lags": int(n_lags),
                "n_obs": len(clean_series),
                "critical_values": {
                    "10%": float(critical_values.get("10%", np.nan)),
                    "5%": float(critical_values.get("5%", np.nan)),
                    "2.5%": float(critical_values.get("2.5%", np.nan)),
                    "1%": float(critical_values.get("1%", np.nan))
                },
                "stationary": bool(is_stationary),
                "interpretation": (
                    "Stationary (fail to reject H0)" if is_stationary 
                    else "Non-stationary (reject H0)"
                )
            }
        except Exception as e:
            return {
                "test": "KPSS",
                "error": str(e),
                "stationary": False
            }
    
    @staticmethod
    def combined_stationarity_verdict(adf_result: Dict, kpss_result: Dict) -> Dict:
        """
        Combine ADF and KPSS results for a robust verdict.
        
        Best practice in time series analysis is to use BOTH tests:
        - ADF: tests if series has unit root
        - KPSS: tests if series is stationary
        
        Cases:
        1. ADF stationary=True & KPSS stationary=True  => DEFINITELY STATIONARY
        2. ADF stationary=True & KPSS stationary=False => LIKELY STATIONARY (but maybe trend)
        3. ADF stationary=False & KPSS stationary=True => LIKELY TREND STATIONARY
        4. ADF stationary=False & KPSS stationary=False => DEFINITELY NON-STATIONARY
        
        Args:
            adf_result: Result dict from adf_test()
            kpss_result: Result dict from kpss_test()
            
        Returns:
            Combined verdict dict
        """
        adf_stat = adf_result.get("stationary", False)
        kpss_stat = kpss_result.get("stationary", False)
        
        if adf_stat and kpss_stat:
            verdict = "DEFINITELY STATIONARY"
            confidence = 1.0
        elif adf_stat and not kpss_stat:
            verdict = "LIKELY STATIONARY (weak trend)"
            confidence = 0.75
        elif not adf_stat and kpss_stat:
            verdict = "TREND STATIONARY (detrend needed)"
            confidence = 0.75
        else:
            verdict = "DEFINITELY NON-STATIONARY"
            confidence = 1.0
        
        return {
            "adf_stationary": adf_stat,
            "kpss_stationary": kpss_stat,
            "verdict": verdict,
            "confidence": float(confidence)
        }
    
    @staticmethod
    def acf_analysis(series: pd.Series, nlags: int = 40) -> Dict:
        """
        Compute autocorrelation function (ACF) and analyze decay pattern.
        
        Purpose: measure how much memory/correlation the series retains.
        
        - Fast decay => low memory => series is noisy/pure returns
        - Slow decay => high memory => series preserves trends/patterns
        
        Args:
            series: Time series to analyze
            nlags: Number of lags to compute (default 40)
            
        Returns:
            Dictionary with ACF analysis
        """
        clean_series = series.dropna()
        
        if len(clean_series) < nlags + 1:
            return {
                "nlags": nlags,
                "n_obs": len(clean_series),
                "error": "Insufficient data for ACF analysis",
                "acf_values": [],
                "decay_rate": np.nan,
                "memory_score": np.nan
            }
        
        try:
            acf_values = acf(clean_series, nlags=nlags, fft=True)
            
            # Compute decay rate: how quickly does ACF drop below 0.05?
            threshold = 0.05
            decay_lag = np.where(np.abs(acf_values[1:]) < threshold)[0]
            if len(decay_lag) > 0:
                decay_lag = decay_lag[0] + 1
            else:
                decay_lag = nlags
            
            # Memory score: fraction of lags with |ACF| > 0.05
            # High = more memory retained, Low = series is like noise
            memory_score = np.mean(np.abs(acf_values[1:]) > threshold)
            
            return {
                "nlags": nlags,
                "n_obs": len(clean_series),
                "acf_values": acf_values.tolist(),
                "lag_1_acf": float(acf_values[1]),
                "decay_lag": int(decay_lag),
                "decay_lag_pct": float(decay_lag / nlags * 100),
                "memory_score": float(memory_score),
                "interpretation": (
                    "High memory retained" if memory_score > 0.5 
                    else "Low memory (noise-like)" if memory_score < 0.2
                    else "Moderate memory"
                )
            }
        except Exception as e:
            return {
                "nlags": nlags,
                "error": str(e),
                "memory_score": np.nan
            }
    
    @staticmethod
    def trend_preservation_score(original: pd.Series, transformed: pd.Series) -> Dict:
        """
        Measure how well a transformed series preserves trends from the original.
        
        Computes:
        1. Correlation: raw correlation between original and transformed
        2. Monotonic trend agreement: do the trends (up/down) match?
        3. Peak/trough preservation: are major swings still visible?
        
        Args:
            original: Original (untransformed) series
            transformed: Transformed series
            
        Returns:
            Dictionary with trend preservation metrics
        """
        # Remove NaNs from both series (aligned)
        mask = original.notna() & transformed.notna()
        orig_clean = original[mask]
        trans_clean = transformed[mask]
        
        if len(orig_clean) < 2:
            return {
                "error": "Insufficient data after NaN removal",
                "preservation_score": 0.0
            }
        
        # 1. Correlation
        corr = orig_clean.corr(trans_clean)
        
        # 2. Trend agreement: compare direction of changes
        orig_diff = orig_clean.diff().fillna(0)
        trans_diff = trans_clean.diff().fillna(0)
        
        # Avoid division by zero
        orig_nonzero = np.abs(orig_diff) > 1e-10
        trans_nonzero = np.abs(trans_diff) > 1e-10
        
        # Where both have non-zero change, check if sign matches
        both_nonzero = orig_nonzero & trans_nonzero
        if both_nonzero.sum() > 0:
            trend_agreement = (
                (np.sign(orig_diff[both_nonzero]) == 
                 np.sign(trans_diff[both_nonzero])).sum() / both_nonzero.sum()
            )
        else:
            trend_agreement = 0.5  # No significant change to compare
        
        # 3. Peak/trough preservation using rolling max/min
        window = max(5, len(orig_clean) // 20)
        orig_rolling_range = orig_clean.rolling(window).max() - orig_clean.rolling(window).min()
        trans_rolling_range = trans_clean.rolling(window).max() - trans_clean.rolling(window).min()
        
        mask2 = orig_rolling_range.notna() & trans_rolling_range.notna()
        if mask2.sum() > 0:
            range_corr = orig_rolling_range[mask2].corr(trans_rolling_range[mask2])
        else:
            range_corr = 0.5
        
        # Combined preservation score (0-1)
        preservation_score = (corr + trend_agreement + range_corr) / 3
        
        return {
            "correlation": float(np.nan_to_num(corr, nan=0.0)),
            "trend_agreement": float(trend_agreement),
            "volatility_preservation": float(np.nan_to_num(range_corr, nan=0.0)),
            "preservation_score": float((preservation_score + 1) / 2),  # Normalize to 0-1
            "interpretation": (
                "Excellent preservation" if preservation_score > 0.7 
                else "Good preservation" if preservation_score > 0.4
                else "Moderate preservation" if preservation_score > 0.1
                else "Poor preservation"
            )
        }
    
    def analyze_single_column(
        self, 
        original: pd.Series, 
        transformed: pd.Series,
        column_name: str = ""
    ) -> Dict:
        """
        Run comprehensive stationarity analysis on a single column.
        
        Args:
            original: Original untransformed series
            transformed: Preprocessed series
            column_name: Name of the column (for reporting)
            
        Returns:
            Comprehensive analysis dictionary
        """
        results = {
            "column": column_name,
            "original": {
                "n_values": len(original),
                "n_nans": original.isna().sum(),
                "mean": float(original.mean()) if original.notna().any() else np.nan,
                "std": float(original.std()) if original.notna().any() else np.nan,
                "min": float(original.min()) if original.notna().any() else np.nan,
                "max": float(original.max()) if original.notna().any() else np.nan,
            },
            "transformed": {
                "n_values": len(transformed),
                "n_nans": transformed.isna().sum(),
                "mean": float(transformed.mean()) if transformed.notna().any() else np.nan,
                "std": float(transformed.std()) if transformed.notna().any() else np.nan,
                "min": float(transformed.min()) if transformed.notna().any() else np.nan,
                "max": float(transformed.max()) if transformed.notna().any() else np.nan,
            }
        }
        
        # Stationarity tests on transformed series
        results["adf_test"] = self.adf_test(transformed, name=column_name)
        results["kpss_test"] = self.kpss_test(transformed, name=column_name)
        results["combined_verdict"] = self.combined_stationarity_verdict(
            results["adf_test"], results["kpss_test"]
        )
        
        # ACF analysis on transformed series
        results["acf_analysis"] = self.acf_analysis(transformed)
        
        # Trend preservation
        results["trend_preservation"] = self.trend_preservation_score(original, transformed)
        
        return results
    
    def analyze_dataframe(
        self,
        original_df: pd.DataFrame,
        transformed_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze all columns in a dataframe.
        
        Args:
            original_df: Original untransformed dataframe
            transformed_df: Preprocessed dataframe
            feature_columns: List of columns to analyze (default: all except datetime/target)
            
        Returns:
            Dictionary with analysis for each column
        """
        if feature_columns is None:
            # Default: analyze numeric columns except datetime-like
            feature_columns = [
                col for col in original_df.columns 
                if original_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
                and col.lower() not in ['datetime', 'timestamp', 'date', 'target', 'label']
            ]
        
        results = {
            "n_columns": len(feature_columns),
            "columns": {}
        }
        
        for col in feature_columns:
            if col not in original_df.columns or col not in transformed_df.columns:
                results["columns"][col] = {"error": f"Column '{col}' not found in data"}
                continue
            
            results["columns"][col] = self.analyze_single_column(
                original_df[col],
                transformed_df[col],
                col
            )
        
        return results


def analyze_preprocessing_methods(
    base_output_dir: str,
    config_features: List[str],
    output_file: Optional[str] = None
) -> Dict:
    """
    Convenience function to analyze all preprocessing methods in outputs directory.
    
    Args:
        base_output_dir: Path to preprocessing_lab/outputs
        config_features: List of feature column names to analyze
        output_file: Optional path to save JSON report
        
    Returns:
        Dictionary with results for each method
    """
    analyzer = StationarityAnalyzer()
    
    # Get list of preprocessing method folders
    if not os.path.exists(base_output_dir):
        raise FileNotFoundError(f"Output directory not found: {base_output_dir}")
    
    method_dirs = [
        d for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d))
    ]
    
    # Load original (untransformed) data from "none" method
    none_path = os.path.join(base_output_dir, "none", "transformed.csv")
    if not os.path.exists(none_path):
        raise FileNotFoundError(f"Original data not found at: {none_path}")
    
    original_df = pd.read_csv(none_path)
    
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "base_output_dir": base_output_dir,
        "n_methods_analyzed": len(method_dirs),
        "methods": {}
    }
    
    for method_name in sorted(method_dirs):
        method_dir = os.path.join(base_output_dir, method_name)
        data_path = os.path.join(method_dir, "transformed.csv")
        
        if not os.path.exists(data_path):
            results["methods"][method_name] = {
                "error": f"No transformed.csv found in {method_dir}"
            }
            continue
        
        print(f"Analyzing {method_name}...", end=" ", flush=True)
        
        try:
            transformed_df = pd.read_csv(data_path)
            
            analysis = analyzer.analyze_dataframe(
                original_df, transformed_df, config_features
            )
            
            results["methods"][method_name] = analysis
            print("✓")
        except Exception as e:
            results["methods"][method_name] = {
                "error": str(e)
            }
            print(f"✗ ({e})")
    
    # Save results if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Stationarity Analysis Module loaded successfully.")
    print("Use analyze_preprocessing_methods() to run full analysis.")