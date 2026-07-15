"""
validate_targets.py
-------------------
Validates ML module target generation by backtesting the signals/targets
against real market data.

Uses actual backtest engine to see if:
1. Regression targets make money when used as entry signals
2. Classification targets (-1/0/1) make money when used as signals

If targets are correct, backtest should be profitable. If losing money,
targets need tuning.

DESIGN NOTES:
- Targets are generated on a specified timeframe (e.g. 1h candles)
- Validation backtests on 1-minute data using merge_asof to align signals
- This matches the signals/backtest workflow (multi-timeframe is intentional)
- TP/SL levels in backtest config should reflect your target thresholds
"""

import os
import json
import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from pathlib import Path

from crypto_pipeline.backtest.backtest import run_backtest
from crypto_pipeline.ml_module.main import run_ml_pipeline
from crypto_pipeline.data.data_downloader import get_data

logger = logging.getLogger(__name__)


def load_validation_config(config_path=None) -> dict:
    """Load validation configuration from config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ml_config(config_path=None) -> dict:
    """Load ML module configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "ml_module" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_target_distribution(df: pd.DataFrame, model_type: str) -> dict:
    """
    Validate that targets have sensible distribution.
    
    Returns:
        dict with validation results and warnings
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "stats": {}
    }
    
    if 'target' not in df.columns:
        validation_results["valid"] = False
        validation_results["warnings"].append("No 'target' column found in dataset")
        return validation_results
    
    target = df['target'].dropna()
    
    if len(target) == 0:
        validation_results["valid"] = False
        validation_results["warnings"].append("All target values are NaN")
        return validation_results
    
    validation_results["stats"] = {
        "non_null_count": len(target),
        "null_count": df['target'].isna().sum(),
        "min": float(target.min()),
        "max": float(target.max()),
        "mean": float(target.mean()),
        "std": float(target.std()),
    }
    
    if model_type == "classification":
        unique_vals = set(target.unique())
        expected_vals = {-1, 0, 1}
        
        if not unique_vals.issubset(expected_vals):
            validation_results["valid"] = False
            validation_results["warnings"].append(
                f"Classification targets contain unexpected values: {unique_vals - expected_vals}. "
                f"Expected only -1, 0, 1"
            )
        
        counts = target.value_counts().to_dict()
        validation_results["stats"]["class_distribution"] = counts
        
        # Warn if heavily imbalanced
        total = len(target)
        for class_val in [-1, 0, 1]:
            if class_val in counts:
                pct = counts[class_val] / total * 100
                if pct < 5:
                    validation_results["warnings"].append(
                        f"Class {class_val} only {pct:.1f}% of data - may not generate enough signals"
                    )
                if pct > 95:
                    validation_results["warnings"].append(
                        f"Class {class_val} dominates {pct:.1f}% of data - targets may be too biased"
                    )
    
    elif model_type == "regression":
        # Check if returns are too small
        if target.abs().mean() < 0.00001:
            validation_results["warnings"].append(
                f"Average return magnitude is very small ({target.abs().mean():.8f}) - "
                f"targets may not have enough signal"
            )
        
        # Check for outliers
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1
        outliers = ((target < q1 - 1.5*iqr) | (target > q3 + 1.5*iqr)).sum()
        if outliers / len(target) > 0.1:
            validation_results["warnings"].append(
                f"High outlier count ({outliers/len(target)*100:.1f}%) - "
                f"consider filtering extreme returns"
            )
    
    return validation_results


def convert_regression_to_signals(df: pd.DataFrame, upper_threshold: float, lower_threshold: float) -> pd.DataFrame:
    """
    Convert regression targets (log returns) to signals.

    Signal logic:
    - target > upper_threshold → buy signal (1)
    - target < lower_threshold → sell signal (-1)
    - else → no signal (0)

    IMPORTANT: upper_threshold/lower_threshold should be the SAME values
    used to build the classification target (ml_module/config.yaml's
    target.upper_threshold / target.lower_threshold) -- not a separate
    number invented here. That's the threshold that was actually used to
    decide what counts as a real move; reusing it keeps regression and
    classification validated against the identical bar (same convention
    already followed by preprocessing_lab/model_evaluation/signal_conversion.py).

    Args:
        df: DataFrame with 'target' column (log returns)
        upper_threshold: return above this counts as a buy signal
        lower_threshold: return below this counts as a sell signal

    Returns:
        DataFrame with 'datetime', 'signal', 'return_value', 'threshold_used' columns.
        return_value/threshold_used are carried along only so they can be
        joined back onto the trade ledger after the backtest runs -- the
        backtest engine itself only reads 'signal'.
    """

    signals = df[['datetime', 'target']].copy()
    signals = signals.rename(columns={'target': 'return_value'})
    signals['signal'] = 0
    signals.loc[signals['return_value'] > upper_threshold, 'signal'] = 1
    signals.loc[signals['return_value'] < lower_threshold, 'signal'] = -1
    signals['threshold_used'] = np.where(
        signals['signal'] == 1, upper_threshold,
        np.where(signals['signal'] == -1, lower_threshold, np.nan)
    )

    return signals[['datetime', 'signal', 'return_value', 'threshold_used']]


def convert_classification_to_signals(df: pd.DataFrame, upper_threshold: float, lower_threshold: float,
                                       horizon: int) -> pd.DataFrame:
    """
    Convert classification targets (-1/0/1) directly to signals.

    The -1/0/1 label alone doesn't say how close price came to the barrier,
    so we also compute the plain forward log-return over the same horizon
    used to build the target -- that's the number to compare against the
    threshold afterward.

    Args:
        df: DataFrame with 'target' (-1/0/1) and 'close' columns
        upper_threshold, lower_threshold: same values used to build the target
        horizon: same horizon used to build the target (candles ahead)

    Returns:
        DataFrame with 'datetime', 'signal', 'return_value', 'threshold_used' columns.
    """

    signals = df[['datetime', 'target']].copy()
    signals = signals.rename(columns={'target': 'signal'})
    signals['return_value'] = np.log(df['close'].shift(-horizon) / df['close']).values
    signals['threshold_used'] = np.where(
        signals['signal'] == 1, upper_threshold,
        np.where(signals['signal'] == -1, lower_threshold, np.nan)
    )

    return signals[['datetime', 'signal', 'return_value', 'threshold_used']]


def validate_targets(ml_config_path: str, validation_config_path: str = None) -> dict:
    """
    Main validation function: runs ML pipeline, generates signals, backtests them.
    
    Workflow:
    1. Run ML pipeline to generate targets on configured timeframe (e.g. 1h)
    2. Validate target distribution for sanity
    3. Convert targets to signals
    4. Fetch 1-minute OHLCV data
    5. Backtest signals on 1-minute data (multi-timeframe via merge_asof, same as signals/backtest flow)
    
    Args:
        ml_config_path: Path to ML module config.yaml
        validation_config_path: Path to validation config.yaml
        
    Returns:
        dict with validation results, trade_ledger, backtest_result, validation_type
    """
    
    # Load configs
    validation_config = load_validation_config(validation_config_path)
    ml_config = load_ml_config(ml_config_path)
    backtest_config = validation_config['backtest_config']
    validation_type = validation_config.get('validation_type', 'classification')
    
    # Extract ML config details for logging
    data_config = ml_config.get('data', {})
    ml_timeframe = data_config.get('timeframe', '1h')
    exchange = data_config.get('exchange', 'binance').lower()
    symbol = data_config.get('symbol', 'btc').lower()
    model_type = ml_config.get('model_type', 'classification')
    
    print("=" * 80)
    print(f"TARGET VALIDATION VIA BACKTESTING")
    print("=" * 80)
    print(f"Model Type:         {model_type}")
    print(f"Validation Type:    {validation_type}")
    print(f"Target Timeframe:   {ml_timeframe}")
    print(f"Backtest Timeframe: 1min (multi-timeframe via merge_asof)")
    print(f"Exchange/Symbol:    {exchange} / {symbol}")
    print("=" * 80)
    
    # Step 1: Run ML pipeline to get dataset with targets
    print("\n[1/5] Running ML pipeline to generate targets...")
    ml_df = run_ml_pipeline(ml_config_path)
    
    initial_rows = len(ml_df)
    ml_df = ml_df.dropna(subset=['target'])
    rows_after_drop = len(ml_df)
    
    print(f"✓ ML pipeline complete: {rows_after_drop}/{initial_rows} rows with valid targets")
    print(f"  - Total columns: {len(ml_df.columns)}")
    
    # Step 2: Validate target distribution
    print("\n[2/5] Validating target distribution...")
    target_validation = validate_target_distribution(ml_df, model_type)
    stats = target_validation['stats']
    
    print(f"✓ Target validation complete:")
    print(f"  - Non-null targets: {stats.get('non_null_count', 0)}")
    print(f"  - Min: {stats.get('min', 0):.6f}")
    print(f"  - Max: {stats.get('max', 0):.6f}")
    print(f"  - Mean: {stats.get('mean', 0):.6f}")
    print(f"  - Std Dev: {stats.get('std', 0):.6f}")
    
    if model_type == "classification" and 'class_distribution' in stats:
        dist = stats['class_distribution']
        print(f"\n  Class Distribution:")
        print(f"    Buy (1):      {dist.get(1, 0):>6} ({dist.get(1, 0)/rows_after_drop*100:>5.1f}%)")
        print(f"    Neutral (0):  {dist.get(0, 0):>6} ({dist.get(0, 0)/rows_after_drop*100:>5.1f}%)")
        print(f"    Sell (-1):    {dist.get(-1, 0):>6} ({dist.get(-1, 0)/rows_after_drop*100:>5.1f}%)")
    
    if target_validation['warnings']:
        print(f"\n  ⚠ Warnings:")
        for warning in target_validation['warnings']:
            print(f"    - {warning}")
    
    if not target_validation['valid']:
        print("\n✗ Target distribution validation FAILED")
        raise ValueError("Target distribution is invalid. Check warnings above.")
    
    # Step 3: Convert targets to signals
    print("\n[3/5] Converting targets to trading signals...")
    target_config = ml_config.get('target', {})
    thresholds_used = {
        "upper_threshold": target_config.get('upper_threshold', 0.001),
        "lower_threshold": target_config.get('lower_threshold', -0.001),
    }
    if validation_type == 'classification':
        horizon = target_config.get('horizon', 1)
        signals = convert_classification_to_signals(
            ml_df,
            upper_threshold=thresholds_used["upper_threshold"],
            lower_threshold=thresholds_used["lower_threshold"],
            horizon=horizon,
        )
        signal_counts = signals['signal'].value_counts().to_dict()
        print(f"✓ Classification signals generated:")
        print(f"  - Buy signals (1):    {signal_counts.get(1, 0):>6}")
        print(f"  - No signals (0):     {signal_counts.get(0, 0):>6}")
        print(f"  - Sell signals (-1):  {signal_counts.get(-1, 0):>6}")
    
    elif validation_type == 'regression':
        # Reuse the SAME thresholds that built the classification target
        # (ml_module/config.yaml), instead of an unrelated local default --
        # this is the value that actually matters, since it's what training
        # / signal_conversion.py already treat as "what counts as a move".
        signals = convert_regression_to_signals(
            ml_df,
            upper_threshold=thresholds_used["upper_threshold"],
            lower_threshold=thresholds_used["lower_threshold"],
        )
        signal_counts = signals['signal'].value_counts().to_dict()
        print(f"✓ Regression signals generated "
              f"(upper={thresholds_used['upper_threshold']}, lower={thresholds_used['lower_threshold']}):")
        print(f"  - Buy signals (1):    {signal_counts.get(1, 0):>6}")
        print(f"  - No signals (0):     {signal_counts.get(0, 0):>6}")
        print(f"  - Sell signals (-1):  {signal_counts.get(-1, 0):>6}")
    
    # Step 4: Get 1-minute OHLCV data for backtest
    print("\n[4/5] Fetching 1-minute OHLCV data for backtest...")
    
    start_date = data_config.get('start_date')
    end_date = data_config.get('end_date')
    
    # Parse dates if strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Fetch 1-minute data (same as backtest/main.py workflow)
    ohlcv_result = get_data(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe='1min'
    )
    ohlcv_1m = ohlcv_result['resampled']
    
    print(f"✓ 1-minute OHLCV fetched: {len(ohlcv_1m)} bars")
    print(f"  - Date range: {ohlcv_1m['datetime'].min()} to {ohlcv_1m['datetime'].max()}")
    
    # Alignment check: ensure signal dates are within data range
    signal_min_dt = signals['datetime'].min()
    signal_max_dt = signals['datetime'].max()
    data_min_dt = ohlcv_1m['datetime'].min()
    data_max_dt = ohlcv_1m['datetime'].max()
    
    if signal_max_dt < data_min_dt or signal_min_dt > data_max_dt:
        raise ValueError(
            f"Signal and OHLCV data don't overlap!\n"
            f"  Signals: {signal_min_dt} to {signal_max_dt}\n"
            f"  OHLCV:   {data_min_dt} to {data_max_dt}"
        )
    
    print(f"\n  ✓ Data/signal alignment check passed")
    
    # Step 5: Run backtest
    print("\n[5/5] Running backtest with generated signals...")
    print(f"  Backtest config:")
    print(f"    - Initial balance: ${backtest_config['initial_balance']:,.0f}")
    print(f"    - Position size: {backtest_config['position_size']['value']}% per trade")
    print(f"    - Take profit: {backtest_config['take_profit']['value']}%")
    print(f"    - Stop loss: {backtest_config['stop_loss']['value']}%")
    print(f"    - Commission: {backtest_config['commission']}%")
    print(f"    - Slippage: {backtest_config['slippage']}%")
    
    backtest_result = run_backtest(ohlcv_1m, signals, backtest_config)
    
    # Extract results
    trade_ledger = backtest_result['trade_ledger']
    final_balance = backtest_result['final_balance']
    total_net_profit = backtest_result['total_net_profit']
    total_trades = backtest_result['total_trades']
    win_loss = backtest_result['win_loss']

    # Enrich the trade ledger with the return value that triggered each
    # trade and how far it sat from the threshold -- this way the SAVED
    # trade_ledger_*.csv already has it, no separate join/script needed
    # later. entry_time is always the 1-minute bar the triggering signal
    # got aligned to (merge_asof backward, inside run_backtest), so the
    # nearest signal at-or-before entry_time is the one that opened it.
    if len(trade_ledger) > 0:
        signal_events = signals[signals['signal'] != 0][
            ['datetime', 'return_value', 'threshold_used']
        ].sort_values('datetime')
        trade_ledger = trade_ledger.sort_values('entry_time').reset_index(drop=True)
        trade_ledger = pd.merge_asof(
            trade_ledger, signal_events,
            left_on='entry_time', right_on='datetime',
            direction='backward',
        )
        trade_ledger = trade_ledger.drop(columns=['datetime'])
        trade_ledger['distance_from_threshold_pct'] = (
            (trade_ledger['return_value'] - trade_ledger['threshold_used']).abs()
            / trade_ledger['threshold_used'].abs() * 100
        )

        print(f"\n  Long/short vs threshold (from enriched trade ledger):")
        for direction_label in ['long', 'short']:
            subset = trade_ledger[trade_ledger['direction'] == direction_label]
            if len(subset) == 0:
                print(f"    {direction_label.upper():<6}: 0 trades")
                continue
            avg_return = subset['return_value'].mean()
            threshold_for_side = subset['threshold_used'].iloc[0]
            avg_distance_pct = subset['distance_from_threshold_pct'].mean()
            print(f"    {direction_label.upper():<6}: {len(subset):>4} trades | "
                  f"avg return_value={avg_return:.6f} vs threshold={threshold_for_side:.6f} "
                  f"(avg {avg_distance_pct:.1f}% away from threshold -- "
                  f"{'CLOSE, consider tuning threshold' if avg_distance_pct < 20 else 'comfortably clear'})")
    
    initial_balance = backtest_config['initial_balance']
    roi_pct = (total_net_profit / initial_balance) * 100 if initial_balance > 0 else 0
    
    print(f"✓ Backtest complete!")
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Initial Balance:      ${initial_balance:,.2f}")
    print(f"Final Balance:        ${final_balance:,.2f}")
    print(f"Total Net Profit:     ${total_net_profit:,.2f}")
    print(f"ROI:                  {roi_pct:>7.2f}%")
    print(f"\nTrade Statistics:")
    print(f"  Total Trades:       {total_trades:>6}")
    print(f"  Wins:               {win_loss['wins']:>6}")
    print(f"  Losses:             {win_loss['losses']:>6}")
    print(f"  Win Rate:           {win_loss['win_rate']*100:>6.2f}%")
    
    if total_trades > 0 and len(trade_ledger) > 0:
        winning_trades = trade_ledger[trade_ledger['net_pnl'] > 0]
        losing_trades = trade_ledger[trade_ledger['net_pnl'] <= 0]
        
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        print(f"  Avg Win:            ${avg_win:>11,.2f}")
        print(f"  Avg Loss:           ${avg_loss:>11,.2f}")
        
        if avg_loss != 0 and len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = abs(avg_win * win_loss['wins'] / (avg_loss * win_loss['losses']))
            print(f"  Profit Factor:      {profit_factor:>6.2f}x")
    
    max_drawdown = backtest_result['drawdown_series'].min()
    print(f"  Max Drawdown:       {max_drawdown*100:>6.2f}%")
    print(f"{'='*80}")
    
    # Validation verdict
    if roi_pct > 5:
        verdict = "✓ PROFITABLE - Targets appear VALID!"
    elif roi_pct > 0:
        verdict = "⚠ WEAKLY PROFITABLE - Targets may be valid but weak"
    elif roi_pct > -5:
        verdict = "⚠ BREAK-EVEN - Targets marginal, needs tuning"
    else:
        verdict = "✗ LOSING MONEY - Targets need significant improvement"
    
    print(f"\nVERDICT: {verdict}\n")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'validation_type': validation_type,
        'target_timeframe': ml_timeframe,
        'exchange_symbol': f"{exchange}/{symbol}",
        'dataset_rows': rows_after_drop,
        'thresholds_used': thresholds_used,
        'target_validation': target_validation,
        'signal_counts': signal_counts,
        'backtest_summary': {
            'initial_balance': float(initial_balance),
            'final_balance': float(final_balance),
            'total_net_profit': float(total_net_profit),
            'roi_percent': float(roi_pct),
            'total_trades': int(total_trades),
            'wins': int(win_loss['wins']),
            'losses': int(win_loss['losses']),
            'win_rate': float(win_loss['win_rate']),
            'max_drawdown': float(max_drawdown),
        },
        'verdict': verdict,
    }
    
    return results, trade_ledger, backtest_result, validation_type


def save_results(results: dict, trade_ledger: pd.DataFrame, backtest_result: dict, 
                validation_type: str, output_dir: str = None) -> None:
    """
    Save validation results to files in separate folders per validation type.
    
    Saves:
    - validation_summary_{timestamp}.json: All results and verdict
    - trade_ledger_{timestamp}.csv: Individual trades
    - equity_curve_{timestamp}.csv: Balance over time
    - drawdown_{timestamp}.csv: Drawdown over time
    """
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "outputs"
    
    # Create subfolder for validation type (classification or regression)
    output_dir = os.path.join(output_dir, validation_type)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, f"validation_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Summary saved: {summary_path}")
    
    # Save trade ledger as CSV (only if trades exist)
    if not trade_ledger.empty:
        ledger_path = os.path.join(output_dir, f"trade_ledger_{timestamp}.csv")
        trade_ledger.to_csv(ledger_path, index=False)
        print(f"✓ Trade ledger saved: {ledger_path} ({len(trade_ledger)} trades)")
    else:
        print(f"⚠ No trades executed - trade ledger empty")
    
    # Save equity curve as CSV
    equity_path = os.path.join(output_dir, f"equity_curve_{timestamp}.csv")
    equity_df = pd.DataFrame({
        'datetime': backtest_result['equity_curve'].index,
        'balance': backtest_result['equity_curve'].values
    })
    equity_df.to_csv(equity_path, index=False)
    print(f"✓ Equity curve saved: {equity_path}")
    
    # Save drawdown as CSV
    drawdown_path = os.path.join(output_dir, f"drawdown_{timestamp}.csv")
    drawdown_df = pd.DataFrame({
        'datetime': backtest_result['drawdown_series'].index,
        'drawdown': backtest_result['drawdown_series'].values
    })
    drawdown_df.to_csv(drawdown_path, index=False)
    print(f"✓ Drawdown saved: {drawdown_path}")


if __name__ == "__main__":
    import os
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    try:
        # Paths
        here = os.path.dirname(os.path.abspath(__file__))
        ml_config_path = os.path.join(here, "..", "ml_module", "config.yaml")
        validation_config_path = os.path.join(here, "config.yaml")
        
        # Verify configs exist
        if not os.path.exists(ml_config_path):
            raise FileNotFoundError(f"ML config not found: {ml_config_path}")
        if not os.path.exists(validation_config_path):
            raise FileNotFoundError(f"Validation config not found: {validation_config_path}")
        
        print("\n" + "="*80)
        print("CRYPTO PIPELINE TARGET VALIDATION")
        print("="*80)
        print(f"ML Config:          {ml_config_path}")
        print(f"Validation Config:  {validation_config_path}")
        print("="*80)
        
        # Run validation
        results, trade_ledger, backtest_result, validation_type = validate_targets(
            ml_config_path=ml_config_path,
            validation_config_path=validation_config_path
        )
        
        # Save results
        output_dir = os.path.join(here, "outputs")
        save_results(results, trade_ledger, backtest_result, validation_type, output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("\n" + "="*80)
        print(f"✓ VALIDATION COMPLETE!")
        print("="*80)
        print(f"Results saved to: ./outputs/{validation_type}/{timestamp}/")
        print(f"Verdict: {results['verdict']}")
        print("="*80 + "\n")
        
        sys.exit(0)
    
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ VALIDATION FAILED")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        print("="*80 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)