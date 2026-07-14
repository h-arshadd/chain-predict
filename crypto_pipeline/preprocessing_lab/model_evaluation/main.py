# preprocessing_lab/model_evaluation/main.py

"""
main.py
-------
Trains identical simple models on every preprocessing method's output,
so the preprocessing method is the only thing that varies (same split,
same models, same hyperparameters, same random seed -- per the task's
"Model Evaluation" requirement).

For each (method, target_type) in config.yaml:
    1. reads preprocessing_lab/outputs/<method>/<target_type>/transformed.csv
    2. does a time-based train/val/test split (never shuffled)
    3. trains every model in models.py for that target_type
    4. scores it (regression: MAE, RMSE | classification: accuracy, precision)
    5. saves per-method metrics + a combined comparison table

No deep learning here on purpose -- just linear/logistic regression,
xgboost, and lightgbm (whichever are installed). Feature columns are read
straight off the CSV (everything except datetime/target), so it matches
whatever preprocessing_lab produced -- no re-deriving column lists here.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
)

from crypto_pipeline.preprocessing_lab.model_evaluation.models import REGRESSION_MODELS, CLASSIFICATION_MODELS


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def time_based_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    """
    Splits a dataframe by row order (already chronological, no shuffling)
    into train/val/test. test_ratio is implied (1 - train - val).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def evaluate_regression(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }


def evaluate_classification(model, X_train, y_train, X_test, y_test, needs_label_remap: bool) -> dict:
    if needs_label_remap:
        # xgboost/lightgbm need 0..n labels, not -1/0/1 -- remap here only,
        # sklearn models (logistic regression) handle -1/0/1 natively
        label_map = {-1: 0, 0: 1, 1: 2}
        inverse_map = {v: k for k, v in label_map.items()}
        y_train_mapped = y_train.map(label_map)
        model.fit(X_train, y_train_mapped)
        preds_mapped = model.predict(X_test)
        preds = pd.Series(preds_mapped).map(inverse_map).values
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average="macro", zero_division=0)),
    }


def run_one(preprocessing_lab_dir: str, method_name: str, target_type: str, split_config: dict) -> dict:
    """Loads one method's transformed.csv, trains every model for target_type, returns metrics."""

    data_path = os.path.join(
        preprocessing_lab_dir, "outputs", method_name, target_type, "transformed.csv"
    )
    if not os.path.exists(data_path):
        return {"error": f"transformed.csv not found: {data_path}"}

    df = pd.read_csv(data_path)

    feature_cols = [c for c in df.columns if c not in ("datetime", "target")]

    train, val, test = time_based_split(df, split_config["train_ratio"], split_config["val_ratio"])
    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    models = REGRESSION_MODELS if target_type == "regression" else CLASSIFICATION_MODELS

    method_results = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "models": {},
    }

    for model_name, make_model in models.items():
        model = make_model()
        needs_label_remap = model_name in ("xgboost", "lightgbm")

        if target_type == "regression":
            metrics = evaluate_regression(model, X_train, y_train, X_test, y_test)
        else:
            metrics = evaluate_classification(model, X_train, y_train, X_test, y_test, needs_label_remap)

        method_results["models"][model_name] = metrics
        print(f"  {model_name}: {metrics}")

    return method_results


def run(config_path: str):
    here = os.path.dirname(os.path.abspath(config_path))
    preprocessing_lab_dir = os.path.dirname(here)  # model_evaluation/ -> preprocessing_lab/
    config = load_config(config_path)

    methods = config["methods"]
    if isinstance(methods, str):
        methods = [methods]

    target_types = config.get("target_types", ["regression"])
    if isinstance(target_types, str):
        target_types = [target_types]

    split_config = config["split"]
    out_dir = os.path.join(here, config["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for target_type in target_types:
        all_results[target_type] = {}

        for method_name in methods:
            print(f"\n=== {method_name} | {target_type} ===")
            result = run_one(preprocessing_lab_dir, method_name, target_type, split_config)
            all_results[target_type][method_name] = result

        # save one comparison file per target_type
        comparison_path = os.path.join(out_dir, f"comparison_{target_type}.json")
        with open(comparison_path, "w") as f:
            json.dump(all_results[target_type], f, indent=2, default=str)
        print(f"\nSaved: {comparison_path}")

    # flatten into a single readable table too (method, target_type, model, metrics)
    rows = []
    for target_type, methods_dict in all_results.items():
        for method_name, result in methods_dict.items():
            if "error" in result:
                continue
            for model_name, metrics in result["models"].items():
                rows.append({"method": method_name, "target_type": target_type, "model": model_name, **metrics})

    table = pd.DataFrame(rows)
    table_path = os.path.join(out_dir, "comparison_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")

    return all_results


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "config.yaml")
    run(config_path)