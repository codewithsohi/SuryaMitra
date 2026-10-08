"""
Training Pipeline for Space Weather Baseline Models.

Trains Gradient Boosted Decision Tree (GBDT) baselines (LightGBM & XGBoost)
on SWAN-SF magnetic feature tables with:
  1. Severe class imbalance handling via dynamic positive weight scaling (scale_pos_weight).
  2. Operational space weather threshold tuning maximizing True Skill Statistic (TSS).
  3. Domain metric evaluation (TSS, HSS, Brier Score, PR-AUC).
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

from src.metrics import (
    evaluate_forecast,
    find_optimal_threshold,
    compute_contingency_table,
    true_skill_statistic,
    heidke_skill_score
)


def load_dataset(data_path):
    """
    Loads tabular feature dataset and extracts features X and binary labels y.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature table not found at: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path} with shape {df.shape}")

    # Exclude metadata columns
    exclude_cols = {"file_id", "label"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Handle any potential NaNs via column medians
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    y = df["label"].astype(int).values

    print(f"Features: {len(feature_cols)} | Total Samples: {len(y)}")
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    pos_rate = (pos_count / len(y)) * 100
    print(f"Positive (>= M-class): {pos_count} ({pos_rate:.2f}%) | Negative: {neg_count}")

    return X, y, feature_cols


def train_baseline(
    data_path="data/processed/partition1_tabular.csv",
    model_type="lightgbm",
    val_size=0.2,
    random_state=42,
    optimize_threshold=True,
    output_model_dir="models"
):
    """
    Trains tabular baseline, optimizes operational decision threshold, and evaluates.
    """
    os.makedirs(output_model_dir, exist_ok=True)
    X, y, feature_names = load_dataset(data_path)

    # Chronological or stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")

    if model_type.lower() == "lightgbm":
        print("\n--- Training LightGBM Classifier ---")
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            verbose=-1
        )
        model.fit(X_train, y_train)

    elif model_type.lower() == "xgboost":
        print("\n--- Training XGBoost Classifier ---")
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'lightgbm' or 'xgboost'.")

    # Predict continuous probabilities for flare occurrence
    val_probs = model.predict_proba(X_val)[:, 1]

    # Evaluate at default threshold (0.5)
    default_metrics = evaluate_forecast(y_val, val_probs, threshold=0.5)
    print("\n[Default Threshold: 0.50]")
    print(f"  TSS: {default_metrics['tss']:.4f} | HSS: {default_metrics['hss']:.4f} | PR-AUC: {default_metrics['pr_auc']:.4f}")
    print(f"  Contingency: TP={default_metrics['tp']}, FP={default_metrics['fp']}, FN={default_metrics['fn']}, TN={default_metrics['tn']}")

    # Optimize threshold on validation set maximizing TSS
    if optimize_threshold:
        best_tau, best_tss, _ = find_optimal_threshold(y_val, val_probs, metric="tss")
        opt_metrics = evaluate_forecast(y_val, val_probs, threshold=best_tau)
        print(f"\n[Optimized Threshold: {best_tau:.3f} (Max TSS)]")
        print(f"  TSS: {opt_metrics['tss']:.4f} | HSS: {opt_metrics['hss']:.4f} | PR-AUC: {opt_metrics['pr_auc']:.4f}")
        print(f"  Recall (TPR): {opt_metrics['recall']:.4f} | FPR: {opt_metrics['fpr']:.4f}")
        print(f"  Contingency: TP={opt_metrics['tp']}, FP={opt_metrics['fp']}, FN={opt_metrics['fn']}, TN={opt_metrics['tn']}")
    else:
        best_tau = 0.5
        opt_metrics = default_metrics

    # Save model artifact
    save_path = os.path.join(output_model_dir, f"{model_type}_baseline.joblib")
    joblib.dump({"model": model, "threshold": best_tau, "features": feature_names}, save_path)
    print(f"\nTrained model and optimal threshold saved to: {save_path}")

    return model, opt_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train space weather flare forecasting baseline")
    parser.add_argument("--data", type=str, default="data/processed/partition1_tabular.csv", help="Path to feature table")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "xgboost"], help="Model type")
    parser.add_argument("--no-optimize", action="store_true", help="Disable threshold optimization")
    args = parser.parse_args()

    train_baseline(
        data_path=args.data,
        model_type=args.model,
        optimize_threshold=not args.no_optimize
    )
