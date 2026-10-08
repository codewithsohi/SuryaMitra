"""
Domain-Specific Space Weather Evaluation Metrics.

In solar physics and operational space weather forecasting (e.g. NOAA SWPC, ISRO Aditya-L1),
forecasts are evaluated on a 2x2 contingency table under severe class imbalance:
    - True Positives (TP): Flare predicted (>= M-class), flare occurred.
    - False Positives (FP): Flare predicted, no flare (False Alarm).
    - False Negatives (FN): No flare predicted, flare occurred (Missed Flare).
    - True Negatives (TN): No flare predicted, no flare occurred.

Standard accuracy is misleading because predicting the majority negative class
yields >98% accuracy with zero operational capability. The primary skill score
in solar flare prediction is the True Skill Statistic (TSS).
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    brier_score_loss,
    average_precision_score,
    roc_auc_score
)


def compute_contingency_table(y_true, y_pred):
    """
    Computes standard 2x2 contingency table elements: TP, FP, FN, TN.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        tuple: (tp, fp, fn, tn) as integers.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    return tp, fp, fn, tn


def true_skill_statistic(y_true, y_pred):
    """
    Computes True Skill Statistic (TSS):
        TSS = Recall - FPR = (TP / (TP + FN)) - (FP / (FP + TN))

    Ranges from -1 to +1. 0 indicates no skill (random guessing),
    +1 indicates perfect forecast skill. Independent of class prevalence.
    """
    tp, fp, fn, tn = compute_contingency_table(y_true, y_pred)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return tpr - fpr


def heidke_skill_score(y_true, y_pred):
    """
    Computes Heidke Skill Score (HSS):
        HSS = 2 * (TP * TN - FP * FN) / ((TP + FN)(FN + TN) + (TP + FP)(FP + TN))

    Measures accuracy of the forecast relative to that of random chance.
    Ranges from -inf to +1; 0 indicates no skill over random chance.
    """
    tp, fp, fn, tn = compute_contingency_table(y_true, y_pred)
    
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def find_optimal_threshold(y_true, y_prob, metric="tss", num_thresholds=100):
    """
    Sweeps probability thresholds tau in [0.01, 0.99] to find the threshold
    that explicitly maximizes operational skill (default: TSS).

    Args:
        y_true: Ground truth binary labels.
        y_prob: Continuous predicted probabilities for positive class (>= M-class).
        metric: 'tss' or 'hss'.
        num_thresholds: Number of discrete threshold evaluation steps.

    Returns:
        tuple: (best_threshold, best_score, score_history)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.01, 0.99, num_thresholds)

    best_score = -float("inf")
    best_threshold = 0.5
    history = []

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        if metric.lower() == "tss":
            score = true_skill_statistic(y_true, y_pred)
        elif metric.lower() == "hss":
            score = heidke_skill_score(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Use 'tss' or 'hss'.")

        history.append((tau, score))
        if score > best_score:
            best_score = score
            best_threshold = tau

    return best_threshold, best_score, history


def evaluate_forecast(y_true, y_prob, threshold=0.5):
    """
    Full comprehensive evaluation suite returning all space weather
    and standard machine learning classification metrics.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted flare probabilities.
        threshold: Decision threshold for binary classification.

    Returns:
        dict: Detailed dictionary of operational and statistical metrics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp, fp, fn, tn = compute_contingency_table(y_true, y_pred)
    tss = true_skill_statistic(y_true, y_pred)
    hss = heidke_skill_score(y_true, y_pred)

    brier = brier_score_loss(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tss": float(tss),
        "hss": float(hss),
        "brier_score": float(brier),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "recall": float(recall),
        "precision": float(precision),
        "fpr": float(fpr),
    }
