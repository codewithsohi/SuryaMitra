import os
import sys
sys.path.insert(0, os.path.abspath("."))

from src.metrics import (
    compute_contingency_table,
    true_skill_statistic,
    heidke_skill_score,
    evaluate_forecast,
    find_optimal_threshold
)

def test_metrics():
    y_true = [1, 0, 1, 0, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 0, 0, 1, 0]
    tp, fp, fn, tn = compute_contingency_table(y_true, y_pred)
    assert tp == 2
    assert fp == 0
    assert fn == 1
    assert tn == 5

    tss = true_skill_statistic(y_true, y_pred)
    # TPR = 2/3, FPR = 0/5 -> TSS = 2/3 ~ 0.6667
    assert abs(tss - (2/3)) < 1e-4

    hss = heidke_skill_score(y_true, y_pred)
    assert hss > 0.0

    y_prob = [0.9, 0.1, 0.4, 0.2, 0.05, 0.15, 0.85, 0.3]
    eval_res = evaluate_forecast(y_true, y_prob, threshold=0.5)
    assert "tss" in eval_res
    assert "hss" in eval_res
    assert "brier_score" in eval_res
    assert "pr_auc" in eval_res

    best_tau, best_tss, _ = find_optimal_threshold(y_true, y_prob, metric="tss")
    assert 0.0 < best_tau < 1.0

    print("All tests passed successfully! TSS:", tss, "HSS:", hss, "Best Threshold:", best_tau)

if __name__ == "__main__":
    test_metrics()
