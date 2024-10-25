"""
fairness_metrics.py — Implement fairness metrics from confusion-matrix primitives.

You will use these in the technical audit (Task 4). Do NOT import these from
fairlearn or sklearn.metrics — implement them yourself. The point of this lab
is to make sure you understand what each metric *is* before you start
interpreting its values.

Run the unit tests to check your work:
    cd clinical-ai-governance-fairness-audit/
    pytest lab/tests/test_fairness_metrics.py -v
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def false_negative_rate(y_true, y_pred) -> float:
    """
    Compute the false-negative rate.

    Definition. The false-negative rate is the fraction of actual positives
    that the model failed to flag — among patients who truly have diabetes,
    the share the model predicted as not-at-risk.

    Why this matters in HealthGuard. A false negative is a missed diagnosis,
    the dominant clinical harm we are trying to avoid. Your D8 launch
    decision will hinge on the false-negative pattern across subgroups.

    Edge case. If the input has no actual positives, the metric is
    mathematically undefined. Return float('nan') to signal "undefined".

    Args:
        y_true: ground-truth labels (0 or 1).
        y_pred: predicted labels after thresholding (0 or 1).

    Returns:
        A single number between zero and one, or float('nan') when
        the metric is undefined.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    positives = (y_true_arr == 1)
    n_pos = np.sum(positives)
    if n_pos == 0:
        return float('nan')

    false_negatives = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
    return float(false_negatives / n_pos)


def false_positive_rate(y_true, y_pred) -> float:
    """
    Compute the false-positive rate.

    Definition. The fraction of actual negatives the model incorrectly
    flagged — among patients who do NOT have diabetes, the share the model
    predicted as at-risk.

    Why this matters. False-positive rate measures unnecessary risk flags.
    A high false-positive rate means clinicians are alerted on patients who
    do not have diabetes — alert fatigue, wasted referrals, lost trust in
    the tool. It is the symmetric counterpart to false-negative rate.

    Edge case. Return float('nan') when there are no actual negatives in the input.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    negatives = (y_true_arr == 0)
    n_neg = np.sum(negatives)
    if n_neg == 0:
        return float('nan')

    false_positives = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
    return float(false_positives / n_neg)


def selection_rate(y_pred) -> float:
    """
    Compute the selection rate.

    Definition. The fraction of inputs the model flagged as positive,
    regardless of whether the prediction was correct.

    Why this matters. Selection rate is the simplest fairness baseline —
    is the model recommending action at similar rates across groups?
    It is the basis for *demographic parity*, the parity-style fairness
    definition that ignores ground truth.
    """
    y_pred_arr = np.asarray(y_pred)
    if len(y_pred_arr) == 0:
        return float('nan')
    return float(np.mean(y_pred_arr == 1))


def demographic_parity_difference(y_pred, sensitive) -> dict:
    """
    Compute the demographic-parity difference across sensitive groups.

    Definition. The largest gap in selection rate between any two groups
    in the input — the most-favored group's selection rate minus the
    least-favored group's selection rate.

    Args:
        y_pred: predicted labels (0 or 1).
        sensitive: group label for each row (same length as y_pred).

    Returns:
        A dictionary that includes the difference, the label of the
        most-favored group, the label of the least-favored group, and a
        per-group breakdown of selection rates.
    """
    y_pred_arr = np.asarray(y_pred)
    sensitive_arr = np.asarray(sensitive)

    groups = pd.Series(y_pred_arr).groupby(sensitive_arr)
    by_group = {grp: float(np.mean(vals == 1)) for grp, vals in groups}

    if not by_group:
        return {
            "difference": float('nan'),
            "max_group": None,
            "min_group": None,
            "by_group": {},
        }

    max_grp = max(by_group, key=lambda k: by_group[k])
    min_grp = min(by_group, key=lambda k: by_group[k])
    diff = float(by_group[max_grp] - by_group[min_grp])

    return {
        "difference": diff,
        "max_group": max_grp,
        "min_group": min_grp,
        "by_group": by_group,
    }


def equalized_odds_difference(y_true, y_pred, sensitive) -> dict:
    """
    Compute the equalized-odds difference across sensitive groups.

    Definition. The larger of two cross-group gaps:
        - the gap in false-positive rate between groups, and
        - the gap in false-negative rate between groups.

    Returns:
        A dictionary that includes the headline equalized-odds difference,
        the false-negative-rate gap and false-positive-rate gap separately,
        an indicator of which of the two is the larger driver, and the
        per-group breakdowns of both rates.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    sensitive_arr = np.asarray(sensitive)

    unique_groups = np.unique(sensitive_arr)
    fnr_by_group = {}
    fpr_by_group = {}

    for grp in unique_groups:
        mask = (sensitive_arr == grp)
        fnr_by_group[grp] = false_negative_rate(y_true_arr[mask], y_pred_arr[mask])
        fpr_by_group[grp] = false_positive_rate(y_true_arr[mask], y_pred_arr[mask])

    valid_fnr = [v for v in fnr_by_group.values() if not math.isnan(v)]
    valid_fpr = [v for v in fpr_by_group.values() if not math.isnan(v)]

    fnr_diff = float(max(valid_fnr) - min(valid_fnr)) if valid_fnr else float('nan')
    fpr_diff = float(max(valid_fpr) - min(valid_fpr)) if valid_fpr else float('nan')

    if math.isnan(fnr_diff) and math.isnan(fpr_diff):
        difference = float('nan')
        driver = "None"
    elif math.isnan(fnr_diff):
        difference = fpr_diff
        driver = "FPR"
    elif math.isnan(fpr_diff):
        difference = fnr_diff
        driver = "FNR"
    else:
        if fpr_diff > fnr_diff:
            difference = fpr_diff
            driver = "FPR"
        else:
            difference = fnr_diff
            driver = "FNR"

    return {
        "difference": difference,
        "fnr_difference": fnr_diff,
        "fpr_difference": fpr_diff,
        "driver": driver,
        "fnr_by_group": fnr_by_group,
        "fpr_by_group": fpr_by_group,
    }
