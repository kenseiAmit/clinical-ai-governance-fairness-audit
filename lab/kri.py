"""
kri.py — Translate the three KRI definitions from D7 into runnable code.

Why this lab matters. A Key Risk Indicator defined only in prose is a wish,
not a control. A KRI defined in code is reproducible, testable, and
auditable. The difference is what makes "post-launch monitoring" credible
to a regulator.

Each function returns the same shape: a status of Green, Amber, or Red,
the metric value, and any context fields the dashboard would need.

Run the unit tests:
    cd clinical-ai-governance-fairness-audit/
    pytest lab/tests/test_kri.py -v
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from .audit import compute_subgroup_metrics


def kri_subgroup_fnr_drift(
    y_true,
    y_pred,
    sensitive_features: pd.DataFrame,
    min_n_positives: int = 20,
) -> dict:
    """
    Implement the KRI defined as KRI-1 in the D7 dashboard tab.
    """
    sub_df = compute_subgroup_metrics(y_true, y_pred, sensitive_features)
    valid_df = sub_df[
        (sub_df["n_positives"] >= min_n_positives) &
        (sub_df["FNR"].notna())
    ]
    if len(valid_df) < 2:
        return {
            "kri_id": "KRI-1",
            "status": "Insufficient",
            "gap_pp": None,
            "worst_cell": None,
            "best_cell": None,
            "n_cells": len(valid_df),
            "note": f"Insufficient qualifying cells ({len(valid_df)} < 2) with >= {min_n_positives} positives.",
        }

    worst_idx = valid_df["FNR"].idxmax()
    best_idx = valid_df["FNR"].idxmin()
    max_fnr = float(valid_df.loc[worst_idx, "FNR"])
    min_fnr = float(valid_df.loc[best_idx, "FNR"])
    gap_pp = float((max_fnr - min_fnr) * 100.0)

    if gap_pp <= 8.0:
        status = "Green"
    elif gap_pp <= 15.0:
        status = "Amber"
    else:
        status = "Red"

    return {
        "kri_id": "KRI-1",
        "status": status,
        "gap_pp": gap_pp,
        "worst_cell": worst_idx,
        "best_cell": best_idx,
        "n_cells": len(valid_df),
        "note": f"FNR gap between worst cell {worst_idx} ({max_fnr*100:.1f}%) and best cell {best_idx} ({min_fnr*100:.1f}%) is {gap_pp:.1f} pp.",
    }


def kri_override_health(
    override_rate: float,
    structured_share: float,
) -> dict:
    """
    Implement the KRI defined as KRI-2 in the D7 dashboard tab.
    """
    if override_rate <= 0.10 and structured_share >= 0.80:
        status = "Green"
        reason = "Override rate <= 10% and structured share >= 80%."
    elif override_rate > 0.15 or structured_share < 0.60:
        status = "Red"
        reasons = []
        if override_rate > 0.15:
            reasons.append(f"Override rate ({override_rate*100:.1f}%) > 15%")
        if structured_share < 0.60:
            reasons.append(f"Structured share ({structured_share*100:.1f}%) < 60%")
        reason = "; ".join(reasons)
    else:
        status = "Amber"
        reasons = []
        if override_rate > 0.10:
            reasons.append(f"Override rate ({override_rate*100:.1f}%) > 10%")
        if structured_share < 0.80:
            reasons.append(f"Structured share ({structured_share*100:.1f}%) < 80%")
        reason = "; ".join(reasons)

    return {
        "kri_id": "KRI-2",
        "status": status,
        "override_rate": override_rate,
        "structured_share": structured_share,
        "reason": reason,
    }


def kri_inference_anomaly(
    anomaly_count: int,
    total_inferences: int,
) -> dict:
    """
    Implement the KRI defined as KRI-3 in the D7 dashboard tab.
    """
    if total_inferences == 0:
        return {
            "kri_id": "KRI-3",
            "status": "Insufficient",
            "per_1000": None,
            "anomaly_count": anomaly_count,
            "total_inferences": total_inferences,
            "note": "Total inferences is 0; rate cannot be computed.",
        }

    rate_per_1000 = float((anomaly_count / total_inferences) * 1000.0)

    if rate_per_1000 < 1.0:
        status = "Green"
    elif rate_per_1000 <= 5.0:
        status = "Amber"
    else:
        status = "Red"

    return {
        "kri_id": "KRI-3",
        "status": status,
        "per_1000": rate_per_1000,
        "anomaly_count": anomaly_count,
        "total_inferences": total_inferences,
        "note": f"{anomaly_count} anomalies across {total_inferences} inferences ({rate_per_1000:.2f} per 1000 inferences).",
    }
