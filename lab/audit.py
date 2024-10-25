"""
audit.py — Subgroup, intersectional, and proxy-pathway analysis.

These functions assemble the metrics from fairness_metrics.py into the kind
of analysis a real AI Risk Officer would run.

Run the unit tests to check your work:
    cd clinical-ai-governance-fairness-audit/
    pytest lab/tests/test_audit.py -v
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# You will import your implementations from the same package
from .fairness_metrics import (
    false_negative_rate, false_positive_rate, selection_rate,
)


def compute_subgroup_metrics(
    y_true,
    y_pred,
    sensitive_features,
) -> pd.DataFrame:
    """
    Compute per-subgroup metrics and return them as a table.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if isinstance(sensitive_features, pd.DataFrame):
        group_df = sensitive_features.copy()
        group_cols = list(group_df.columns)
        group_df["_idx"] = np.arange(len(group_df))
        grouped = group_df.groupby(group_cols, observed=False)
        index_names = group_cols
        rows = []
        indices = []
        for name, group in grouped:
            idx = group["_idx"].to_numpy()
            y_t = y_true_arr[idx]
            y_p = y_pred_arr[idx]
            n_tot = len(y_t)
            n_pos = int((y_t == 1).sum())
            sel_r = selection_rate(y_p)
            fnr = false_negative_rate(y_t, y_p)
            fpr = false_positive_rate(y_t, y_p)
            acc = float((y_t == y_p).mean()) if n_tot > 0 else np.nan
            rows.append({
                "n_total": n_tot,
                "n_positives": n_pos,
                "selection_rate": sel_r,
                "FNR": fnr,
                "FPR": fpr,
                "accuracy": acc,
            })
            indices.append(name if isinstance(name, tuple) else (name,))
        multi_index = pd.MultiIndex.from_tuples(indices, names=index_names)
        return pd.DataFrame(rows, index=multi_index)
    else:
        if isinstance(sensitive_features, pd.Series):
            s = sensitive_features
            name = s.name if s.name is not None else "subgroup"
        else:
            s = pd.Series(sensitive_features, name="subgroup")
            name = s.name
        
        group_df = pd.DataFrame({"group": s, "_idx": np.arange(len(s))})
        grouped = group_df.groupby("group", observed=False)
        rows = []
        indices = []
        for grp_val, group in grouped:
            idx = group["_idx"].to_numpy()
            y_t = y_true_arr[idx]
            y_p = y_pred_arr[idx]
            n_tot = len(y_t)
            n_pos = int((y_t == 1).sum())
            sel_r = selection_rate(y_p)
            fnr = false_negative_rate(y_t, y_p)
            fpr = false_positive_rate(y_t, y_p)
            acc = float((y_t == y_p).mean()) if n_tot > 0 else np.nan
            rows.append({
                "n_total": n_tot,
                "n_positives": n_pos,
                "selection_rate": sel_r,
                "FNR": fnr,
                "FPR": fpr,
                "accuracy": acc,
            })
            indices.append(grp_val)
        idx_obj = pd.Index(indices, name=name)
        return pd.DataFrame(rows, index=idx_obj)


def find_most_harmed_subgroup(
    subgroup_df: pd.DataFrame,
    metric_name: str = "FNR",
    min_n_positives: int = 20,
    higher_is_worse: bool = True,
) -> Optional[dict]:
    """
    Return the subgroup row with the worst value of `metric_name`,
    AMONG subgroups that meet a minimum-positive-count threshold.
    """
    filtered = subgroup_df[
        (subgroup_df["n_positives"] >= min_n_positives) &
        (subgroup_df[metric_name].notna())
    ]
    if filtered.empty:
        return None

    if higher_is_worse:
        worst_idx = filtered[metric_name].idxmax()
    else:
        worst_idx = filtered[metric_name].idxmin()

    row = filtered.loc[worst_idx]
    return {
        "subgroup": worst_idx,
        "metric_name": metric_name,
        "metric_value": float(row[metric_name]),
        "n_positives": int(row["n_positives"]),
        "n_total": int(row["n_total"]),
    }


def assess_proxy_pathway(
    intersectional_df: pd.DataFrame,
    primary_axis_value,
    secondary_axis_low,
    secondary_axis_high,
    metric_name: str = "FNR",
) -> dict:
    """
    Compare the within-primary-axis disparity to the across-primary-
    axis baseline, and return a verdict on whether a proxy pathway is
    consistent with the data.
    """
    cell_low = (primary_axis_value, secondary_axis_low)
    cell_high = (primary_axis_value, secondary_axis_high)

    if cell_low not in intersectional_df.index or cell_high not in intersectional_df.index:
        return {
            "within_gap_pp": np.nan,
            "reference_gap_pp": np.nan,
            "verdict": "inconclusive",
            "reference_primary_value": None,
            "rationale": f"Required cells {cell_low} or {cell_high} missing from intersectional table.",
        }

    val_low = intersectional_df.loc[cell_low, metric_name]
    val_high = intersectional_df.loc[cell_high, metric_name]
    if pd.isna(val_low) or pd.isna(val_high):
        return {
            "within_gap_pp": np.nan,
            "reference_gap_pp": np.nan,
            "verdict": "inconclusive",
            "reference_primary_value": None,
            "rationale": f"Metric {metric_name} contains NaN in required cells.",
        }

    within_gap_pp = float((val_low - val_high) * 100.0)

    # Find reference primary value (different from primary_axis_value with largest sample size)
    primary_values = intersectional_df.index.get_level_values(0).unique()
    candidate_refs = [pv for pv in primary_values if pv != primary_axis_value]

    best_ref = None
    best_n = -1
    for pv in candidate_refs:
        ref_c_low = (pv, secondary_axis_low)
        ref_c_high = (pv, secondary_axis_high)
        if ref_c_low in intersectional_df.index and ref_c_high in intersectional_df.index:
            n_tot = 0
            if "n_total" in intersectional_df.columns:
                n_tot = intersectional_df.loc[ref_c_low, "n_total"] + intersectional_df.loc[ref_c_high, "n_total"]
            if n_tot > best_n:
                best_n = n_tot
                best_ref = pv

    if best_ref is None:
        return {
            "within_gap_pp": within_gap_pp,
            "reference_gap_pp": np.nan,
            "verdict": "inconclusive",
            "reference_primary_value": None,
            "rationale": "No candidate reference primary group has both required secondary axis cells.",
        }

    ref_val_low = intersectional_df.loc[(best_ref, secondary_axis_low), metric_name]
    ref_val_high = intersectional_df.loc[(best_ref, secondary_axis_high), metric_name]
    if pd.isna(ref_val_low) or pd.isna(ref_val_high):
        return {
            "within_gap_pp": within_gap_pp,
            "reference_gap_pp": np.nan,
            "verdict": "inconclusive",
            "reference_primary_value": best_ref,
            "rationale": f"Reference group {best_ref} has NaN metric values.",
        }

    ref_gap_pp = float((ref_val_low - ref_val_high) * 100.0)

    if within_gap_pp > ref_gap_pp and within_gap_pp > 0:
        verdict = "proxy_supported"
        rationale = (
            f"Within {primary_axis_value}, the {secondary_axis_low} vs {secondary_axis_high} gap is "
            f"{within_gap_pp:+.1f} pp, exceeding the reference ({best_ref}) gap of {ref_gap_pp:+.1f} pp."
        )
    elif abs(within_gap_pp - ref_gap_pp) <= 5.0 or within_gap_pp <= 0:
        verdict = "inconsistent"
        rationale = (
            f"Within gap ({within_gap_pp:+.1f} pp) does not meaningfully exceed reference gap ({ref_gap_pp:+.1f} pp)."
        )
    else:
        verdict = "inconclusive"
        rationale = f"Within gap ({within_gap_pp:+.1f} pp) and reference gap ({ref_gap_pp:+.1f} pp) non-directional."

    return {
        "within_gap_pp": within_gap_pp,
        "reference_gap_pp": ref_gap_pp,
        "verdict": verdict,
        "reference_primary_value": best_ref,
        "rationale": rationale,
    }
