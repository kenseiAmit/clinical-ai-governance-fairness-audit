"""
shap_analysis.py — SHAP setup and proxy-feature detection.

The conceptual jump in this lab is the proxy-feature detector. Anyone can
call a SHAP plotting helper. Encoding what a proxy feature *looks like* in
code is the hard part — and is exactly the skill an AI Risk Officer is
hired for.

Run the unit tests to check your work:
    cd clinical-ai-governance-fairness-audit/
    pytest lab/tests/test_shap_analysis.py -v
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np
import pandas as pd
import shap


def setup_explainer(pipeline, X_background: pd.DataFrame):
    """
    Initialize a SHAP explainer for the trained scikit-learn pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_bg_t = preprocessor.transform(X_background)
    feat_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
    return explainer, X_bg_t, feat_names


def detect_proxy_features(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    protected_feature_substrings: Sequence[str],
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Identify candidate proxy features and return a ranked table.
    """
    if isinstance(shap_values, list):
        sv_matrix = np.asarray(shap_values[1])
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        sv_matrix = shap_values[:, :, 1]
    else:
        sv_matrix = np.asarray(shap_values)

    sub_lower = [s.lower() for s in protected_feature_substrings]
    records = []
    for j, name in enumerate(feature_names):
        if any(sub in name.lower() for sub in sub_lower):
            continue
        col = sv_matrix[:, j]
        mean_abs = float(np.mean(np.abs(col)))
        mean_signed = float(np.mean(col))
        ratio = float(abs(mean_signed) / (mean_abs + 1e-12))
        signed_strength = float(min(1.0, max(0.0, ratio)))
        records.append({
            "feature": name,
            "mean_abs_shap": mean_abs,
            "mean_shap": mean_signed,
            "signed_strength": signed_strength,
        })

    if not records:
        return pd.DataFrame(columns=["rank", "feature", "mean_abs_shap", "mean_shap", "signed_strength"])

    df = pd.DataFrame(records)
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df[["rank", "feature", "mean_abs_shap", "mean_shap", "signed_strength"]]


def explain_false_negatives(
    explainer,
    X_transformed: np.ndarray,
    feature_names: Sequence[str],
    y_true,
    y_pred,
    mask,
    n_examples: int = 3,
) -> List[dict]:
    """
    For up to n_examples false negatives matching the mask, return
    their top contributing features (positive AND negative) with SHAP values.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    m = np.asarray(mask, dtype=bool)

    fn_indices = np.where((yt == 1) & (yp == 0) & m)[0]
    selected_indices = fn_indices[:n_examples]

    results = []
    for idx in selected_indices:
        x_inst = X_transformed[idx : idx + 1]
        sv = explainer.shap_values(x_inst)
        if isinstance(sv, list):
            sv_row = np.asarray(sv[1]).ravel()
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            sv_row = sv[0, :, 1].ravel()
        else:
            sv_row = np.asarray(sv).ravel()

        pairs = [(feature_names[j], float(sv_row[j])) for j in range(len(feature_names))]
        pos_features = sorted([(f, v) for f, v in pairs if v > 0], key=lambda x: x[1], reverse=True)
        neg_features = sorted([(f, v) for f, v in pairs if v < 0], key=lambda x: x[1])

        sum_pos = float(sum(v for _, v in pos_features))
        sum_neg = float(sum(v for _, v in neg_features))

        results.append({
            "instance_index": int(idx),
            "shap_values": sv_row,
            "top_positive_features": pos_features,
            "top_negative_features": neg_features,
            "sum_positive": sum_pos,
            "sum_negative": sum_neg,
        })

    return results
