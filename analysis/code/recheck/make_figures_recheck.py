"""Summary figures for the re-check.

Usage:  python -X utf8 make_figures_recheck.py <original_results_dir> <results_recheck_dir>
Writes fig_folds_f1_prauc.png: fatal F1 and fatal PR-AUC per validation year for the main variants (original ordinal LightGBM,
native LightGBM with class weights / inner-year F1 threshold, native XGBoost, SMOTE-NC scaled), showing the 2023 fold effect.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(orig_dir, out_dir):
    out = Path(out_dir)
    log = pd.read_csv(out / "experiment_log.csv")
    log = log.drop_duplicates(subset=["stage", "family", "encoding", "strategy", "val_year"])
    series = {
        "LightGBM ordinal, class weights (original)": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "lightgbm") & (log.encoding == "ordinal") & (log.strategy == "class_weight")],
        "LightGBM native, class weights": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "lightgbm") & (log.encoding == "native") & (log.strategy == "class_weight")],
        "LightGBM native, plain + inner-year F1 threshold": log[(log.stage == "exp4_threshold") & (log.family == "lightgbm") & (log.encoding == "native") & (log.strategy == "thr_f1_inner")],
        "XGBoost ordinal, class weights (original)": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "xgboost") & (log.encoding == "ordinal") & (log.strategy == "class_weight")],
        "XGBoost native, class weights": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "xgboost") & (log.encoding == "native") & (log.strategy == "class_weight")],
    }
    tuned = log[(log.stage == "exp9_tuned_threshold")]
    if len(tuned) == 4:
        series["LightGBM native tuned + inner-year F1 threshold (post hoc)"] = tuned
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, d in series.items():
        d = d.sort_values("val_year")
        axes[0].plot(d["val_year"], d["fatal_f1"], "o-", label=label)
    for label, d in {"LightGBM ordinal, plain (original)": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "lightgbm") & (log.encoding == "ordinal") & (log.strategy == "plain")],
                     "LightGBM native, plain": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "lightgbm") & (log.encoding == "native") & (log.strategy == "plain")],
                     "XGBoost ordinal, plain (original)": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "xgboost") & (log.encoding == "ordinal") & (log.strategy == "plain")],
                     "XGBoost native, plain": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "xgboost") & (log.encoding == "native") & (log.strategy == "plain")],
                     "LightGBM native, SMOTE-NC (scaled)": log[(log.stage == "exp2_smote") & (log.family == "lightgbm") & (log.encoding == "smote_scaled_native")]}.items():
        d = d.sort_values("val_year")
        axes[1].plot(d["val_year"], d["fatal_pr_auc"], "o-", label=label)
    axes[0].set_title("fatal F1 by validation year")
    axes[1].set_title("fatal PR-AUC by validation year")
    for ax in axes:
        ax.set_xticks([2022, 2023, 2024, 2025])
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig_folds_f1_prauc.png", dpi=140)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
