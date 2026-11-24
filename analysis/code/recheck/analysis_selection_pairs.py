"""Paired fold-by-fold comparison of the recheck's best candidates with the ORIGINALLY selected model (validation years only).

Usage:  python -X utf8 analysis_selection_pairs.py <original_results_dir> <results_recheck_dir>

Reference = original selected model: tuned LightGBM + class weights (ordinal integers) from results/cv_thai_all_folds.csv.
Writes selection_pairs.csv: per-fold fatal F1 / PR-AUC / macro-F1 of each candidate, mean +- SD, the paired difference to the
reference (mean, SD of the differences, folds ahead).  No test-year data are touched.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(orig_dir, out_dir):
    out = Path(out_dir)
    o = pd.read_csv(Path(orig_dir) / "cv_thai_all_folds.csv")
    log = pd.read_csv(out / "experiment_log.csv").drop_duplicates(subset=["stage", "family", "encoding", "strategy", "val_year"])
    ref = o[(o.model == "lightgbm_tuned") & (o.variant == "class_weight")].set_index("val_year").sort_index()
    pick = {
        "ORIGINAL SELECTED: LightGBM tuned (ordinal) + class weights": ref.reset_index().rename(columns={"fatal_f1": "fatal_f1", "fatal_pr_auc": "fatal_pr_auc"}),
        "LightGBM native + inner-year F1 threshold": log[(log.stage == "exp4_threshold") & (log.family == "lightgbm") & (log.encoding == "native") & (log.strategy == "thr_f1_inner")],
        "LightGBM native + class weights": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "lightgbm") & (log.encoding == "native") & (log.strategy == "class_weight")],
        "LightGBM native tuned + class weights": log[(log.stage == "exp7_tuning_native") & (log.strategy == "class_weight")],
        "LightGBM native tuned + inner-year F1 threshold (post hoc)": log[(log.stage == "exp9_tuned_threshold")],
        "XGBoost ordinal + class weights (original)": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "xgboost") & (log.encoding == "ordinal") & (log.strategy == "class_weight")],
        "LightGBM ordinal + inner-year F1 threshold": log[(log.stage == "exp4_threshold") & (log.family == "lightgbm") & (log.encoding == "ordinal") & (log.strategy == "thr_f1_inner")],
        "XGBoost native + class weights": log[(log.stage == "exp1_native_vs_ordinal") & (log.family == "xgboost") & (log.encoding == "native") & (log.strategy == "class_weight")],
    }
    rows = []
    base = pick["ORIGINAL SELECTED: LightGBM tuned (ordinal) + class weights"].set_index("val_year").sort_index()
    for name, d in pick.items():
        d = d.set_index("val_year").sort_index()
        diff = (d["fatal_f1"] - base["fatal_f1"]).to_numpy()
        rows.append({"candidate": name, "fatal_f1_per_fold_2022_2025": ";".join(f"{v:.4f}" for v in d["fatal_f1"]),
                     "fatal_f1_mean": d["fatal_f1"].mean(), "fatal_f1_sd": d["fatal_f1"].std(ddof=1),
                     "fatal_pr_auc_mean": d["fatal_pr_auc"].mean(), "fatal_pr_auc_sd": d["fatal_pr_auc"].std(ddof=1),
                     "macro_f1_mean": d["macro_f1"].mean(),
                     "diff_to_original_selected_mean": diff.mean(), "diff_sd": diff.std(ddof=1) if name.startswith("ORIGINAL") is False else np.nan,
                     "folds_ahead_of_original": int((diff > 0).sum()) if not name.startswith("ORIGINAL") else np.nan,
                     "diff_per_fold": ";".join(f"{v:+.4f}" for v in diff)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "selection_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
