"""Summaries of the final experiments and the pre-set model selection.

Usage:  python -X utf8 aggregate_final.py <results_dir>

Reads the fold files
    cv_temporal_thai_folds.csv        Milestone 3: 6 families x default / class_weight / prior_adjusted (untuned)
    cv_thai_smote_untuned_folds.csv   the same families with SMOTE-NC (untuned)
    cv_thai_tuned_folds.csv           XGBoost and LightGBM tuned inside each fold, 4 variants
    cv_motorcycle_folds.csv           motorcycle-only subset
    cv_uk_folds.csv                   UK STATS19 replication
and writes cv_thai_all_summary.csv, cv_motorcycle_summary.csv, cv_uk_summary.csv, selection_report.csv and
final_selection.json.

Selection rule (fixed before the 2026 test year is scored): among all Thai candidates, the (model, variant) with the
highest mean fatal-class F1 over the four validation years; ties are broken by macro-F1. Fatal F1 is used because
the research question asks for a model and imbalance strategy that find fatal crashes (recall) while keeping false
alarms useful (precision). The report also lists the paired fold-by-fold differences to the runners-up, because
gaps inside one standard deviation should be called comparable, not a win.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["macro_f1", "balanced_acc", "accuracy", "mcc", "fatal_recall", "fatal_precision", "fatal_f1",
           "fatal_pr_auc", "fatal_auc", "fatal_vs_rest_accuracy"]


def read(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def summarise(folds):
    present = [m for m in METRICS if m in folds.columns]
    g = folds.groupby(["model", "variant"])[present]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).reset_index()
    out["n_folds"] = folds.groupby(["model", "variant"]).size().to_numpy()
    return out


def main(results_dir):
    results = Path(results_dir)
    parts = [read(results / name) for name in ("cv_temporal_thai_folds.csv", "cv_thai_smote_untuned_folds.csv", "cv_thai_tuned_folds.csv")]
    thai = pd.concat([p for p in parts if p is not None], ignore_index=True)
    thai["val_year"] = thai["val_year"].astype(int)
    thai.to_csv(results / "cv_thai_all_folds.csv", index=False)
    summary = summarise(thai)
    summary.to_csv(results / "cv_thai_all_summary.csv", index=False)

    for name, label in (("cv_motorcycle_folds.csv", "motorcycle"), ("cv_uk_folds.csv", "uk")):
        folds = read(results / name)
        if label == "motorcycle":                      # the tuned motorcycle models (ROC-AUC criterion) join the same summary
            tuned = read(results / "cv_motorcycle_tuned_folds.csv")
            if tuned is not None:
                folds = pd.concat([folds, tuned], ignore_index=True)
        if folds is not None:
            summarise(folds).to_csv(results / f"cv_{label}_summary.csv", index=False)

    # pre-set selection among the Thai candidates
    cand = summary[summary["model"] != "majority_class"].sort_values(["fatal_f1_mean", "macro_f1_mean"], ascending=False).reset_index(drop=True)
    best = cand.iloc[0]
    key = lambda m, v: thai[(thai.model == m) & (thai.variant == v)].sort_values("val_year")
    best_folds = key(best["model"], best["variant"])["fatal_f1"].to_numpy()
    rows = []
    for i, r in cand.head(10).iterrows():
        f = key(r["model"], r["variant"])
        diff = best_folds - f["fatal_f1"].to_numpy() if len(f) == len(best_folds) else np.full(len(best_folds), np.nan)
        rows.append({"rank": i + 1, "model": r["model"], "variant": r["variant"], "fatal_f1_mean": round(r["fatal_f1_mean"], 4),
                     "fatal_f1_std": round(r["fatal_f1_std"], 4), "macro_f1_mean": round(r["macro_f1_mean"], 4),
                     "fatal_recall_mean": round(r["fatal_recall_mean"], 4), "fatal_precision_mean": round(r["fatal_precision_mean"], 4),
                     "fatal_pr_auc_mean": round(r["fatal_pr_auc_mean"], 4),
                     "gap_to_winner_mean": round(float(np.nanmean(diff)), 4),
                     "winner_ahead_in_folds": int(np.nansum(diff > 0)) if i else None})
    report = pd.DataFrame(rows)
    report.to_csv(results / "selection_report.csv", index=False)

    tuned = str(best["model"]).endswith("_tuned")
    selection = {"rule": "highest mean fatal-class F1 over the four validation years (2022-2025), ties by macro-F1",
                 "model": str(best["model"]).replace("_tuned", ""), "tuned": tuned, "variant": str(best["variant"]),
                 "fatal_f1_mean": round(float(best["fatal_f1_mean"]), 4), "macro_f1_mean": round(float(best["macro_f1_mean"]), 4),
                 "candidates_compared": int(len(cand))}
    (results / "final_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    print("\nTop candidates by mean fatal F1 (Thai, 4 temporal folds)")
    print(report.to_string(index=False))
    print("\nselection:", selection)

    moto = read(results / "cv_motorcycle_summary.csv")
    if moto is not None:
        show = moto[moto.model != "majority_class"].sort_values("fatal_auc_mean", ascending=False)
        print("\nMotorcycle subset (fatal vs rest): accuracy / ROC-AUC / fatal recall vs Mohamad et al. RF 0.777 / 0.768 / 0.198")
        print(show[["model", "variant", "fatal_vs_rest_accuracy_mean", "fatal_auc_mean", "fatal_recall_mean", "fatal_precision_mean", "fatal_f1_mean"]].round(3).head(12).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
