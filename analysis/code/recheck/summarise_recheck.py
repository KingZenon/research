"""Summaries of the re-check experiment log (read-only on the original results).

Usage:  python -X utf8 summarise_recheck.py <original_results_dir> <results_recheck_dir>

Reads  <recheck>/experiment_log.csv  (append-only, one row per stage / family / encoding / strategy / fold) and the original
<original>/cv_thai_all_folds.csv, cv_thai_smote_control_folds.csv, cv_thai_tuned_folds.csv.
Writes in <recheck>:
  summary_all.csv            mean and SD (ddof = 1, 4 folds) of every metric for every recheck row group
  summary_vs_original.csv    reproduction check of the ordinal runs against the original CSV + original rows side by side
  paired_native_vs_ordinal.csv   fold-by-fold paired differences (native minus ordinal)
  paired_strategy_differences.csv   paired differences between strategies inside one model
  selection_recheck.csv / final_selection_recheck.json   the pre-set rule (highest mean fatal F1, ties by macro-F1) applied to the
                             recheck candidates AND the original ones; nothing is written about 2026.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["fatal_recall", "fatal_precision", "fatal_f1", "fatal_pr_auc", "fatal_auc", "macro_f1", "balanced_acc", "mcc",
           "accuracy", "brier_fatal", "ece_fatal"]


def group_summary(folds, keys):
    present = [m for m in METRICS if m in folds.columns and folds[m].notna().any()]
    g = folds.groupby(keys)[present]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index()
    out["n_folds"] = folds.groupby(keys).size().to_numpy()
    return out


def original_rows(orig_dir):
    f = pd.read_csv(Path(orig_dir) / "cv_thai_all_folds.csv")
    f = f[f["model"].isin(["lightgbm", "xgboost", "lightgbm_tuned", "xgboost_tuned", "random_forest", "logistic_regression", "majority_class"])].copy()
    f = f.rename(columns={"model": "family", "variant": "strategy"})
    f["strategy"] = f["strategy"].replace({"default": "plain"})
    f["encoding"] = "ordinal_ORIGINAL_FILE"
    f["stage"] = "original_results"
    return f


def main(orig_dir, out_dir):
    out = Path(out_dir)
    log = pd.read_csv(out / "experiment_log.csv")
    # a restarted run could in principle leave duplicate keys: keep the first row of each key and say so
    key = ["stage", "family", "encoding", "strategy", "val_year"]
    dup = log.duplicated(subset=key, keep="first")
    if dup.any():
        print(f"WARNING: {int(dup.sum())} duplicate log rows ignored (kept first)")
    log = log[~dup]
    summ = group_summary(log, ["stage", "family", "encoding", "strategy"])
    summ.to_csv(out / "summary_all.csv", index=False)

    orig = original_rows(orig_dir)
    # reproduction check: ordinal rows of exp1 against the original file (same code path, same seed)
    ord_rows = log[(log.stage == "exp1_native_vs_ordinal") & (log.encoding == "ordinal")]
    chk = ord_rows.merge(orig[["family", "strategy", "val_year", "fatal_f1", "fatal_pr_auc", "fatal_recall", "fatal_precision"]],
                         on=["family", "strategy", "val_year"], suffixes=("_recheck", "_original"))
    chk["abs_diff_f1"] = (chk["fatal_f1_recheck"] - chk["fatal_f1_original"]).abs()
    chk["abs_diff_prauc"] = (chk["fatal_pr_auc_recheck"] - chk["fatal_pr_auc_original"]).abs()
    chk[["family", "strategy", "val_year", "fatal_f1_recheck", "fatal_f1_original", "fatal_pr_auc_recheck", "fatal_pr_auc_original",
         "abs_diff_f1", "abs_diff_prauc"]].to_csv(out / "reproduction_check_ordinal_vs_original.csv", index=False)
    print(f"reproduction check (ordinal exp1 vs original CSV): {len(chk)} fold rows, max |dF1| {chk['abs_diff_f1'].max():.6f}, "
          f"max |dPR-AUC| {chk['abs_diff_prauc'].max():.6f}")

    osum = group_summary(orig, ["stage", "family", "encoding", "strategy"])
    pd.concat([osum, summ], ignore_index=True).to_csv(out / "summary_vs_original.csv", index=False)

    # paired native - ordinal differences
    e1 = log[log.stage == "exp1_native_vs_ordinal"]
    wide = e1.pivot_table(index=["family", "strategy", "val_year"], columns="encoding", values=METRICS)
    rows = []
    for (family, strategy), sub in wide.groupby(level=[0, 1]):
        for m in ("fatal_f1", "fatal_pr_auc", "fatal_auc", "fatal_recall", "fatal_precision", "macro_f1", "mcc", "brier_fatal"):
            d = (sub[(m, "native")] - sub[(m, "ordinal")]).to_numpy()
            rows.append({"family": family, "strategy": strategy, "metric": m, "mean_diff_native_minus_ordinal": d.mean(),
                         "sd_diff": d.std(ddof=1), "folds_native_better": int((d > 0).sum()) if m != "brier_fatal" else int((d < 0).sum()),
                         "per_fold_2022_2023_2024_2025": ";".join(f"{v:+.4f}" for v in d)})
    pd.DataFrame(rows).to_csv(out / "paired_native_vs_ordinal.csv", index=False)

    # paired strategy differences against plain, per model (all recheck stages)
    rows = []
    for (stage, family, enc), sub in log.groupby(["stage", "family", "encoding"]):
        base = sub[sub.strategy == "plain"].set_index("val_year")
        for strategy, s2 in sub.groupby("strategy"):
            if strategy == "plain" or len(base) == 0:
                continue
            s2 = s2.set_index("val_year")
            common = base.index.intersection(s2.index)
            for m in ("fatal_f1", "fatal_pr_auc", "fatal_recall", "fatal_precision", "macro_f1"):
                d = (s2.loc[common, m] - base.loc[common, m]).to_numpy()
                rows.append({"stage": stage, "family": family, "encoding": enc, "strategy_minus_plain": strategy, "metric": m,
                             "mean_diff": d.mean(), "sd_diff": d.std(ddof=1) if len(d) > 1 else np.nan, "n_folds": len(d)})
    pd.DataFrame(rows).to_csv(out / "paired_strategy_differences.csv", index=False)

    # pre-set selection rule on the recheck candidates (argmax-type strategies and the inner-year thresholds; oracle excluded)
    cand = summ[(~summ["strategy"].isin(["thr_f1_oracle"])) & (summ["n_folds"] == 4)].copy()
    cand["source"] = "recheck"
    ocand = osum[(osum["family"] != "majority_class") & (osum["n_folds"] == 4)].copy()
    ocand["source"] = "original"
    allc = pd.concat([cand, ocand], ignore_index=True).sort_values(["fatal_f1_mean", "macro_f1_mean"], ascending=False).reset_index(drop=True)
    cols = ["source", "stage", "family", "encoding", "strategy", "fatal_f1_mean", "fatal_f1_sd", "macro_f1_mean", "fatal_recall_mean",
            "fatal_precision_mean", "fatal_pr_auc_mean", "fatal_pr_auc_sd"]
    allc[cols].to_csv(out / "selection_recheck.csv", index=False)
    recheck_only = allc[allc.source == "recheck"].reset_index(drop=True)
    best = recheck_only.iloc[0]
    runner = recheck_only.iloc[1:6][["family", "encoding", "strategy", "fatal_f1_mean"]].round(4).to_dict("records")
    sel = {"rule": "highest mean fatal-class F1 over the four validation years (2022-2025), ties by macro-F1 (same rule as the original study; "
                   "2026 not used, no new test scoring done)",
           "best_recheck_candidate": {k: (v if isinstance(v, str) else float(v)) for k, v in best[["stage", "family", "encoding", "strategy", "fatal_f1_mean", "fatal_f1_sd", "macro_f1_mean"]].items()},
           "next_five": runner,
           "best_original_candidate": {k: (v if isinstance(v, str) else float(v)) for k, v in allc[allc.source == "original"].iloc[0][["family", "strategy", "fatal_f1_mean", "fatal_f1_sd"]].items()}}
    (out / "final_selection_recheck.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    show = ["stage", "family", "encoding", "strategy", "fatal_recall_mean", "fatal_precision_mean", "fatal_f1_mean", "fatal_f1_sd",
            "fatal_pr_auc_mean", "fatal_pr_auc_sd", "fatal_auc_mean", "macro_f1_mean", "balanced_acc_mean", "mcc_mean", "n_folds"]
    print(summ[show].round(4).to_string(index=False))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
