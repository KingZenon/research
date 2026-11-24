"""Paired fold-by-fold look at the ORIGINAL results (read-only): conclusions 3 and 4.

Usage:  python -X utf8 analysis_original_paired.py <original_results_dir> <results_recheck_dir>

Writes original_tuning_effect_by_fold.csv (tuned minus untuned, per fold, for the same strategy) and
original_top_candidates_paired.csv (fold-by-fold fatal F1 of the four headline candidates and their paired differences).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(orig_dir, out_dir):
    f = pd.read_csv(Path(orig_dir) / "cv_thai_all_folds.csv")
    rows = []
    for fam in ("lightgbm", "xgboost"):
        for var in ("default", "class_weight", "prior_adjusted"):
            a = f[(f.model == f"{fam}_tuned") & (f.variant == var)].set_index("val_year").sort_index()
            b = f[(f.model == fam) & (f.variant == var)].set_index("val_year").sort_index()
            for m in ("fatal_pr_auc", "fatal_f1", "fatal_auc"):
                d = (a[m] - b[m]).to_numpy()
                rows.append({"family": fam, "strategy": var, "metric": m, "mean_tuned_minus_untuned": d.mean(), "sd": d.std(ddof=1),
                             "folds_tuned_better": int((d > 0).sum()), "per_fold_2022_2025": ";".join(f"{v:+.4f}" for v in d)})
    pd.DataFrame(rows).to_csv(Path(out_dir) / "original_tuning_effect_by_fold.csv", index=False)

    cands = [("lightgbm_tuned", "class_weight"), ("xgboost_tuned", "class_weight"), ("xgboost", "class_weight"),
             ("lightgbm_tuned", "prior_adjusted"), ("xgboost_tuned", "prior_adjusted"), ("lightgbm", "class_weight")]
    tab = {}
    for m, v in cands:
        tab[f"{m}|{v}"] = f[(f.model == m) & (f.variant == v)].set_index("val_year").sort_index()["fatal_f1"]
    tab = pd.DataFrame(tab)
    best = tab.iloc[:, 0]
    out = []
    for c in tab.columns:
        d = (best - tab[c]).to_numpy()
        out.append({"candidate": c, "fatal_f1_mean": tab[c].mean(), "fatal_f1_sd": tab[c].std(ddof=1),
                    "per_fold_2022_2025": ";".join(f"{v:.4f}" for v in tab[c]), "winner_minus_candidate_mean": d.mean(),
                    "winner_minus_candidate_sd": d.std(ddof=1) if c != tab.columns[0] else np.nan, "winner_ahead_in_folds": int((d > 0).sum())})
    pd.DataFrame(out).to_csv(Path(out_dir) / "original_top_candidates_paired.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
