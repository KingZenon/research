"""Conclusion 1 check: do class weights / prior adjustment change the RANKING of P(fatal) or only the cut-off?

Usage:  python -X utf8 analysis_ranking_equivalence.py <combined_csv> <results_recheck_dir>

From the saved outer-year probabilities (exp1) it reports, per family x encoding x fold:
  * fatal PR-AUC of the plain and class-weighted model and their difference
  * Spearman rank correlation of P(fatal) between the two
  * matched operating point: the plain model's top-k P(fatal) rows are called fatal with k = the number of rows the class-weighted
    (resp. prior-adjusted) model calls fatal; precision / recall / F1 of that rule against the weighted model's own
  * precision of each model at fixed recall levels 0.4, 0.5, 0.6 (read from its precision-recall curve)
Writes ranking_equivalence_by_fold.csv and ranking_equivalence_summary.csv (mean, SD over folds) and fig_pr_plain_vs_weighted.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve

import recheck_common as rc

S1 = "exp1_native_vs_ordinal"


def prf(truth, called):
    tp = int((called & (truth == 1)).sum())
    p = tp / max(int(called.sum()), 1)
    r = tp / max(int(truth.sum()), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def precision_at_recall(truth, score, level):
    prec, rec, _ = precision_recall_curve(truth, score)
    ok = rec >= level
    return float(prec[ok].max()) if ok.any() else np.nan


def main(csv_path, out_dir):
    out = Path(out_dir)
    df, X, y, year = rc.load_pre2026(csv_path)
    rows = []
    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        truth = (y[va] == 0).astype(int)
        prior = np.bincount(y[tr], minlength=3) / tr.sum()
        for family in ("lightgbm", "xgboost"):
            for enc in ("ordinal", "native"):
                _, pp = rc.load_proba(out, f"{S1}_{family}_{enc}_plain", val_year)
                _, pc = rc.load_proba(out, f"{S1}_{family}_{enc}_class_weight", val_year)
                rho = float(spearmanr(pp[:, 0], pc[:, 0]).statistic)
                row = {"family": family, "encoding": enc, "val_year": val_year,
                       "pr_auc_plain": average_precision_score(truth, pp[:, 0]), "pr_auc_class_weight": average_precision_score(truth, pc[:, 0]),
                       "spearman_plain_vs_class_weight": rho}
                row["pr_auc_diff_cw_minus_plain"] = row["pr_auc_class_weight"] - row["pr_auc_plain"]
                for rec_level in (0.4, 0.5, 0.6):
                    row[f"prec_at_recall_{rec_level}_plain"] = precision_at_recall(truth, pp[:, 0], rec_level)
                    row[f"prec_at_recall_{rec_level}_class_weight"] = precision_at_recall(truth, pc[:, 0], rec_level)
                for name, pred in (("class_weight", pc.argmax(1)), ("prior_adjusted", (pp / prior).argmax(1))):
                    called_w = pred == 0
                    k = int(called_w.sum())
                    p_w, r_w, f_w = prf(truth, called_w)
                    top = np.zeros(len(truth), bool)
                    top[np.argsort(-pp[:, 0])[:k]] = True              # plain ranking, same number of fatal calls
                    p_m, r_m, f_m = prf(truth, top)
                    row.update({f"{name}_calls": k, f"{name}_F1_own": f_w, f"{name}_F1_plain_ranking_same_k": f_m,
                                f"{name}_precision_own": p_w, f"{name}_precision_plain_ranking_same_k": p_m,
                                f"{name}_recall_own": r_w, f"{name}_recall_plain_ranking_same_k": r_m})
                rows.append(row)
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(out / "ranking_equivalence_by_fold.csv", index=False)
    num = [c for c in by_fold.columns if c not in ("family", "encoding", "val_year")]
    g = by_fold.groupby(["family", "encoding"])[num]
    summ = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index()
    summ.to_csv(out / "ranking_equivalence_summary.csv", index=False)

    # figure: 2025 PR curves, plain vs class-weighted, for both encodings of LightGBM
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    va = year == 2025
    truth = (y[va] == 0).astype(int)
    for ax, enc in zip(axes, ("ordinal", "native")):
        for nm in ("plain", "class_weight"):
            _, p = rc.load_proba(out, f"{S1}_lightgbm_{enc}_{nm}", 2025)
            a, b, _ = precision_recall_curve(truth, p[:, 0])
            ax.plot(b, a, label=f"{nm} (AP {average_precision_score(truth, p[:, 0]):.3f})", lw=1.4)
        ax.set_title(f"LightGBM {enc}, 2025 validation")
        ax.set_xlabel("fatal recall")
        ax.set_ylabel("fatal precision")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_pr_plain_vs_weighted.png", dpi=140)

    pd.set_option("display.width", 250)
    cols = ["family", "encoding", "pr_auc_plain_mean", "pr_auc_class_weight_mean", "pr_auc_diff_cw_minus_plain_mean", "pr_auc_diff_cw_minus_plain_sd",
            "spearman_plain_vs_class_weight_mean", "class_weight_F1_own_mean", "class_weight_F1_plain_ranking_same_k_mean",
            "prior_adjusted_F1_own_mean", "prior_adjusted_F1_plain_ranking_same_k_mean",
            "prec_at_recall_0.5_plain_mean", "prec_at_recall_0.5_class_weight_mean"]
    print(summ[cols].round(4).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
