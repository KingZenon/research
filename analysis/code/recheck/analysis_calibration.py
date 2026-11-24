"""Experiment 3: calibration of P(fatal) for plain, class-weighted and prior-adjusted models (from saved probabilities).

Usage:  python -X utf8 analysis_calibration.py <combined_csv> <results_recheck_dir>

Needs the outer-year probabilities of exp1_native_vs_ordinal.py (and the inner-year ones of exp4_threshold.py for isotonic
recalibration). For every family x encoding x strategy and every fold it computes
    Brier score of P(fatal), Brier of the constant 'training prevalence' forecast, Brier skill score,
    ECE with 10 equal-width bins and with 10 equal-count bins, calibration-in-the-large (mean prediction minus observed rate),
    calibration slope (logistic regression of the outcome on logit P(fatal); 1 = perfect)
Strategies
    plain, class_weight (raw output), prior_adjusted (plain probabilities / training shares, renormalised),
    class_weight_backcorrected (class-weighted probabilities x training shares, renormalised: undoes the weighting),
    plain_isotonic (isotonic regression fitted on the inner year, applied to the outer year)
Writes calibration_by_fold.csv, calibration_summary.csv (mean, SD over folds), reliability_table.csv (pooled folds, 10 equal-count bins)
and fig_reliability.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import recheck_common as rc

S1 = "exp1_native_vs_ordinal"


def slope(truth, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, truth)
    return float(lr.coef_[0, 0])


def main(csv_path, out_dir):
    out = Path(out_dir)
    df, X, y, year = rc.load_pre2026(csv_path)
    rows, rel = [], []
    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        truth = (y[va] == 0).astype(int)
        prev_tr = float(np.mean(y[tr] == 0))
        for family in ("lightgbm", "xgboost"):
            for enc in ("ordinal", "native"):
                _, p_plain = rc.load_proba(out, f"{S1}_{family}_{enc}_plain", val_year)
                _, p_cw = rc.load_proba(out, f"{S1}_{family}_{enc}_class_weight", val_year)
                prior = np.bincount(y[tr], minlength=3) / tr.sum()
                adj = p_plain / prior
                adj /= adj.sum(axis=1, keepdims=True)
                back = p_cw * prior
                back /= back.sum(axis=1, keepdims=True)
                variants = {"plain": p_plain[:, 0], "class_weight": p_cw[:, 0], "prior_adjusted": adj[:, 0],
                            "class_weight_backcorrected": back[:, 0]}
                inner_path = out / "preds" / f"exp4_inner_{family}_{enc}_plain_{val_year}.npz"
                if inner_path.exists():
                    _, p_in = rc.load_proba(out, f"exp4_inner_{family}_{enc}_plain", val_year)
                    inner_truth = (y[year == last_train] == 0).astype(int)
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_in[:, 0], inner_truth)
                    variants["plain_isotonic_inner"] = iso.predict(p_plain[:, 0])
                for strat, p in variants.items():
                    p = np.clip(p, 0, 1)
                    brier = float(np.mean((p - truth) ** 2))
                    brier_ref = float(np.mean((prev_tr - truth) ** 2))
                    rows.append({"family": family, "encoding": enc, "strategy": strat, "val_year": val_year, "brier": brier,
                                 "brier_constant_train_prevalence": brier_ref, "brier_skill_score": 1 - brier / brier_ref,
                                 "ece_10_equal_width": rc.ece_binary(truth, p), "ece_10_equal_count": rc.ece_quantile(truth, p),
                                 "mean_pred_minus_observed": float(p.mean() - truth.mean()), "calibration_slope": slope(truth, p),
                                 "observed_rate": float(truth.mean()), "mean_pred": float(p.mean()),
                                 "pr_auc": float(average_precision_score(truth, p))})
                    for b, chunk in enumerate(np.array_split(np.argsort(p), 10)):
                        rel.append({"family": family, "encoding": enc, "strategy": strat, "val_year": val_year, "bin": b,
                                    "n": len(chunk), "mean_pred": float(p[chunk].mean()), "observed_rate": float(truth[chunk].mean())})
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(out / "calibration_by_fold.csv", index=False)
    num = [c for c in by_fold.columns if c not in ("family", "encoding", "strategy", "val_year")]
    g = by_fold.groupby(["family", "encoding", "strategy"])[num]
    summ = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index()
    summ.to_csv(out / "calibration_summary.csv", index=False)
    reld = pd.DataFrame(rel)
    pooled = reld.groupby(["family", "encoding", "strategy", "bin"]).apply(
        lambda d: pd.Series({"n": d["n"].sum(), "mean_pred": np.average(d["mean_pred"], weights=d["n"]),
                             "observed_rate": np.average(d["observed_rate"], weights=d["n"])}), include_groups=False).reset_index()
    pooled.to_csv(out / "reliability_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, enc in zip(axes, ("native", "ordinal")):
        for strat, style in (("plain", "o-"), ("class_weight", "s-"), ("prior_adjusted", "^-"), ("plain_isotonic_inner", "d--")):
            d = pooled[(pooled.family == "lightgbm") & (pooled.encoding == enc) & (pooled.strategy == strat)]
            if len(d):
                ax.plot(d["mean_pred"], d["observed_rate"], style, label=strat, ms=4)
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_title(f"LightGBM, {enc} categoricals (4 folds pooled, deciles)")
        ax.set_xlabel("mean predicted P(fatal)")
        ax.set_ylabel("observed fatal rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_reliability.png", dpi=140)

    pd.set_option("display.width", 250)
    cols = ["family", "encoding", "strategy", "brier_mean", "brier_sd", "brier_skill_score_mean", "ece_10_equal_width_mean", "ece_10_equal_width_sd",
            "ece_10_equal_count_mean", "mean_pred_minus_observed_mean", "calibration_slope_mean", "pr_auc_mean"]
    print(summ[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
