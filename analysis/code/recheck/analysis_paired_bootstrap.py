"""Paired bootstrap of model differences on the validation folds (answers 'are the top candidates comparable?').

Usage:  python -X utf8 analysis_paired_bootstrap.py <combined_csv> <results_recheck_dir> [--reps 300]

For each comparison (A vs B) the SAME resampled incidents are used for both models inside each validation year; the metric
difference (A - B) is averaged over the four years, and the 95% percentile interval over --reps resamples is reported
(incidents are resampled independently, so clustering of incidents in space / time is ignored: intervals are optimistic).
Reads the probability files saved by exp1 / exp2 / exp4 / exp7 (those that exist), writes paired_bootstrap.csv.
Rule 'argmax' = predicted class with the largest probability, 'prior' = probabilities divided by the training class shares.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import recheck_common as rc

S1 = "exp1_native_vs_ordinal"


def f1_fatal(truth, called):
    tp = np.sum(called & (truth == 1))
    d = called.sum() + truth.sum()
    return 2 * tp / d if d else 0.0


def metric_value(kind, truth, proba, rule, prior, idx):
    """rule: 'argmax', 'prior' or a float threshold on P(fatal)."""
    t = truth[idx]
    if kind == "pr_auc":
        return average_precision_score(t, proba[idx, 0])
    if isinstance(rule, float):
        pred0 = proba[idx, 0] >= rule
    else:
        pred0 = (proba[idx] / prior).argmax(1) == 0 if rule == "prior" else proba[idx].argmax(1) == 0
    return f1_fatal(t, pred0)


def resolve_rule(rule, log, val_year):
    """'thr_f1_inner@lightgbm@native' -> the threshold chosen on the inner year for that fold (from the experiment log)."""
    if not rule.startswith("thr_"):
        return rule
    name, family, enc = rule.split("@")
    r = log[(log.stage == "exp4_threshold") & (log.family == family) & (log.encoding == enc) & (log.strategy == name) & (log.val_year == val_year)]
    return float(r["threshold"].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.results_dir)
    df, X, y, year = rc.load_pre2026(args.csv)
    log = pd.read_csv(out / "experiment_log.csv")

    P = lambda stage, fam, enc, strat: f"{stage}_{fam}_{enc}_{strat}"
    specs = []
    for fam in ("lightgbm", "xgboost"):
        specs.append((f"{fam}: native minus ordinal, plain, PR-AUC", (P(S1, fam, "native", "plain"), "argmax"), (P(S1, fam, "ordinal", "plain"), "argmax"), "pr_auc"))
        specs.append((f"{fam}: native minus ordinal, class weights, fatal F1", (P(S1, fam, "native", "class_weight"), "argmax"), (P(S1, fam, "ordinal", "class_weight"), "argmax"), "f1"))
        specs.append((f"{fam}: class weights minus plain, native, PR-AUC", (P(S1, fam, "native", "class_weight"), "argmax"), (P(S1, fam, "native", "plain"), "argmax"), "pr_auc"))
        specs.append((f"{fam}: prior-adjusted minus class weights, native, fatal F1", (P(S1, fam, "native", "plain"), "prior"), (P(S1, fam, "native", "class_weight"), "argmax"), "f1"))
    for enc in ("ordinal", "native"):
        specs.append((f"{enc}: xgboost minus lightgbm, class weights, fatal F1", (P(S1, "xgboost", enc, "class_weight"), "argmax"), (P(S1, "lightgbm", enc, "class_weight"), "argmax"), "f1"))
        specs.append((f"{enc}: xgboost minus lightgbm, plain, PR-AUC", (P(S1, "xgboost", enc, "plain"), "argmax"), (P(S1, "lightgbm", enc, "plain"), "argmax"), "pr_auc"))
    for fam in ("lightgbm", "xgboost"):
        for v in ("smote_scaled_native", "smote_scaled_ordinal"):
            specs.append((f"{fam}: {v} minus plain of same encoding, PR-AUC", (P("exp2_smote", fam, v, "").rstrip("_"), "argmax"),
                          (P(S1, fam, v.split("_")[-1], "plain"), "argmax"), "pr_auc"))
        specs.append((f"{fam}: smote_scaled_native minus control_native (imputation only), PR-AUC", (P("exp2_smote", fam, "smote_scaled_native", "").rstrip("_"), "argmax"),
                      (P("exp2_smote", fam, "control_native", "").rstrip("_"), "argmax"), "pr_auc"))
    thr = "thr_f1_inner@lightgbm@native"
    specs.append(("lightgbm native: inner-year F1 threshold minus class weights, fatal F1", (P(S1, "lightgbm", "native", "plain"), thr), (P(S1, "lightgbm", "native", "class_weight"), "argmax"), "f1"))
    specs.append(("lightgbm native: inner-year F1 threshold minus prior adjustment, fatal F1", (P(S1, "lightgbm", "native", "plain"), thr), (P(S1, "lightgbm", "native", "plain"), "prior"), "f1"))
    specs.append(("lightgbm native: inner-year F1 threshold minus plain argmax, fatal F1", (P(S1, "lightgbm", "native", "plain"), thr), (P(S1, "lightgbm", "native", "plain"), "argmax"), "f1"))
    specs.append(("lightgbm native threshold minus xgboost ordinal class weights, fatal F1", (P(S1, "lightgbm", "native", "plain"), thr), (P(S1, "xgboost", "ordinal", "class_weight"), "argmax"), "f1"))
    specs.append(("lightgbm tuned native minus untuned native, plain, PR-AUC", (P("exp7_tuning_native", "lightgbm", "native", "plain"), "argmax"), (P(S1, "lightgbm", "native", "plain"), "argmax"), "pr_auc"))
    specs.append(("lightgbm tuned native minus untuned native, class weights, fatal F1", (P("exp7_tuning_native", "lightgbm", "native", "class_weight"), "argmax"), (P(S1, "lightgbm", "native", "class_weight"), "argmax"), "f1"))

    rng = np.random.default_rng(rc.SEED)
    fold_idx = {}
    for last_train, val_year in rc.FOLDS:
        va = year == val_year
        n = int(va.sum())
        fold_idx[val_year] = [rng.integers(0, n, n) for _ in range(args.reps)]     # same resamples for every comparison
    rows = []
    for label, (n1, r1), (n2, r2), kind in specs:
        try:
            probs1 = {vy: rc.load_proba(out, n1, vy)[1] for _, vy in rc.FOLDS}
            probs2 = {vy: rc.load_proba(out, n2, vy)[1] for _, vy in rc.FOLDS}
        except FileNotFoundError:
            continue
        boot = np.zeros(args.reps)
        point = []
        for last_train, vy in rc.FOLDS:
            tr, va = year <= last_train, year == vy
            truth = (y[va] == 0).astype(int)
            prior = np.bincount(y[tr], minlength=3) / tr.sum()
            r1_, r2_ = resolve_rule(r1, log, vy), resolve_rule(r2, log, vy)
            point.append(metric_value(kind, truth, probs1[vy], r1_, prior, np.arange(len(truth))) - metric_value(kind, truth, probs2[vy], r2_, prior, np.arange(len(truth))))
            for b in range(args.reps):
                idx = fold_idx[vy][b]
                boot[b] += (metric_value(kind, truth, probs1[vy], r1_, prior, idx) - metric_value(kind, truth, probs2[vy], r2_, prior, idx)) / len(rc.FOLDS)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"comparison": label, "metric": kind, "mean_diff_over_folds": float(np.mean(point)), "ci95_low": float(lo), "ci95_high": float(hi),
                     "per_fold_diffs_2022_2025": ";".join(f"{v:+.4f}" for v in point), "interval_excludes_zero": bool(lo > 0 or hi < 0)})
        rc.log(f"{label}: {np.mean(point):+.4f} [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(rows).to_csv(out / "paired_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
