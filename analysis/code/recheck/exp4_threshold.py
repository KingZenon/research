"""Experiment 4: decision threshold on P(fatal) chosen on the INNER year only, applied to the outer validation year.

Usage:  python -X utf8 exp4_threshold.py <combined_csv> <results_recheck_dir> [--families lightgbm xgboost]
                                         [--encodings ordinal native] [--cost-ratio 5]

Needs the outer validation probabilities written by exp1_native_vs_ordinal.py (preds/exp1_*_plain_<year>.npz).
For every fold (train <= L, validate L+1):
  * inner fit : the plain model with the original fixed settings is fitted on years <= L-1 and scores year L (the last
                training year) - the same inner split the nested tuning used. The outer validation year is not touched.
  * thresholds chosen on the inner year (fatal if P(fatal) >= t, otherwise the larger of the other two classes):
        thr_f1_inner       t maximising fatal F1 on the inner year
        thr_cost_inner     t minimising  cost_FN * FN + cost_FP * FP on the inner year (cost ratio FN:FP = --cost-ratio, default 5,
                           an ASSUMED illustrative ratio, not an estimate of real costs)
        thr_cost_theory    t = 1 / (1 + ratio), the Bayes-optimal threshold if P(fatal) were calibrated (no inner data used)
  * diagnostic only: thr_f1_oracle = F1-optimal threshold chosen on the outer year itself (an upper bound, never a candidate)
  * the chosen thresholds are applied to the outer-year probabilities of the plain model, which were produced by a model
    trained on all years <= L.
The inner probabilities are saved to preds/exp4_inner_*.npz (used by the calibration analysis for isotonic recalibration).
"""
import argparse
import time
from pathlib import Path

import numpy as np

import recheck_common as rc

STAGE = "exp4_threshold"


def cost_threshold(truth, p, ratio):
    """Threshold minimising ratio*FN + FP on the given year (ties -> larger threshold)."""
    grid = np.unique(np.concatenate([np.quantile(p, np.linspace(0.01, 0.99, 197)), [0.5]]))
    best_t, best_c = 0.5, np.inf
    for t in grid:
        pred = p >= t
        c = ratio * np.sum(~pred & (truth == 1)) + np.sum(pred & (truth == 0))
        if c <= best_c:
            best_t, best_c = float(t), c
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--families", nargs="+", default=["lightgbm", "xgboost"])
    ap.add_argument("--encodings", nargs="+", default=["ordinal", "native"])
    ap.add_argument("--cost-ratio", type=float, default=5.0)
    args = ap.parse_args()
    out = Path(args.results_dir)

    df, X, y, year = rc.load_pre2026(args.csv)
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    r = args.cost_ratio

    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        inner_tr, inner_va = year <= last_train - 1, year == last_train
        cats = rc.fit_categories(X[inner_tr])
        for family in args.families:
            for enc in args.encodings:
                native = enc == "native"
                if (family, enc, "thr_f1_inner", val_year) in done:
                    continue
                t0 = time.time()
                p_in, = rc.fit_predict(family, native, "default", X[inner_tr], y[inner_tr], [X[inner_va]], categories=cats)
                rc.save_proba(out, f"exp4_inner_{family}_{enc}_plain", val_year, np.flatnonzero(inner_va), p_in)
                secs = time.time() - t0
                truth_in = (y[inner_va] == 0).astype(int)
                t_f1, f1_in = rc.best_f1_threshold(truth_in, p_in[:, 0])
                t_cost = cost_threshold(truth_in, p_in[:, 0], r)
                t_theory = 1.0 / (1.0 + r)
                idx, p_out = rc.load_proba(out, f"exp1_native_vs_ordinal_{family}_{enc}_plain", val_year)
                assert np.array_equal(idx, np.flatnonzero(va)), "outer predictions do not match the fold"
                t_oracle, _ = rc.best_f1_threshold((y[va] == 0).astype(int), p_out[:, 0])
                rows = []
                for name, t, note in (("thr_f1_inner", t_f1, f"inner-year best F1 {f1_in:.3f}"),
                                      ("thr_cost_inner", t_cost, f"cost ratio FN:FP = {r:g} (assumed)"),
                                      ("thr_cost_theory", t_theory, f"cost ratio FN:FP = {r:g} (assumed), no inner data"),
                                      ("thr_f1_oracle", t_oracle, "DIAGNOSTIC ONLY: threshold picked on the outer year itself")):
                    pred = rc.predict_with_threshold(p_out, t)
                    rows.append(rc.log_row(run_id, STAGE, family, enc, name, val_year, tr, va, y, p_out, pred,
                                           seconds=secs, threshold=round(t, 5), note=note))
                rc.append_log(out, rows)
                for row in rows:
                    rc.log(f"val {val_year} {family:8s} {enc:8s} {row['strategy']:16s} t={float(row['threshold']):.3f} F1 {row['fatal_f1']:.3f} "
                           f"rec {row['fatal_recall']:.3f} prec {row['fatal_precision']:.3f} macroF1 {row['macro_f1']:.3f}")
    rc.log("done")


if __name__ == "__main__":
    main()
