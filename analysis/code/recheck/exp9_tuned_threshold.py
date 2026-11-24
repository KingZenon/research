"""Experiment 9 (added after seeing exp4 and exp7 - flagged as post hoc): inner-year F1 threshold for the TUNED native LightGBM.

Usage:  python -X utf8 exp9_tuned_threshold.py <combined_csv> <results_recheck_dir> [--family lightgbm]

Combines exp7 (tuned native model: configuration chosen on the inner year by fatal PR-AUC) with exp4's decision rule: for each
fold the winning configuration is refitted on years <= L-1, its P(fatal) on the inner year L gives the F1-optimal threshold, and
that threshold is applied to the outer-year probabilities of the tuned model fitted on years <= L (preds from exp7).
The outer validation year is not used to choose anything. Because it was added after the earlier results were seen, it is one
more candidate in the selection and increases selection optimism; it is reported as such.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import recheck_common as rc

STAGE = "exp9_tuned_threshold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--family", default="lightgbm")
    args = ap.parse_args()
    out = Path(args.results_dir)
    df, X, y, year = rc.load_pre2026(args.csv)
    log = pd.read_csv(out / "experiment_log.csv")
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    for last_train, val_year in rc.FOLDS:
        if (args.family, "native_tuned", "thr_f1_inner", val_year) in done:
            continue
        row = log[(log.stage == "exp7_tuning_native") & (log.family == args.family) & (log.strategy == "plain") & (log.val_year == val_year)]
        if row.empty:
            rc.log(f"fold {val_year}: no tuned model in the log yet, skipped")
            continue
        params = json.loads(row["params"].iloc[0])
        tr, va = year <= last_train, year == val_year
        inner_tr, inner_va = year <= last_train - 1, year == last_train
        t0 = time.time()
        p_in, = rc.fit_predict(args.family, True, "default", X[inner_tr], y[inner_tr], [X[inner_va]], params=params, categories=rc.fit_categories(X[inner_tr]))
        secs = time.time() - t0
        t_f1, f1_in = rc.best_f1_threshold((y[inner_va] == 0).astype(int), p_in[:, 0])
        idx, p_out = rc.load_proba(out, f"exp7_tuning_native_{args.family}_native_plain", val_year)
        assert np.array_equal(idx, np.flatnonzero(va))
        pred = rc.predict_with_threshold(p_out, t_f1)
        r = rc.log_row(run_id, STAGE, args.family, "native_tuned", "thr_f1_inner", val_year, tr, va, y, p_out, pred, params=json.dumps(params, sort_keys=True),
                       seconds=secs, threshold=round(t_f1, 5), note=f"inner-year best F1 {f1_in:.3f}; POST HOC candidate")
        rc.append_log(out, [r])
        rc.log(f"val {val_year} tuned native threshold t={t_f1:.3f} F1 {r['fatal_f1']:.3f} rec {r['fatal_recall']:.3f} prec {r['fatal_precision']:.3f} macroF1 {r['macro_f1']:.3f}")
    rc.log("done")


if __name__ == "__main__":
    main()
