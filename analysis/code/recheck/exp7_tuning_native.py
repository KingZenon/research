"""Experiment 7 (optional): nested, time-aware tuning of the native-categorical model with the ORIGINAL budget.

Usage:  python -X utf8 exp7_tuning_native.py <combined_csv> <results_recheck_dir> [--family lightgbm] [--n-random 9]

For each outer fold (train <= L, validate L+1): the last training year L is the inner validation year; the default configuration
plus 9 random ones (the SAME list cv_tools.sample_configs gives the original study, seed 42) are fitted on years < L, and the
configuration with the best inner fatal-class PR-AUC wins. The outer validation year is never seen during tuning. The winner is
then fitted on all years <= L with the strategies plain / class_weight (prior_adjusted derived from plain) and logged.
Inner results go to tuning_native_<family>.csv (append-only).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import recheck_common as rc

STAGE = "exp7_tuning_native"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--family", default="lightgbm")
    ap.add_argument("--n-random", type=int, default=9)
    args = ap.parse_args()
    out = Path(args.results_dir)
    df, X, y, year = rc.load_pre2026(args.csv)
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    configs = rc.cv.sample_configs(args.family, args.n_random)
    for last_train, val_year in rc.FOLDS:
        if (args.family, "native", "plain", val_year) in done:
            continue
        tr, va = year <= last_train, year == val_year
        inner_tr, inner_va = year <= last_train - 1, year == last_train
        truth = (y[inner_va] == 0).astype(int)
        cats_in = rc.fit_categories(X[inner_tr])
        trows, best, best_ap = [], None, -1.0
        for i, cfg in enumerate(configs):
            t0 = time.time()
            p, = rc.fit_predict(args.family, True, "default", X[inner_tr], y[inner_tr], [X[inner_va]], params=cfg, categories=cats_in)
            ap_i = float(average_precision_score(truth, p[:, 0]))
            secs = time.time() - t0
            trows.append({"run_id": run_id, "val_year": val_year, "inner_year": last_train, "config_id": i, "params": json.dumps(cfg, sort_keys=True),
                          "inner_pr_auc": round(ap_i, 5), "seconds": round(secs, 1)})
            if ap_i > best_ap:
                best, best_ap = cfg, ap_i
            rc.log(f"  val {val_year} tune cfg {i:2d} inner PR-AUC {ap_i:.4f} ({secs:.0f}s)")
        import csv
        path = out / f"tuning_native_{args.family}.csv"
        with path.open("a", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(trows[0].keys()))
            if path.stat().st_size == 0:
                w.writeheader()
            w.writerows(trows)
        cats = rc.fit_categories(X[tr])
        rows = []
        for variant, strat in (("default", "plain"), ("class_weight", "class_weight")):
            t0 = time.time()
            proba, = rc.fit_predict(args.family, True, variant, X[tr], y[tr], [X[va]], params=best, categories=cats)
            secs = time.time() - t0
            rc.save_proba(out, f"{STAGE}_{args.family}_native_{strat}", val_year, np.flatnonzero(va), proba)
            rows.append(rc.log_row(run_id, STAGE, args.family, "native", strat, val_year, tr, va, y, proba, proba.argmax(1),
                                   params=json.dumps(best, sort_keys=True), seconds=secs, note=f"inner PR-AUC {best_ap:.4f}; default config id 0"))
            if strat == "plain":
                adj, _ = rc.prior_adjust(proba, y[tr])
                rows.append(rc.log_row(run_id, STAGE, args.family, "native", "prior_adjusted", val_year, tr, va, y, proba, adj.argmax(1),
                                       params=json.dumps(best, sort_keys=True), calib_proba=adj, note=f"inner PR-AUC {best_ap:.4f}"))
        rc.append_log(out, rows)
        for r in rows:
            rc.log(f"val {val_year} TUNED {args.family} native {r['strategy']:14s} F1 {r['fatal_f1']:.3f} rec {r['fatal_recall']:.3f} PR-AUC {r['fatal_pr_auc']:.4f}  cfg {r['params']}")
    rc.log("done")


if __name__ == "__main__":
    main()
