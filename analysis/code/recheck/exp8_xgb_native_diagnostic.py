"""Experiment 8 (diagnostic): why does XGBoost with native categoricals do worse than with ordinal integers?

Usage:  python -X utf8 exp8_xgb_native_diagnostic.py <combined_csv> <results_recheck_dir>

Two cheap variants of the native-categorical XGBoost (original fixed settings, plain model; prior-adjusted rows derived from it):
    native_pooled100       labels with fewer than 100 training rows are merged into one '__rare__' label (training counts only)
    native_province_ordinal  province (the widest column, 78 labels) is kept as an ordinal integer, the other six columns are native
These are diagnostics, not new candidates: they were not part of the pre-set comparison. Validation folds only; 2026 never loaded.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import recheck_common as rc

STAGE = "exp8_xgb_native_diagnostic"


def pooled(X_tr, X_va, min_count=100):
    keep = {c: set(X_tr[c].value_counts()[lambda s: s >= min_count].index) for c in rc.CAT}
    f = lambda X: X.assign(**{c: X[c].where(X[c].isin(keep[c]), "__rare__") for c in rc.CAT})
    return f(X_tr), f(X_va)


def main(csv_path, out_dir):
    out = Path(out_dir)
    df, X, y, year = rc.load_pre2026(csv_path)
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        for variant in ("native_pooled100", "native_province_ordinal"):
            if ("xgboost", variant, "plain", val_year) in done:
                continue
            t0 = time.time()
            if variant == "native_pooled100":
                Xtr, Xva = pooled(X[tr], X[va])
                cats_cols = rc.CAT
            else:
                Xtr, Xva = X[tr].copy(), X[va].copy()
                codes = {lab: i for i, lab in enumerate(sorted(Xtr["province"].unique()))}
                for Z in (Xtr, Xva):
                    Z["province"] = Z["province"].map(codes).fillna(-1).astype(float)
                cats_cols = [c for c in rc.CAT if c != "province"]
            cats = {c: sorted(Xtr[c].dropna().unique().tolist()) for c in cats_cols}
            clf = rc.make_clf("xgboost", True, "default")
            clf.fit(rc.to_native(Xtr, cats, cats_cols), y[tr])
            proba = clf.predict_proba(rc.to_native(Xva, cats, cats_cols))
            secs = time.time() - t0
            adj, _ = rc.prior_adjust(proba, y[tr])
            rows = [rc.log_row(run_id, STAGE, "xgboost", variant, "plain", val_year, tr, va, y, proba, proba.argmax(1), seconds=secs),
                    rc.log_row(run_id, STAGE, "xgboost", variant, "prior_adjusted", val_year, tr, va, y, proba, adj.argmax(1), calib_proba=adj)]
            rc.append_log(out, rows)
            for r in rows:
                rc.log(f"val {val_year} xgboost {variant:24s} {r['strategy']:14s} F1 {r['fatal_f1']:.3f} rec {r['fatal_recall']:.3f} PR-AUC {r['fatal_pr_auc']:.4f}")
    rc.log("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
