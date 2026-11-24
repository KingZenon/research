"""Experiment 10 (diagnostic): is the native-categorical gain in the 2023 fold due to the handling of UNSEEN labels?

Usage:  python -X utf8 exp10_ordinal_unseen_nan.py <combined_csv> <results_recheck_dir> [--families lightgbm xgboost]

The original ordinal encoder sends a label never seen in training to the integer -1 (smaller than every real code). Here the
categories stay ordinal integers (still treated as continuous numbers, no native categorical splits) but an unseen label becomes
NaN, i.e. 'missing', exactly what the native variants do. If the 2023-fold dip of the original models disappears, the dip came from the
-1 code and not from the lack of native categorical splits. Plain model; prior-adjusted rows derived from it. Validation folds only.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import recheck_common as rc

STAGE = "exp10_ordinal_unseen_nan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--families", nargs="+", default=["lightgbm", "xgboost"])
    args = ap.parse_args()
    out = Path(args.results_dir)
    df, X, y, year = rc.load_pre2026(args.csv)
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        cats = rc.fit_categories(X[tr])
        enc = OrdinalEncoder(categories=[cats[c] for c in rc.CAT], handle_unknown="use_encoded_value", unknown_value=np.nan)
        ctr, cva = enc.fit_transform(X.loc[tr, rc.CAT]), enc.transform(X.loc[va, rc.CAT])
        Xtr = pd.concat([pd.DataFrame(ctr, columns=rc.CAT, index=X.index[tr]), X.loc[tr, rc.NUM]], axis=1)
        Xva = pd.concat([pd.DataFrame(cva, columns=rc.CAT, index=X.index[va]), X.loc[va, rc.NUM]], axis=1)
        for family in args.families:
            if (family, "ordinal_unseen_nan", "plain", val_year) in done:
                continue
            t0 = time.time()
            clf = rc.make_clf(family, False, "default")
            clf.fit(Xtr, y[tr])
            proba = clf.predict_proba(Xva)
            secs = time.time() - t0
            adj, _ = rc.prior_adjust(proba, y[tr])
            rows = [rc.log_row(run_id, STAGE, family, "ordinal_unseen_nan", "plain", val_year, tr, va, y, proba, proba.argmax(1), seconds=secs),
                    rc.log_row(run_id, STAGE, family, "ordinal_unseen_nan", "prior_adjusted", val_year, tr, va, y, proba, adj.argmax(1), calib_proba=adj)]
            rc.append_log(out, rows)
            for r in rows:
                rc.log(f"val {val_year} {family:8s} ordinal_unseen_nan {r['strategy']:14s} F1 {r['fatal_f1']:.3f} rec {r['fatal_recall']:.3f} PR-AUC {r['fatal_pr_auc']:.4f}")
    rc.log("done")


if __name__ == "__main__":
    main()
