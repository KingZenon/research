"""Experiment 6: how much do the features that may not be known when an incident is first reported, and the relabelled 2023
columns, contribute?  (audit check; validation folds only, 2026 never loaded)

Usage:  python -X utf8 exp6_feature_ablation.py <combined_csv> <results_recheck_dir> [--family lightgbm] [--sets ...]

Native-categorical LightGBM with the original fixed settings, strategies plain and prior_adjusted, on the same four folds.
Feature sets (columns REMOVED relative to the full 17-feature set used everywhere else):
    no_n_involved               n_involved  ("vehicles and persons involved": counts persons, 2019 file has none)
    no_cause_acctype            cause, acc_type  (presumed cause and collision type: investigator's description written after the event)
    no_cause_acctype_ninvolved  the three above
    no_location_cause_acctype   location, cause, acc_type  (the three columns whose labels were respelt in the 2023 file)
The full-feature reference rows are those of exp1 (native, same family). Log encoding field = 'native__<set>'.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import recheck_common as rc

STAGE = "exp6_feature_ablation"
SETS = {"no_n_involved": ["n_involved"], "no_cause_acctype": ["cause", "acc_type"],
        "no_cause_acctype_ninvolved": ["cause", "acc_type", "n_involved"], "no_location_cause_acctype": ["location", "cause", "acc_type"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--family", default="lightgbm")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    args = ap.parse_args()
    out = Path(args.results_dir)
    df, X, y, year = rc.load_pre2026(args.csv)
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        for name in args.sets:
            enc = f"native__{name}"
            if (args.family, enc, "plain", val_year) in done:
                continue
            drop = SETS[name]
            Xs = X.drop(columns=drop)
            cats_kept = [c for c in rc.CAT if c not in drop]
            cats = {c: sorted(Xs.loc[tr, c].dropna().unique().tolist()) for c in cats_kept}
            t0 = time.time()
            clf = rc.make_clf(args.family, True, "default")
            clf.fit(rc.to_native(Xs[tr], cats, cats_kept), y[tr])
            proba = clf.predict_proba(rc.to_native(Xs[va], cats, cats_kept))
            secs = time.time() - t0
            adj, _ = rc.prior_adjust(proba, y[tr])
            rows = [rc.log_row(run_id, STAGE, args.family, enc, "plain", val_year, tr, va, y, proba, proba.argmax(1), seconds=secs, note=f"dropped {drop}"),
                    rc.log_row(run_id, STAGE, args.family, enc, "prior_adjusted", val_year, tr, va, y, proba, adj.argmax(1), calib_proba=adj, note=f"dropped {drop}")]
            rc.append_log(out, rows)
            for r in rows:
                rc.log(f"val {val_year} {name:28s} {r['strategy']:14s} F1 {r['fatal_f1']:.3f} rec {r['fatal_recall']:.3f} prec {r['fatal_precision']:.3f} PR-AUC {r['fatal_pr_auc']:.4f}")
    rc.log("done")


if __name__ == "__main__":
    main()
