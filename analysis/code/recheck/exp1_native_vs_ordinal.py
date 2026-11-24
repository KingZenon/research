"""Experiment 1: native categorical handling vs the original ordinal-integer encoding.

Usage:  python -X utf8 exp1_native_vs_ordinal.py <combined_csv> <results_recheck_dir> [--families lightgbm xgboost]
                                                 [--encodings ordinal native] [--sample 1.0]

For each rolling-origin fold (train <= 2021 -> 2022, ..., <= 2024 -> 2025) and each family it fits, with the ORIGINAL fixed
settings (LightGBM 400 trees lr 0.05 63 leaves; XGBoost 400 trees lr 0.1 depth 6, seed 42):
    plain          argmax of the model (default cut-off)
    class_weight   balanced class weights (LightGBM class_weight, XGBoost sample weights)
    prior_adjusted the plain model's probabilities divided by the training class shares, then argmax
and writes one row per (family, encoding, strategy, fold) to the append-only experiment_log.csv, plus the validation
probabilities to preds/ (used later for calibration, thresholds and the error analysis).
encoding = ordinal : the original code path (cv_tools.fit_variant) -> reproduces the original CSVs (a check)
encoding = native  : pandas category dtype (LightGBM) / enable_categorical (XGBoost); labels unseen in the training
                     years become missing values. The 2026 year is never loaded.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import recheck_common as rc

STAGE = "exp1_native_vs_ordinal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--families", nargs="+", default=["lightgbm", "xgboost"])
    ap.add_argument("--encodings", nargs="+", default=["ordinal", "native"])
    ap.add_argument("--sample", type=float, default=1.0, help="row fraction for smoke tests only")
    args = ap.parse_args()
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, X, y, year = rc.load_pre2026(args.csv)
    if args.sample < 1.0:
        keep = np.random.default_rng(rc.SEED).random(len(df)) < args.sample
        X, y, year = X[keep].reset_index(drop=True), y[keep], year[keep]
    rc.log(f"seed {rc.SEED}; rows {len(y):,}; years {year.min()}-{year.max()}; fatal share {np.mean(y == 0):.4f}; n_jobs {rc.N_JOBS}")
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        cats = rc.fit_categories(X[tr])
        for family in args.families:
            for enc in args.encodings:
                if all((family, enc, s, val_year) in done for s in ("plain", "class_weight", "prior_adjusted")):
                    continue
                native = enc == "native"
                rows = []
                t0 = time.time()
                p_plain, = rc.fit_predict(family, native, "default", X[tr], y[tr], [X[va]], categories=cats)
                secs = time.time() - t0
                rc.save_proba(out, f"{STAGE}_{family}_{enc}_plain", val_year, np.flatnonzero(va), p_plain)
                rows.append(rc.log_row(run_id, STAGE, family, enc, "plain", val_year, tr, va, y, p_plain, p_plain.argmax(1), seconds=secs))
                adj, _ = rc.prior_adjust(p_plain, y[tr])
                rows.append(rc.log_row(run_id, STAGE, family, enc, "prior_adjusted", val_year, tr, va, y, p_plain, adj.argmax(1),
                                       calib_proba=adj, note="Brier/ECE on prior-adjusted (renormalised) probabilities"))
                t0 = time.time()
                p_cw, = rc.fit_predict(family, native, "class_weight", X[tr], y[tr], [X[va]], categories=cats)
                secs = time.time() - t0
                rc.save_proba(out, f"{STAGE}_{family}_{enc}_class_weight", val_year, np.flatnonzero(va), p_cw)
                rows.append(rc.log_row(run_id, STAGE, family, enc, "class_weight", val_year, tr, va, y, p_cw, p_cw.argmax(1), seconds=secs))
                rc.append_log(out, [r for r in rows if (r["family"], r["encoding"], r["strategy"], r["val_year"]) not in done])
                for r in rows:
                    rc.log(f"val {val_year} {family:8s} {enc:8s} {r['strategy']:14s} F1 {r['fatal_f1']:.3f} rec {r['fatal_recall']:.3f} "
                           f"prec {r['fatal_precision']:.3f} PR-AUC {r['fatal_pr_auc']:.4f} Brier {r['brier_fatal']:.4f} ECE {r['ece_fatal']:.4f}")
    rc.log("done")


if __name__ == "__main__":
    main()
