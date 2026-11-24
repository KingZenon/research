"""Experiment 2: SMOTE-NC with numeric features standardised before the nearest-neighbour search, plus controls.

Usage:  python -X utf8 exp2_smote_scaled.py <combined_csv> <results_recheck_dir> [--families lightgbm xgboost]
                                            [--variants control_native smote_scaled_native smote_scaled_ordinal smote_unscaled_native]
                                            [--sample 1.0]

All steps are fitted on the training years of each rolling-origin fold only:
  1. ordinal codes for the 7 categorical columns (categories learnt from the training fold, unseen label -> -1)
  2. numeric columns: median imputation (SMOTE needs complete data), then StandardScaler (scaled variants only)
  3. SMOTE-NC (categorical columns flagged, k = 5, seed 42): the fatal class is oversampled up to the size of the smaller of
     the other two classes - the same target as the original study, so the results are comparable
  4. the boosted model (original fixed settings) is fitted on the resampled data; validation rows go through the same
     imputer / scaler / category codes
Variants
  control_native         imputation (+ scaling) only, NO resampling, native categoricals   (isolates the imputation effect)
  control_ordinal        same, ordinal integers                                            (reproduces the original control)
  smote_scaled_native    SMOTE-NC on standardised numerics, native categoricals downstream
  smote_scaled_ordinal   SMOTE-NC on standardised numerics, ordinal integers downstream
  smote_unscaled_native  SMOTE-NC on UNSCALED numerics (as the original), native categoricals downstream
The original unscaled + ordinal SMOTE result is already in results/cv_thai_all_folds.csv (strategy 'smote').
The 2026 year is never loaded.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import recheck_common as rc

STAGE = "exp2_smote"
DEFAULT_VARIANTS = ["control_native", "control_ordinal", "smote_scaled_native", "smote_scaled_ordinal", "smote_unscaled_native"]


def prepare(X_tr, X_va, cats, scale):
    """Fit encoder / imputer / scaler on the training fold; return numeric blocks and category codes for both frames."""
    enc = OrdinalEncoder(categories=[cats[c] for c in rc.CAT], handle_unknown="use_encoded_value", unknown_value=-1)
    c_tr, c_va = enc.fit_transform(X_tr[rc.CAT]), enc.transform(X_va[rc.CAT])
    imp = SimpleImputer(strategy="median")
    n_tr, n_va = imp.fit_transform(X_tr[rc.NUM]), imp.transform(X_va[rc.NUM])
    if scale:
        sc = StandardScaler()
        n_tr, n_va = sc.fit_transform(n_tr), sc.transform(n_va)
    return c_tr, c_va, n_tr, n_va


def frame(codes, nums, cats, native):
    data = {}
    for j, c in enumerate(rc.CAT):
        col = np.asarray(codes[:, j]).round().astype(int)
        data[c] = pd.Categorical.from_codes(col, categories=cats[c]) if native else col.astype(float)
    for j, c in enumerate(rc.NUM):
        data[c] = nums[:, j]
    return pd.DataFrame(data)


def resample(c_tr, n_tr, y_tr):
    """SMOTE-NC on [category codes | numerics]; fatal -> size of the smaller other class (as in the original study)."""
    Z = np.hstack([c_tr, n_tr])
    counts = np.bincount(y_tr, minlength=3)
    target = int(min(counts[1], counts[2]))
    sampler = SMOTENC(categorical_features=list(range(len(rc.CAT))), sampling_strategy={0: target}, k_neighbors=5, random_state=rc.SEED)
    Zr, yr = sampler.fit_resample(Z, y_tr)
    return Zr[:, :len(rc.CAT)], Zr[:, len(rc.CAT):], yr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--families", nargs="+", default=["lightgbm", "xgboost"])
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--sample", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, X, y, year = rc.load_pre2026(args.csv)
    if args.sample < 1.0:
        keep = np.random.default_rng(rc.SEED).random(len(df)) < args.sample
        X, y, year = X[keep].reset_index(drop=True), y[keep], year[keep]
    rc.log(f"seed {rc.SEED}; rows {len(y):,}; variants {args.variants}; n_jobs {rc.N_JOBS}")
    done = rc.logged_keys(out, STAGE)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        cats = rc.fit_categories(X[tr])
        y_tr = y[tr]
        prep = {}
        for scale in (True, False):
            prep[scale] = prepare(X[tr], X[va], cats, scale)
        resampled = {}
        for variant in args.variants:
            parts = variant.split("_")                      # e.g. smote_scaled_native / control_native
            is_smote = parts[0] == "smote"
            scale = (parts[1] == "scaled") if is_smote else True
            native = parts[-1] == "native"
            c_tr, c_va, n_tr, n_va = prep[scale]
            if is_smote:
                if scale not in resampled:
                    t0 = time.time()
                    resampled[scale] = resample(c_tr, n_tr, y_tr)
                    rc.log(f"  val {val_year} SMOTE-NC (scaled={scale}): {int((resampled[scale][2] == 0).sum() - (y_tr == 0).sum()):,} synthetic "
                           f"fatal rows added in {time.time() - t0:.0f}s")
                rc_tr, rn_tr, ry = resampled[scale]
                Xf, yf = frame(rc_tr, rn_tr, cats, native), ry
            else:
                Xf, yf = frame(c_tr, n_tr, cats, native), y_tr
            Xv = frame(c_va, n_va, cats, native)
            for family in args.families:
                if (family, variant, "plain", val_year) in done:
                    continue
                t0 = time.time()
                clf = rc.make_clf(family, native, "default")
                clf.fit(Xf, yf)
                proba = clf.predict_proba(Xv)
                secs = time.time() - t0
                rc.save_proba(out, f"{STAGE}_{family}_{variant}", val_year, np.flatnonzero(va), proba)
                note = f"n_fit_rows={len(yf)}; fatal_in_fit={int((yf == 0).sum())}"
                row = rc.log_row(run_id, STAGE, family, variant, "plain", val_year, tr, va, y, proba, proba.argmax(1), seconds=secs, note=note)
                rc.append_log(out, [row])
                rc.log(f"val {val_year} {family:8s} {variant:22s} F1 {row['fatal_f1']:.3f} rec {row['fatal_recall']:.3f} "
                       f"prec {row['fatal_precision']:.3f} PR-AUC {row['fatal_pr_auc']:.4f} ({secs:.0f}s)")
    rc.log("done")


if __name__ == "__main__":
    main()
