"""Shared helpers for the re-check of the ICT 324 severity results (all scripts in this folder import this file).

Hard rules enforced here
  * the 2026 test year is dropped immediately after the CSV is read (acc_year <= 2025 only); an assertion guards it;
  * the existing code in ../ is only imported (never modified);
  * every result row is appended to results_recheck/experiment_log.csv after each fold (append-only);
  * seed 42, same rolling-origin folds as the original study.

Run scripts with:  python -X utf8 <script>.py <combined_csv> <results_recheck_dir> [options]   (never python -I)
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))          # the original scripts, imported read-only
import baseline_models as bm               # noqa: E402
import cv_tools as cv                      # noqa: E402

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

SEED = 42
FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
CAT, NUM = bm.CATEGORICAL, bm.NUMERIC
N_JOBS = int(os.environ.get("CV_NJOBS", "8"))
LOG_COLUMNS = ["run_id", "stage", "family", "encoding", "strategy", "val_year", "n_train", "n_val", "fatal_share_val",
               "macro_f1", "balanced_acc", "accuracy", "mcc", "fatal_recall", "fatal_precision", "fatal_f1",
               "fatal_pr_auc", "fatal_auc", "brier_fatal", "ece_fatal", "threshold", "params", "seconds", "note"]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ------------------------------------------------------------------------------------------------
# data: 2026 never enters
# ------------------------------------------------------------------------------------------------
def load_pre2026(csv_path):
    """Read the harmonised table, drop 2026 (and undated rows) at once, then apply the original duplicate rule.

    Returns (df, X, y, year). The row set equals the original study's 2019-2025 rows (checked: 151,314)."""
    raw = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    raw = raw[raw["acc_year"].notna() & (raw["acc_year"] <= 2025)].copy()
    assert raw["acc_year"].max() <= 2025, "2026 rows must not be loaded"
    df = raw[raw["severity"].isin(bm.CLASSES)].copy()
    key = ["acc_code", "date_parsed", "time", "province", "deaths", "injured_total"]   # same rule as baseline_models.load
    dup = df["acc_code"].notna() & df.duplicated(subset=key, keep="first")
    df = df[~dup].reset_index(drop=True)
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy().astype(int)
    assert year.max() <= 2025
    return df, X, y, year


# ------------------------------------------------------------------------------------------------
# native categorical frames
# ------------------------------------------------------------------------------------------------
def fit_categories(X_tr, cats=CAT):
    """Category lists learnt from the training fold only (sorted, so the codes are reproducible)."""
    return {c: sorted(X_tr[c].dropna().unique().tolist()) for c in cats}


def to_native(X, categories, cats=CAT):
    """Cast the categorical columns to pandas 'category' with the TRAINING categories. A label that never occurred in
    training becomes NaN (missing), which LightGBM and XGBoost both route down a fixed default branch: no crash,
    no arbitrary integer code."""
    out = X.copy()
    for c in cats:
        out[c] = pd.Categorical(out[c].astype(object), categories=categories[c])
    return out


def unseen_share(X_tr, X_va, cats=CAT):
    """Share of validation rows whose label was never seen in the training years, per categorical column."""
    return {c: float((~X_va[c].isin(set(X_tr[c].unique()))).mean()) for c in cats}


# ------------------------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------------------------
def make_clf(family, native, variant, params=None):
    """Fixed original settings (LightGBM 400 trees lr 0.05 63 leaves; XGBoost 400 trees lr 0.1 depth 6)."""
    params = dict(params or {})
    if family == "lightgbm":
        from lightgbm import LGBMClassifier
        kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, n_jobs=N_JOBS, random_state=SEED, verbose=-1)
        kw.update(params)
        if variant == "class_weight":
            kw["class_weight"] = "balanced"
        return LGBMClassifier(**kw)
    from xgboost import XGBClassifier
    kw = dict(n_estimators=400, learning_rate=0.1, max_depth=6, tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
              enable_categorical=bool(native))
    kw.update(params)
    return XGBClassifier(**kw)


def fit_predict(family, native, variant, X_tr, y_tr, X_ev_list, params=None, categories=None):
    """Fit one model on the training fold and return a list of probability arrays, one per evaluation frame.

    native=True : pandas category dtype (LightGBM) / enable_categorical (XGBoost), unseen labels -> NaN
    native=False: the ORIGINAL code path (ordinal integers as continuous features), via cv_tools.fit_variant"""
    if not native:
        model = cv.fit_variant(family, variant, X_tr, y_tr, CAT, NUM, params or {})
        return [model.predict_proba(X) for X in X_ev_list]
    cats = categories if categories is not None else fit_categories(X_tr)
    clf = make_clf(family, True, variant, params)
    kwargs = {}
    if variant == "class_weight" and family == "xgboost":
        from sklearn.utils.class_weight import compute_sample_weight
        kwargs["sample_weight"] = compute_sample_weight("balanced", y_tr)
    clf.fit(to_native(X_tr, cats), y_tr, **kwargs)
    return [clf.predict_proba(to_native(X, cats)) for X in X_ev_list]


# ------------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------------
def ece_binary(truth, p, n_bins=10):
    """Expected calibration error of P(fatal): equal-width bins, weighted mean |observed rate - mean predicted|."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total, n = 0.0, len(p)
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() / n * abs(truth[m].mean() - p[m].mean())
    return float(total)


def ece_quantile(truth, p, n_bins=10):
    """Same with equal-count bins (more informative when almost all probabilities are small)."""
    order = np.argsort(p)
    total, n = 0.0, len(p)
    for chunk in np.array_split(order, n_bins):
        total += len(chunk) / n * abs(truth[chunk].mean() - p[chunk].mean())
    return float(total)


def full_score(y, proba, pred, calib_proba=None):
    """cv_tools.score (the original metrics) + Brier and ECE of P(fatal).
    calib_proba: probabilities used for Brier/ECE only (e.g. prior-adjusted); ranking metrics always use proba."""
    s = cv.score(y, proba, pred)
    t = (y == 0).astype(int)
    pf = np.clip((proba if calib_proba is None else calib_proba)[:, 0], 0.0, 1.0)
    s["brier_fatal"] = float(np.mean((pf - t) ** 2))
    s["ece_fatal"] = ece_binary(t, pf)
    return s


def prior_adjust(proba, y_tr):
    prior = np.bincount(y_tr, minlength=3) / len(y_tr)
    adj = proba / prior
    return adj / adj.sum(axis=1, keepdims=True), prior


def predict_with_threshold(proba, t):
    """Fatal if P(fatal) >= t, otherwise the larger of the two other classes."""
    pred = np.where(proba[:, 1] >= proba[:, 2], 1, 2)
    pred[proba[:, 0] >= t] = 0
    return pred


def best_f1_threshold(truth, p):
    """Threshold on p maximising F1 of the positive class, scanned over the precision-recall curve."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(truth, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


# ------------------------------------------------------------------------------------------------
# append-only log
# ------------------------------------------------------------------------------------------------
def append_log(results_dir, rows):
    path = Path(results_dir) / "experiment_log.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, extrasaction="ignore", restval="")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def log_row(run_id, stage, family, encoding, strategy, val_year, tr, va, y, proba, pred, params="{}", seconds=0.0,
            threshold="", note="", calib_proba=None):
    s = full_score(y[va], proba, pred, calib_proba)
    row = {"run_id": run_id, "stage": stage, "family": family, "encoding": encoding, "strategy": strategy,
           "val_year": int(val_year), "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "fatal_share_val": float((y[va] == 0).mean()), "threshold": threshold, "params": params,
           "seconds": round(seconds, 1), "note": note}
    row.update({k: (round(float(v), 6)) for k, v in s.items() if k in LOG_COLUMNS})
    return row


def save_proba(results_dir, name, val_year, idx, proba):
    d = Path(results_dir) / "preds"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{name}_{val_year}.npz", idx=idx.astype(np.int32), proba=proba.astype(np.float32))


def load_proba(results_dir, name, val_year):
    z = np.load(Path(results_dir) / "preds" / f"{name}_{val_year}.npz")
    return z["idx"], z["proba"].astype(np.float64)


def logged_keys(results_dir, stage):
    """(family, encoding, strategy, val_year) already in the log for this stage, so a restarted run skips finished work."""
    path = Path(results_dir) / "experiment_log.csv"
    if not path.exists():
        return set()
    d = pd.read_csv(path)
    d = d[d["stage"] == stage]
    return set(zip(d["family"], d["encoding"], d["strategy"], d["val_year"].astype(int)))
