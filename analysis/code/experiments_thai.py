"""Temporal cross-validation and imbalance experiments on the Thai MOT accident data.

Usage:  python -I -X utf8 experiments_thai.py <combined_csv> <results_dir>

Rolling-origin (expanding window) folds, trained only on earlier years:
    train 2019-2021 -> validate 2022 | 2019-2022 -> 2023 | 2019-2023 -> 2024 | 2019-2024 -> 2025
The 2026 test year is never scored here.

Variants compared for each model family
    default           plain model (the baseline)
    class_weight      minority classes weighted more in training (balanced weights)
    prior_adjusted    plain model, but each class probability is divided by its training-set share
                      before choosing the label (a parameter-free threshold shift towards fatal)
Reported per variant: mean and standard deviation over the 4 folds.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

import baseline_models as bm

FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]   # (last train year, validation year)
MODELS = ["majority_class", "logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]


def fit_variant(model_name, variant, X_tr, y_tr):
    model = bm.make_models()[model_name]
    fit_kwargs = {}
    if variant == "class_weight":
        if model_name == "logistic_regression":
            model.set_params(clf__class_weight="balanced")
        elif model_name == "decision_tree":
            model.set_params(clf__class_weight="balanced")
        elif model_name == "random_forest":
            model.set_params(clf__class_weight="balanced_subsample")
        elif model_name == "lightgbm":
            model.set_params(clf__class_weight="balanced")
        elif model_name == "xgboost":   # multiclass XGBoost has no class_weight option: use sample weights
            fit_kwargs["clf__sample_weight"] = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr, y_tr, **fit_kwargs)
    return model


def score(y, proba, pred):
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "fatal_recall": rec[0],
        "fatal_precision": precision_score(y, pred, labels=[0], average=None, zero_division=0)[0],
        "fatal_f1": f1_score(y, pred, labels=[0], average=None, zero_division=0)[0],
        "fatal_pr_auc": average_precision_score((y == 0).astype(int), proba[:, 0]),
        "fatal_auc": roc_auc_score((y == 0).astype(int), proba[:, 0]),
    }


def main(csv, out_dir):
    df = bm.load(csv, "all")
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    rows = []
    for model_name in MODELS:
        # the majority-class baseline has no probabilities to re-weight: default only
        for variant in (("default",) if model_name == "majority_class" else ("default", "class_weight")):
            for last_train, val_year in FOLDS:
                tr, va = year <= last_train, year == val_year
                t0 = time.time()
                model = fit_variant(model_name, variant, X[tr], y[tr])
                proba = model.predict_proba(X[va])
                variants = {variant: proba.argmax(axis=1)}
                if variant == "default" and model_name != "majority_class":   # prior-adjusted decisions reuse the default model's probabilities
                    prior = np.bincount(y[tr], minlength=3) / tr.sum()
                    variants["prior_adjusted"] = (proba / prior).argmax(axis=1)
                for v, pred in variants.items():
                    r = {"model": model_name, "variant": v, "val_year": val_year, "n_train": int(tr.sum()),
                         "n_val": int(va.sum()), "fatal_share_val": float((y[va] == 0).mean())}
                    r.update(score(y[va], proba, pred))
                    rows.append(r)
                print(f"{model_name:20s} {variant:13s} val {val_year}  {time.time() - t0:5.1f}s", flush=True)
    res = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "cv_temporal_thai_folds.csv", index=False)
    metrics = ["macro_f1", "balanced_acc", "mcc", "fatal_recall", "fatal_precision", "fatal_f1", "fatal_pr_auc", "fatal_auc", "accuracy"]
    g = res.groupby(["model", "variant"])[metrics]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).reset_index()
    summary["n_folds"] = len(FOLDS)
    summary.to_csv(out / "cv_temporal_thai_summary.csv", index=False)
    pd.set_option("display.width", 250)
    show = pd.DataFrame({"model": summary.model, "variant": summary.variant})
    for m in ("macro_f1", "balanced_acc", "fatal_recall", "fatal_precision", "fatal_f1", "fatal_pr_auc"):
        show[m] = summary[f"{m}_mean"].round(3).astype(str) + " ± " + summary[f"{m}_std"].round(3).astype(str)
    print("\nmean ± std over 4 temporal folds (validation years 2022-2025)\n")
    print(show.sort_values(["model", "variant"]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
