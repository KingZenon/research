"""Shared pieces of the final experiments (tuning, SMOTE, rolling-origin CV).

Used by experiments_final.py (Thai data), uk_replication.py (UK STATS19) and final_test_2026.py.
Everything here works on a feature table X, an integer label y (0 = fatal, 1 = middle class,
2 = least severe) and a year array, so the same code runs on both countries.

Imbalance strategies ("variants"):
    default         plain model
    class_weight    balanced class weights in training
    prior_adjusted  plain model, class probabilities divided by their training share before the label is chosen
    smote           SMOTE-NC on the training years only: the fatal class is oversampled up to the size of the
                    smaller of the other two classes (the numeric gaps are median-imputed for this variant,
                    because SMOTE needs complete data)

Tuning is time-aware and nested: for each outer fold the last training year is held out as an inner
validation year, a fixed set of random configurations (plus the default one) is fitted on the earlier years,
and the one with the best fatal-class PR-AUC on the inner year wins. The outer validation year is never seen.
"""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import baseline_models as bm

SEED = bm.SEED
N_JOBS = int(os.environ.get("CV_NJOBS", "-1"))
VARIANTS = ["default", "class_weight", "prior_adjusted", "smote"]

# search spaces for the two boosted families (the best families in the Milestone 3 cross-validation)
SPACES = {
    "xgboost": {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [200, 400, 600],
                "min_child_weight": [1, 5, 20], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [1, 5, 20]},
    "random_forest": {"n_estimators": [200, 300, 500], "min_samples_leaf": [1, 2, 5, 10], "max_features": ["sqrt", 0.3, 0.5],
                      "max_depth": [None, 12, 20]},
    "lightgbm": {"num_leaves": [31, 63, 127, 255], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [200, 400, 600],
                 "min_child_samples": [20, 50, 100], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
                 "reg_lambda": [0, 5, 20]},
}


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def make_model(name, categorical, numeric, params=None):
    model = bm.make_models(categorical, numeric)[name]
    if name in ("random_forest", "xgboost", "lightgbm"):
        model.set_params(clf__n_jobs=N_JOBS)
    for key, value in (params or {}).items():
        model.set_params(**{f"clf__{key}": value})
    return model


def sample_configs(name, n_random, seed=SEED):
    """The default configuration first, then n_random distinct random ones (the same list for every fold)."""
    rng = np.random.default_rng(seed)
    space = SPACES[name]
    configs, seen = [{}], set()
    while len(configs) < n_random + 1:
        cfg = {key: values[int(rng.integers(len(values)))] for key, values in space.items()}
        if name == "lightgbm" and cfg["subsample"] < 1.0:
            cfg["subsample_freq"] = 1          # LightGBM only subsamples rows when this is set
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


class SmoteModel:
    """Prediction wrapper for a classifier that was fitted on SMOTE-NC resampled training data."""

    def __init__(self, prep, post, clf):
        self.prep, self.post, self.clf = prep, post, clf

    def predict_proba(self, X):
        Z = self.prep.transform(X)
        if self.post is not None:
            Z = self.post.transform(Z)
        return self.clf.predict_proba(Z)


def _smote_data(X_tr, y_tr, categorical, numeric, cache, key):
    if cache is not None and key in cache:
        return cache[key]
    prep = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ("num", SimpleImputer(strategy="median"), numeric)])
    Z = prep.fit_transform(X_tr)
    counts = np.bincount(y_tr, minlength=3)
    target = int(min(counts[1], counts[2]))
    if counts[0] < target:
        sampler = SMOTENC(categorical_features=list(range(len(categorical))), sampling_strategy={0: target},
                          k_neighbors=5, random_state=SEED)
        Zr, yr = sampler.fit_resample(Z, y_tr)
    else:
        Zr, yr = Z, y_tr
    if cache is not None:
        cache[key] = (prep, Zr, yr)
    return prep, Zr, yr


def fit_variant(name, variant, X_tr, y_tr, categorical, numeric, params=None, cache=None, cache_key=None):
    if name == "majority_class":
        model = bm.make_models(categorical, numeric)[name]
        model.fit(X_tr, y_tr)
        return model
    if variant == "smote":
        prep, Zr, yr = _smote_data(X_tr, y_tr, categorical, numeric, cache, cache_key)
        post = None
        if name == "logistic_regression":      # the linear model needs one-hot categories and scaled numbers
            c = len(categorical)
            post = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), list(range(c))),
                                      ("num", StandardScaler(), list(range(c, c + len(numeric))))])
            Zr = post.fit_transform(Zr)
        clf = make_model(name, categorical, numeric, params).named_steps["clf"]
        clf.fit(Zr, yr)
        return SmoteModel(prep, post, clf)
    model = make_model(name, categorical, numeric, params)
    kwargs = {}
    if variant == "class_weight":
        if name in ("logistic_regression", "decision_tree"):
            model.set_params(clf__class_weight="balanced")
        elif name == "random_forest":
            model.set_params(clf__class_weight="balanced_subsample")
        elif name == "lightgbm":
            model.set_params(clf__class_weight="balanced")
        elif name == "xgboost":              # multiclass XGBoost has no class_weight option: sample weights
            kwargs["clf__sample_weight"] = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr, y_tr, **kwargs)
    return model


# ----------------------------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------------------------
def score(y, proba, pred):
    rec = recall_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    fatal = (y == 0).astype(int)
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "fatal_recall": rec[0],
        "fatal_precision": precision_score(y, pred, labels=[0], average=None, zero_division=0)[0],
        "fatal_f1": f1_score(y, pred, labels=[0], average=None, zero_division=0)[0],
        "fatal_pr_auc": average_precision_score(fatal, proba[:, 0]),
        "fatal_auc": roc_auc_score(fatal, proba[:, 0]),
        "fatal_vs_rest_accuracy": accuracy_score(fatal, (pred == 0).astype(int)),
    }


# ----------------------------------------------------------------------------------------------
# tuning
# ----------------------------------------------------------------------------------------------
def tune_params(name, X_tr, y_tr, year_tr, categorical, numeric, n_random, tuning_rows=None, context=None, metric="pr_auc"):
    """Nested, time-aware random search inside the training years. Returns (best params, best inner score).

    metric: "pr_auc" (fatal-class average precision, the default used for the Thai runs) or "roc_auc" (fatal-versus-rest
    ROC-AUC, used for the motorcycle comparison with a benchmark that reports AUC)."""
    inner_year = year_tr.max()
    inner_va, inner_tr = year_tr == inner_year, year_tr < inner_year
    truth = (y_tr[inner_va] == 0).astype(int)
    best, best_ap = None, -1.0
    for i, cfg in enumerate(sample_configs(name, n_random)):
        t0 = time.time()
        model = make_model(name, categorical, numeric, cfg)
        model.fit(X_tr[inner_tr], y_tr[inner_tr])
        p_fatal = model.predict_proba(X_tr[inner_va])[:, 0]
        ap = float(roc_auc_score(truth, p_fatal) if metric == "roc_auc" else average_precision_score(truth, p_fatal))
        secs = time.time() - t0
        if ap > best_ap:
            best, best_ap = cfg, ap
        if tuning_rows is not None:
            tuning_rows.append({**(context or {}), "model": name, "inner_year": int(inner_year), "config_id": i,
                                "params": json.dumps(cfg, sort_keys=True), f"inner_{metric}": round(ap, 5), "seconds": round(secs, 1)})
        log(f"    tune {name} cfg {i:2d} inner {metric} {ap:.4f} ({secs:.0f}s)")
    return best, best_ap


# ----------------------------------------------------------------------------------------------
# rolling-origin cross-validation
# ----------------------------------------------------------------------------------------------
def append_rows(path, rows):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def run_cv(X, y, year, folds, model_names, variants, categorical, numeric, out_path, label_suffix="",
           tune_names=(), n_random=9, tuning_path=None, dataset="", tune_metric="pr_auc"):
    """For every fold, fit each model under each variant on earlier years and score the next year.

    Rows are appended to out_path after every fold, so a stopped run keeps what it finished.
    Models in tune_names are tuned inside the training years first and all their variants share the tuned settings.
    """
    for last_train, val_year in folds:
        tr, va = year <= last_train, year == val_year
        cache, rows, tuning_rows = {}, [], []
        for name in model_names:
            params, inner_ap = {}, None
            if name in tune_names:
                t0 = time.time()
                params, inner_ap = tune_params(name, X[tr], y[tr], year[tr], categorical, numeric, n_random,
                                               tuning_rows, {"dataset": dataset, "val_year": val_year}, tune_metric)
                log(f"  {name} tuned in {time.time() - t0:.0f}s: {json.dumps(params, sort_keys=True)} (inner {tune_metric} {inner_ap:.4f})")
            probas = {}
            for variant in variants:
                if variant == "prior_adjusted":
                    continue
                if name == "majority_class" and variant != "default":
                    continue
                t0 = time.time()
                model = fit_variant(name, variant, X[tr], y[tr], categorical, numeric, params, cache, f"fold{val_year}")
                proba = model.predict_proba(X[va])
                probas[variant] = proba
                secs = time.time() - t0
                rows.append(_row(name + label_suffix, variant, val_year, tr, va, y, proba, proba.argmax(axis=1), params, inner_ap, secs, tune_metric))
                log(f"  {dataset} val {val_year} {name + label_suffix:22s} {variant:14s} fatal F1 {rows[-1]['fatal_f1']:.3f} "
                    f"recall {rows[-1]['fatal_recall']:.3f} PR-AUC {rows[-1]['fatal_pr_auc']:.3f} ({secs:.0f}s)")
            if "prior_adjusted" in variants and "default" in probas and name != "majority_class":
                prior = np.bincount(y[tr], minlength=3) / tr.sum()
                pred = (probas["default"] / prior).argmax(axis=1)
                rows.append(_row(name + label_suffix, "prior_adjusted", val_year, tr, va, y, probas["default"], pred, params, inner_ap, 0.0, tune_metric))
        append_rows(out_path, rows)
        if tuning_rows and tuning_path:
            append_rows(tuning_path, tuning_rows)
        cache.clear()


def _row(model, variant, val_year, tr, va, y, proba, pred, params, inner_ap, secs, metric="pr_auc"):
    row = {"model": model, "variant": variant, "val_year": int(val_year), "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "fatal_share_val": float((y[va] == 0).mean())}
    row.update(score(y[va], proba, pred))
    row["params"] = json.dumps(params, sort_keys=True)
    row[f"inner_{metric}"] = None if inner_ap is None else round(inner_ap, 5)
    row["seconds"] = round(secs, 1)
    return row
