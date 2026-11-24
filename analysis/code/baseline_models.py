"""Baseline models for three-class road-accident severity (fatal / injury / property_only).

Usage:
    python -I -X utf8 baseline_models.py <combined_csv> <results_dir> [--subset motorcycle] [--final-test]

Design (follows Proposal B):
  * temporal split: train 2019-2024, validation 2025, test 2026 (partial year)
  * the 2026 test set is NOT scored unless --final-test is given, so it stays untouched
    while models and features are still being chosen
  * features are only what is known when the accident is reported; casualty columns
    (deaths, serious, minor, injured_total) define the target and are excluded (leakage)
  * fixed, untuned hyper-parameters, no imbalance handling yet -> these are the baselines that
    later experiments (class weights, SMOTE, threshold tuning, tuning) must beat
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
CLASSES = ["fatal", "injury", "property_only"]  # label order used everywhere (0, 1, 2)
CATEGORICAL = ["province", "agency", "vehicle1", "location", "cause", "acc_type", "weather"]
NUMERIC = ["lat", "lon", "km", "n_vehicles", "n_involved", "hour", "month", "weekday",
           "is_weekend", "is_night"]


def parse_hour(value):
    """Time of day appears as 'H:MM' text in most years and as a day fraction in 2024."""
    s = str(value).strip()
    if ":" in s:
        m = re.match(r"^(\d{1,2}):", s)
        return int(m.group(1)) if m else np.nan
    try:
        f = float(s)
    except ValueError:
        return np.nan
    return int(f * 24) if 0 <= f < 1 else np.nan


def build_features(df):
    """Turn the harmonised table into model features (no casualty columns)."""
    out = pd.DataFrame(index=df.index)
    out["hour"] = df["time"].map(parse_hour)
    when = pd.to_datetime(df["date_parsed"])
    out["month"] = when.dt.month
    out["weekday"] = when.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(float)
    out["is_night"] = ((out["hour"] < 6) | (out["hour"] >= 18)).astype(float)
    out.loc[out["hour"].isna(), "is_night"] = np.nan
    for col in ("lat", "lon", "km", "n_vehicles", "n_involved"):
        out[col] = pd.to_numeric(df[col], errors="coerce")
    # coordinates outside Thailand are data-entry errors
    bad = ~out["lat"].between(5, 21) | ~out["lon"].between(97, 106)
    out.loc[bad, ["lat", "lon"]] = np.nan
    for col in CATEGORICAL:
        # category labels differ in spacing between yearly files; normalise them
        out[col] = df[col].astype("string").str.replace(r"\s+", "", regex=True).fillna("missing")
    return out[CATEGORICAL + NUMERIC]


def load(path, subset):
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = df[df["severity"].isin(CLASSES)].copy()
    # 2023 repeats some accident codes; drop rows that are identical on the key fields
    key = ["acc_code", "date_parsed", "time", "province", "deaths", "injured_total"]
    has_code = df["acc_code"].notna()
    dup = has_code & df.duplicated(subset=key, keep="first")
    print(f"rows {len(df):,}; dropped {int(dup.sum()):,} exact duplicate accident codes")
    df = df[~dup]
    if subset == "motorcycle":
        df = df[df["vehicle1"].astype("string").str.contains("จักรยานยนต์", na=False)]
        print(f"motorcycle subset: {len(df):,} rows")
    return df.reset_index(drop=True)


def make_models(categorical=None, numeric=None):
    """The model zoo. The column lists default to the Thai table; the UK replication passes its own."""
    categorical = CATEGORICAL if categorical is None else categorical
    numeric = NUMERIC if numeric is None else numeric
    onehot = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), categorical),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), numeric),
    ])
    ordinal = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ("num", "passthrough", numeric),  # tree models handle missing values natively
    ])
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([("prep", onehot),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "decision_tree": Pipeline([("prep", ordinal),
                                   ("clf", DecisionTreeClassifier(min_samples_leaf=20, random_state=SEED))]),
        "random_forest": Pipeline([("prep", ordinal),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                                  n_jobs=-1, random_state=SEED))]),
        "xgboost": Pipeline([("prep", ordinal),
                             ("clf", XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6,
                                                   tree_method="hist", n_jobs=-1, random_state=SEED))]),
        "lightgbm": Pipeline([("prep", ordinal),
                              ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                                     n_jobs=-1, random_state=SEED, verbose=-1))]),
    }


def evaluate(y_true, proba):
    """Three-class metrics plus the fatal-class view used to compare with earlier studies."""
    pred = proba.argmax(axis=1)
    rec = recall_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "recall_fatal": rec[0], "recall_injury": rec[1], "recall_property": rec[2],
        "precision_fatal": precision_score(y_true, pred, labels=[0], average=None, zero_division=0)[0],
        "f1_fatal": f1_score(y_true, pred, labels=[0], average=None, zero_division=0)[0],
        # PR-AUC of the fatal class (average precision of P(fatal) for fatal vs the rest)
        "pr_auc_fatal": average_precision_score((y_true == 0).astype(int), proba[:, 0]),
        "auc_ovr_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "auc_fatal_vs_rest": roc_auc_score((y_true == 0).astype(int), proba[:, 0]),
        "log_loss": log_loss(y_true, proba, labels=[0, 1, 2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--subset", choices=["all", "motorcycle"], default="all")
    ap.add_argument("--final-test", action="store_true", help="also score the 2026 test set")
    args = ap.parse_args()

    df = load(args.csv, args.subset)
    X = build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    train, val, test = year <= 2024, year == 2025, year == 2026
    print(f"train {train.sum():,} | validation(2025) {val.sum():,} | test(2026) {test.sum():,}")
    print("class share train:", dict(zip(CLASSES, np.bincount(y[train], minlength=3) / train.sum())))
    print("class share val:  ", dict(zip(CLASSES, np.bincount(y[val], minlength=3) / val.sum())))

    rows, fitted = [], {}
    for name, model in make_models().items():
        t0 = time.time()
        model.fit(X[train], y[train])
        proba = model.predict_proba(X[val])
        res = {"model": name, "split": "validation_2025", "fit_seconds": round(time.time() - t0, 1)}
        res.update(evaluate(y[val], proba))
        rows.append(res)
        fitted[name] = model
        print(f"  {name:20s} acc={res['accuracy']:.3f} bal={res['balanced_acc']:.3f} "
              f"macroF1={res['macro_f1']:.3f} fatalRecall={res['recall_fatal']:.3f} "
              f"({res['fit_seconds']}s)")

    if args.final_test:
        for name, model in fitted.items():
            res = {"model": name, "split": "test_2026", "fit_seconds": np.nan}
            res.update(evaluate(y[test], model.predict_proba(X[test])))
            rows.append(res)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    tag = args.subset
    table.to_csv(out / f"baseline_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + table.drop(columns=["split"]).round(3).to_string(index=False))

    best = table[table["split"] == "validation_2025"].sort_values("macro_f1").iloc[-1]["model"]
    cm = confusion_matrix(y[val], fitted[best].predict_proba(X[val]).argmax(axis=1), labels=[0, 1, 2])
    print(f"\nconfusion matrix on 2025 for best macro-F1 model ({best}); rows=actual, cols=predicted {CLASSES}")
    print(cm)


if __name__ == "__main__":
    main()
