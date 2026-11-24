"""Replicate the Thai pipeline on UK STATS19 collisions (2021-2024), as a second-country check.

Usage:
    python -X utf8 uk_replication.py <stats19_dir> <results_dir> [--sample 0.05]

The same model families, the same four imbalance variants and the same rolling-origin idea as the Thai run:
    train 2021 -> validate 2022 | 2021-2022 -> 2023 | 2021-2023 -> 2024
Nothing is tuned on UK data: every setting is the fixed Milestone 3 setting from baseline_models.py.

Classes (STATS19 collision_severity): 1 fatal -> 0, 2 serious -> 1, 3 slight -> 2. These are not the same
definitions as the Thai classes (fatal / injury / property damage only): the UK has no damage-only records, and
"serious" is a subset of injury. The fatal class is comparable (at least one death), so the comparison between
countries is made on fatal-class metrics, not on three-class accuracy.

Features that exist before the outcome is known: place, time, road, junction, light, weather, surface, speed limit,
vehicle count. Left out because they describe the outcome or its handling: number_of_casualties, whether the police
attended, the enhanced/adjusted severity fields, and the LSOA code (about 35,000 values).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_models as bm
import cv_tools as cv

FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]
FAMILIES = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]
CATEGORICAL = ["police_force", "local_authority_ons_district", "first_road_class", "road_type", "junction_detail",
               "junction_control", "second_road_class", "pedestrian_crossing", "light_conditions", "weather_conditions",
               "road_surface_conditions", "special_conditions_at_site", "carriageway_hazards", "urban_or_rural_area",
               "trunk_road_flag"]
NUMERIC = ["lat", "lon", "n_vehicles", "speed_limit", "hour", "month", "weekday", "is_weekend", "is_night"]


def load_uk(folder):
    frames = [pd.read_csv(path, low_memory=False) for path in sorted(Path(folder).glob("collision_20*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["collision_severity"].isin([1, 2, 3])].reset_index(drop=True)
    y = df["collision_severity"].map({1: 0, 2: 1, 3: 2}).to_numpy()
    when = pd.to_datetime(df["date"], format="%d/%m/%Y")
    out = pd.DataFrame(index=df.index)
    out["hour"] = pd.to_numeric(df["time"].astype("string").str.slice(0, 2), errors="coerce")
    out["month"] = when.dt.month
    out["weekday"] = when.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(float)
    out["is_night"] = df["light_conditions"].isin([4, 5, 6, 7]).astype(float)   # darkness, lit or unlit
    out["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    out.loc[~out["lat"].between(49, 61) | ~out["lon"].between(-9, 2), ["lat", "lon"]] = np.nan
    out["n_vehicles"] = pd.to_numeric(df["number_of_vehicles"], errors="coerce")
    speed = pd.to_numeric(df["speed_limit"], errors="coerce")
    out["speed_limit"] = speed.where(speed > 0)                                  # -1 means unknown
    for col in CATEGORICAL:
        out[col] = df[col].astype("string").fillna("missing")
    return out[CATEGORICAL + NUMERIC], y, df["collision_year"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats19_dir")
    ap.add_argument("results_dir")
    ap.add_argument("--sample", type=float, default=1.0, help="random fraction of rows, for smoke tests only")
    args = ap.parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, year = load_uk(args.stats19_dir)
    audit = {"rows": int(len(y)), "years": {int(k): int(v) for k, v in zip(*np.unique(year, return_counts=True))},
             "classShare": {n: round(float(np.mean(y == i)), 5) for i, n in enumerate(["fatal", "serious", "slight"])},
             "fatalShareByYear": {int(yr): round(float(np.mean(y[year == yr] == 0)), 5) for yr in np.unique(year)},
             "categorical": CATEGORICAL, "numeric": NUMERIC,
             "missing": {c: round(float(X[c].isna().mean()), 5) for c in NUMERIC}}
    if args.sample < 1.0:
        keep = np.random.default_rng(cv.SEED).random(len(y)) < args.sample
        X, y, year = X[keep].reset_index(drop=True), y[keep], year[keep]
    else:
        (out_dir / "uk_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    cv.log(f"UK STATS19: {len(y):,} collisions, fatal {np.mean(y == 0):.4f}, serious {np.mean(y == 1):.4f}, slight {np.mean(y == 2):.4f}")
    cv.run_cv(X, y, year, FOLDS, ["majority_class"] + FAMILIES, cv.VARIANTS, CATEGORICAL, NUMERIC,
              out_dir / "cv_uk_folds.csv", dataset="uk")
    cv.log("done")


if __name__ == "__main__":
    main()
