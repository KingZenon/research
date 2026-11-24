"""Figure data for the project website: confusion matrices, precision-recall curves, permutation
importance and a province map of fatal share.

Usage:
    python -I -X utf8 make_site_figures.py <combined_csv> <geojson> <results_dir> <site_dir>

Writes
    <site_dir>/figures-data.js   window.FIGURES_DATA  (confusion matrices, PR curves, importance)
    <site_dir>/map-data.js       window.MAP_DATA      (simplified province outlines + counts)
    <results_dir>/confusion_matrices_2025.csv, pr_curves_fatal_2025.csv,
    <results_dir>/permutation_importance_2025.csv, province_fatal_share.csv

Rules kept from the experiments:
  * model figures use the last temporal fold only: train 2019-2024, validate 2025, with the same
    features, settings and seed as the cross-validation (the scores are checked against
    cv_temporal_thai_folds.csv before anything is written);
  * the 2026 test year is never scored or predicted here;
  * nothing on the website is typed by hand: the page only draws what these files contain.

Province outlines: geoBoundaries gbOpen THA ADM1 (simplified), derived from OpenStreetMap
(c) OpenStreetMap contributors, Open Database License 1.0. If <geojson> does not exist it is
downloaded from the pinned release below. Thai province labels in the accident table are matched
to outlines by where their incidents are (median GPS point), then checked to be one-to-one.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, shape
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve

import baseline_models as bm
import experiments_thai as ex

GEOJSON_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/THA/ADM1/"
               "geoBoundaries-THA-ADM1_simplified.geojson")
MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]
VARIANTS = ["default", "class_weight", "prior_adjusted"]
LABELS = {"logistic_regression": "Logistic regression", "decision_tree": "Decision tree",
          "random_forest": "Random forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
# columns shuffled together in the importance chart (near-duplicates or two views of one thing)
IMPORTANCE_GROUPS = [
    ("First vehicle type", ["vehicle1"]),
    ("Vehicle and person counts", ["n_vehicles", "n_involved"]),
    ("Accident type", ["acc_type"]),
    ("GPS position", ["lat", "lon"]),
    ("Province", ["province"]),
    ("Kilometre marker", ["km"]),
    ("Hour and night flag", ["hour", "is_night"]),
    ("Month", ["month"]),
    ("Weekday and weekend flag", ["weekday", "is_weekend"]),
    ("Responsible agency", ["agency"]),
    ("Presumed cause", ["cause"]),
    ("Road location", ["location"]),
    ("Weather", ["weather"]),
]
IMPORTANCE_REPEATS = 5
LAST_TRAIN, VAL_YEAR = 2024, 2025
IMPORTANCE_MODEL, IMPORTANCE_VARIANT = "xgboost", "class_weight"
RECALL_GRID = np.round(np.arange(0.02, 1.0001, 0.02), 2)


def operating_point(y_fatal, pred):
    """(recall, precision) of the fatal class for hard predictions."""
    is_pred = pred == 0
    tp = int((is_pred & (y_fatal == 1)).sum())
    return [tp / max(1, int((y_fatal == 1).sum())), tp / max(1, int(is_pred.sum()))]


def pr_points(y_fatal, p_fatal):
    """Precision at fixed recall levels (the highest threshold that reaches each recall)."""
    precision, recall, _ = precision_recall_curve(y_fatal, p_fatal)
    idx = np.clip(np.searchsorted(-recall, -RECALL_GRID, side="right") - 1, 0, len(recall) - 1)
    return [round(float(v), 4) for v in precision[idx]]


def model_figures(df, results_dir):
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    tr, va = year <= LAST_TRAIN, year == VAL_YEAR
    y_va = y[va]
    y_fatal = (y_va == 0).astype(int)
    reference = pd.read_csv(results_dir / "cv_temporal_thai_folds.csv")
    reference = reference[reference.val_year == VAL_YEAR].set_index(["model", "variant"])

    out, cm_rows, pr_rows, fitted = [], [], [], {}
    for name in MODELS:
        entry = {"model": name, "label": LABELS[name], "confusion": {}, "pr": {}, "operating": {}}
        probas = {}
        for variant in ("default", "class_weight"):
            t0 = time.time()
            model = ex.fit_variant(name, variant, X[tr], y[tr])
            probas[variant] = model.predict_proba(X[va])
            fitted[(name, variant)] = model
            print(f"  {name:20s} {variant:13s} {time.time() - t0:5.1f}s", flush=True)
        prior = np.bincount(y[tr], minlength=3) / tr.sum()
        preds = {"default": probas["default"].argmax(axis=1),
                 "class_weight": probas["class_weight"].argmax(axis=1),
                 "prior_adjusted": (probas["default"] / prior).argmax(axis=1)}
        for variant in VARIANTS:
            pred = preds[variant]
            # reproducibility check against the cross-validation fold that used the same split
            ref = reference.loc[(name, variant)]
            macro = f1_score(y_va, pred, average="macro", zero_division=0)
            fatal_f1 = f1_score(y_va, pred, labels=[0], average=None, zero_division=0)[0]
            if abs(macro - ref["macro_f1"]) > 2e-3 or abs(fatal_f1 - ref["fatal_f1"]) > 2e-3:
                raise SystemExit(f"{name}/{variant}: macro-F1 {macro:.4f} vs fold file {ref['macro_f1']:.4f}, "
                                 f"fatal F1 {fatal_f1:.4f} vs {ref['fatal_f1']:.4f}: split or settings changed")
            cm = confusion_matrix(y_va, pred, labels=[0, 1, 2])
            entry["confusion"][variant] = cm.tolist()
            entry["operating"][variant] = [round(v, 4) for v in operating_point(y_fatal, pred)]
            for a, actual in enumerate(bm.CLASSES):
                for p, predicted in enumerate(bm.CLASSES):
                    cm_rows.append({"model": name, "variant": variant, "actual": actual,
                                    "predicted": predicted, "count": int(cm[a, p])})
        for variant in ("default", "class_weight"):
            p_fatal = probas[variant][:, 0]
            entry["pr"][variant] = {"ap": round(float(average_precision_score(y_fatal, p_fatal)), 4),
                                    "precision": pr_points(y_fatal, p_fatal)}
            for r, pr in zip(RECALL_GRID, entry["pr"][variant]["precision"]):
                pr_rows.append({"model": name, "variant": variant, "recall": float(r), "precision": pr})
        out.append(entry)
    pd.DataFrame(cm_rows).to_csv(results_dir / "confusion_matrices_2025.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(results_dir / "pr_curves_fatal_2025.csv", index=False)

    # permutation importance for the model that has the best fatal F1 in the cross-validation.
    # Columns that carry almost the same information are shuffled together (one shuffle, same row order),
    # otherwise shuffling one of a near-duplicate pair creates rows that cannot occur and overstates it.
    model = fitted[(IMPORTANCE_MODEL, IMPORTANCE_VARIANT)]
    rng = np.random.default_rng(bm.SEED)
    X_va = X[va]
    base = average_precision_score(y_fatal, model.predict_proba(X_va)[:, 0])
    t0 = time.time()
    features = []
    for label, cols in IMPORTANCE_GROUPS:
        drops = []
        for _ in range(IMPORTANCE_REPEATS):
            order = rng.permutation(len(X_va))
            shuffled = X_va.copy()
            for c in cols:
                shuffled[c] = X_va[c].to_numpy()[order]
            drops.append(base - average_precision_score(y_fatal, model.predict_proba(shuffled)[:, 0]))
        features.append({"name": "+".join(cols), "label": label, "mean": round(float(np.mean(drops)), 5),
                         "std": round(float(np.std(drops, ddof=1)), 5)})
    features.sort(key=lambda d: -d["mean"])
    print(f"  grouped permutation importance {time.time() - t0:.1f}s", flush=True)
    pd.DataFrame(features).to_csv(results_dir / "permutation_importance_2025.csv", index=False)

    # check on the 'vehicles and persons involved' column: how it relates to the vehicle count and the target,
    # and whether the class-weighted XGBoost scores change when the column is emptied
    n_inv = pd.to_numeric(df["n_involved"], errors="coerce")
    n_veh = pd.to_numeric(df["n_vehicles"], errors="coerce")
    both = n_inv.notna() & n_veh.notna()
    extra = both & (n_inv > n_veh)
    fatal_all = df["severity"] == "fatal"
    emptied = X.copy()
    emptied["n_involved"] = np.nan
    model_without = ex.fit_variant(IMPORTANCE_MODEL, IMPORTANCE_VARIANT, emptied[tr], y[tr])
    proba_without = model_without.predict_proba(emptied[va])
    pred_with = model.predict_proba(X_va).argmax(axis=1)
    pred_without = proba_without.argmax(axis=1)
    check = {
        "column": "Vehicles and persons involved",
        "equalShare": round(float((n_inv == n_veh)[both].mean()), 5),
        "extraRows": int(extra.sum()),
        "extraFatalShare": round(float(fatal_all[extra].mean()), 4),
        "otherFatalShare": round(float(fatal_all[both & ~extra].mean()), 4),
        "with": {"fatalF1": round(float(f1_score(y_va, pred_with, labels=[0], average=None)[0]), 4), "prAuc": round(float(base), 4)},
        "without": {"fatalF1": round(float(f1_score(y_va, pred_without, labels=[0], average=None)[0]), 4),
                    "prAuc": round(float(average_precision_score(y_fatal, proba_without[:, 0])), 4)},
    }

    payload = {
        "source": "Generated by analysis/code/make_site_figures.py. Fold: train 2019-2024, validate 2025 "
                  "(the last temporal fold). The 2026 test year is not used.",
        "validation": {"train": f"2019–{LAST_TRAIN}", "year": VAL_YEAR, "nTrain": int(tr.sum()),
                       "nValidation": int(va.sum()), "fatalShare": round(float(y_fatal.mean()), 6)},
        "classes": ["Fatal", "Injury", "Property damage only"],
        "recallGrid": [float(r) for r in RECALL_GRID],
        "models": out,
        "importance": {"model": IMPORTANCE_MODEL, "label": LABELS[IMPORTANCE_MODEL], "variant": IMPORTANCE_VARIANT,
                       "metric": "fatal-class PR-AUC", "baseline": round(float(base), 5), "repeats": IMPORTANCE_REPEATS,
                       "features": features, "check": check},
    }
    text = ("/* Generated file: do not edit by hand. Run analysis/code/make_site_figures.py to rebuild. */\n"
            "window.FIGURES_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    return text


def province_map(df, geojson_path, results_dir):
    path = Path(geojson_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", GEOJSON_URL)
        urllib.request.urlretrieve(GEOJSON_URL, path)
    shapes = json.loads(path.read_text(encoding="utf-8"))["features"]
    polys = [(f["properties"]["shapeISO"], f["properties"]["shapeName"].replace(" Province", ""),
              shape(f["geometry"])) for f in shapes]
    assert len(polys) == 77, len(polys)

    d = df.copy()
    d["prov"] = d["province"].astype("string").str.replace(r"\s+", "", regex=True)
    d = d[d["prov"].notna()]
    for c in ("lat", "lon"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ok = d["lat"].between(5, 21) & d["lon"].between(97, 106)

    # match each Thai label to the outline that contains its median GPS point
    mapping = {}
    for prov, g in d[ok].groupby("prov"):
        pt = Point(g["lon"].median(), g["lat"].median())
        hit = [iso for iso, _, poly in polys if poly.contains(pt)]
        if not hit:   # median can fall just offshore: take the nearest outline
            hit = [min(polys, key=lambda t: t[2].distance(pt))[0]]
        mapping[prov] = hit[0]
    assert len(set(mapping.values())) == len(mapping) == 77, (len(mapping), len(set(mapping.values())))
    print("matched", len(mapping), "province labels to 77 outlines (one-to-one)")

    thai = {v: k for k, v in mapping.items()}
    counts = d.groupby("prov")["severity"].agg(incidents="size", fatal=lambda s: int((s == "fatal").sum()))
    rows = [{"iso": iso, "name": name, "thai": thai[iso], "incidents": int(counts.loc[thai[iso], "incidents"]),
             "fatal": int(counts.loc[thai[iso], "fatal"])} for iso, name, _ in polys]
    table = pd.DataFrame(rows)
    table["fatal_share"] = (table["fatal"] / table["incidents"]).round(5)
    z = 1.96   # Wilson 95% interval, shown so small provinces are not over-read
    p, n = table["fatal_share"], table["incidents"]
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    table["ci_low"], table["ci_high"] = (centre - half).round(5), (centre + half).round(5)
    table.sort_values("fatal_share", ascending=False).to_csv(results_dir / "province_fatal_share.csv", index=False,
                                                             encoding="utf-8-sig")

    # simplified outlines projected to a flat 500-unit-wide drawing
    minx, miny, maxx, maxy = 97.0, 5.5, 106.0, 20.6
    k = 500 / ((maxx - minx) * np.cos(np.radians((miny + maxy) / 2)))
    height = round((maxy - miny) * k, 1)

    def ring_path(coords):
        pts = [(round((x - minx) * np.cos(np.radians((miny + maxy) / 2)) * k, 1), round((maxy - y) * k, 1)) for x, y in coords]
        return "M" + "L".join(f"{x:g} {y:g}" for x, y in pts) + "Z"

    provinces = []
    for iso, name, poly in polys:
        simple = poly.simplify(0.012, preserve_topology=True)
        parts = list(simple.geoms) if isinstance(simple, MultiPolygon) else [simple]
        d_parts = []
        for part in parts:
            if part.area < 8e-4:
                continue
            d_parts.append(ring_path(part.exterior.coords))
            d_parts.extend(ring_path(r.coords) for r in part.interiors)
        row = table[table.iso == iso].iloc[0]
        provinces.append({"id": iso, "name": name, "thai": row["thai"], "d": "".join(d_parts),
                          "incidents": int(row["incidents"]), "fatal": int(row["fatal"])})
    payload = {
        "source": "Generated by analysis/code/make_site_figures.py from the harmonised accident table "
                  "(all years, 2019 to 12 Oct 2026) and simplified province outlines.",
        "attribution": "Province outlines: geoBoundaries (gbOpen, THA ADM1, simplified), from OpenStreetMap, "
                       "© OpenStreetMap contributors, Open Database License 1.0.",
        "width": 500, "height": height,
        "total": int(table["incidents"].sum()), "fatalTotal": int(table["fatal"].sum()),
        "provinces": provinces,
    }
    text = ("/* Generated file: do not edit by hand. Run analysis/code/make_site_figures.py to rebuild. */\n"
            "window.MAP_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    unlabelled = int(df["province"].isna().sum())
    print(f"map: {len(provinces)} provinces, {payload['total']:,} incidents with a province "
          f"({unlabelled:,} without one are left out), {len(text) / 1024:.0f} KB")
    return text


def main(combined_csv, geojson_path, results_dir, site_dir):
    results_dir, site_dir = Path(results_dir), Path(site_dir)
    df = bm.load(combined_csv, "all")
    print("model figures (train 2019-2024, validate 2025)")
    figures_js = model_figures(df, results_dir)
    (site_dir / "figures-data.js").write_text(figures_js, encoding="utf-8")
    print("wrote", site_dir / "figures-data.js", f"{len(figures_js) / 1024:.0f} KB")

    # the map describes every dated incident, so it reads the table without the modelling de-duplication
    raw = pd.read_csv(combined_csv, encoding="utf-8-sig", low_memory=False,
                      usecols=["province", "lat", "lon", "severity", "acc_year"]).dropna(subset=["acc_year", "severity"])
    raw = raw[raw["severity"].isin(bm.CLASSES)]
    map_js = province_map(raw, geojson_path, results_dir)
    (site_dir / "map-data.js").write_text(map_js, encoding="utf-8")
    print("wrote", site_dir / "map-data.js")


if __name__ == "__main__":
    main(*sys.argv[1:5])
