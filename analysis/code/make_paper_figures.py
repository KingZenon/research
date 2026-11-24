"""Figures and a numbers file for the final paper.

Usage:
    python -X utf8 make_paper_figures.py <combined_csv> <geojson> <results_dir> <paper_dir>

Writes <paper_dir>/figures/{fig_pr,fig_cm,fig_imp,fig_map}.png and <paper_dir>/paper_numbers.json.

* fig_pr, fig_imp: the selected model family (LightGBM with the settings tuned inside 2019-2024) fitted on 2019-2024
  and scored on the 2025 validation year, the same split as the last cross-validation fold. Nothing here touches 2026.
* fig_cm: the single 2026 test result written by final_test_2026.py (results/test_2026_confusion.json).
* fig_map: incidents and fatal share by province, from results/province_fatal_share.csv.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from shapely.geometry import MultiPolygon, shape
from sklearn.metrics import average_precision_score, precision_recall_curve

import baseline_models as bm
import cv_tools as cv
from make_site_figures import IMPORTANCE_GROUPS

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#555555"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5,
                     "axes.linewidth": 0.6, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.8, "figure.dpi": 150, "savefig.dpi": 300})


def operating_point(y_fatal, pred):
    hit = int(((pred == 0) & (y_fatal == 1)).sum())
    return [hit / max(1, int(y_fatal.sum())), hit / max(1, int((pred == 0).sum()))]


def model_figures(csv, results, fig_dir, numbers):
    tuned = pd.read_csv(results / "cv_thai_tuned_folds.csv")
    row = tuned[(tuned.model == "lightgbm_tuned") & (tuned.val_year == 2025)].iloc[0]
    params = json.loads(row["params"])
    numbers["lightgbm_fold4_params"] = params
    df = bm.load(csv, "all")
    df = df[df["acc_year"].notna()].reset_index(drop=True)
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    tr, va = year <= 2024, year == 2025
    cat, num = bm.CATEGORICAL, bm.NUMERIC
    yf = (y[va] == 0).astype(int)
    cache, probas, preds, models = {}, {}, {}, {}
    for variant in ("default", "class_weight", "smote"):
        m = cv.fit_variant("lightgbm", variant, X[tr], y[tr], cat, num, params, cache, "paper")
        models[variant] = m
        probas[variant] = m.predict_proba(X[va])
        preds[variant] = probas[variant].argmax(axis=1)
        cv.log("fitted", variant)
    prior = np.bincount(y[tr], minlength=3) / tr.sum()
    preds["prior_adjusted"] = (probas["default"] / prior).argmax(axis=1)
    numbers["pr"] = {"prevalence": float(yf.mean()),
                     "ap": {v: float(average_precision_score(yf, probas[v][:, 0])) for v in probas},
                     "operating": {v: operating_point(yf, preds[v]) for v in preds}}

    # ---- Fig. PR curves
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    styles = {"default": (BLUE, "-", "Plain"), "class_weight": (ORANGE, "--", "Class weights"), "smote": (GREEN, ":", "SMOTE-NC")}
    for v, (color, ls, label) in styles.items():
        p, r, _ = precision_recall_curve(yf, probas[v][:, 0])
        ax.plot(r, p, color=color, ls=ls, lw=1.3, label=f"{label} (AP {numbers['pr']['ap'][v]:.3f})")
    ax.axhline(yf.mean(), color=GREY, lw=0.7, ls="-.", label=f"Random ({yf.mean():.3f})")
    marks = {"default": ("o", BLUE), "class_weight": ("s", ORANGE), "prior_adjusted": ("D", "#444444"), "smote": ("^", GREEN)}
    names = {"default": "plain", "class_weight": "class weights", "prior_adjusted": "prior-adjusted", "smote": "SMOTE-NC"}
    for v, (mk, color) in marks.items():
        r_, p_ = numbers["pr"]["operating"][v]
        ax.scatter([r_], [p_], marker=mk, s=26, color=color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(names[v], (r_, p_), textcoords="offset points", xytext=(5, 4) if v not in ("prior_adjusted", "smote") else (5, -9), fontsize=6.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Recall (fatal)")
    ax.set_ylabel("Precision (fatal)")
    ax.legend(loc="upper right", frameon=False, handlelength=2.2)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(fig_dir / "fig_pr.png")
    plt.close(fig)

    # ---- Fig. grouped permutation importance (class-weighted LightGBM, 2025 validation)
    model = models["class_weight"]
    X_va = X[va]
    rng = np.random.default_rng(bm.SEED)
    base = average_precision_score(yf, model.predict_proba(X_va)[:, 0])
    feats = []
    for label, cols in IMPORTANCE_GROUPS:
        drops = []
        for _ in range(5):
            order = rng.permutation(len(X_va))
            shuffled = X_va.copy()
            for c in cols:
                shuffled[c] = X_va[c].to_numpy()[order]
            drops.append(base - average_precision_score(yf, model.predict_proba(shuffled)[:, 0]))
        feats.append({"label": label, "mean": float(np.mean(drops)), "std": float(np.std(drops, ddof=1))})
    feats.sort(key=lambda d: -d["mean"])
    numbers["importance"] = {"baseline_ap": float(base), "features": feats}
    top = feats[:8][::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.barh([f["label"] for f in top], [f["mean"] for f in top], xerr=[f["std"] for f in top], color=BLUE, height=0.62,
            error_kw={"lw": 0.7, "capsize": 1.5})
    ax.set_xlabel("Drop in fatal-class PR-AUC when shuffled")
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(fig_dir / "fig_imp.png")
    plt.close(fig)


def confusion_figure(results, fig_dir, numbers):
    data = json.loads((results / "test_2026_confusion.json").read_text(encoding="utf-8"))
    m = np.array(data["matrix"], dtype=float)
    share = m / m.sum(axis=1, keepdims=True)
    numbers["test_confusion"] = {"matrix": data["matrix"]}
    names = ["Fatal", "Injury", "Damage only"]
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{share[i, j] * 100:.0f}%\n({int(m[i, j]):,})", ha="center", va="center", fontsize=6.6,
                    color="white" if share[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(fig_dir / "fig_cm.png")
    plt.close(fig)


def map_figure(geojson_path, results, fig_dir, numbers):
    table = pd.read_csv(results / "province_fatal_share.csv", encoding="utf-8-sig")
    share = dict(zip(table["iso"], table["fatal_share"]))
    features = json.loads(Path(geojson_path).read_text(encoding="utf-8"))["features"]
    patches, values = [], []
    for f in features:
        iso = f["properties"]["shapeISO"]
        geom = shape(f["geometry"])
        parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for part in parts:
            patches.append(Polygon(np.array(part.exterior.coords), closed=True))
            values.append(share[iso])
    fig, ax = plt.subplots(figsize=(2.7, 3.75))
    pc = PatchCollection(patches, cmap="Blues", edgecolor="white", linewidth=0.25)
    pc.set_array(np.array(values))
    pc.set_clim(0, max(values))
    ax.add_collection(pc)
    ax.set_xlim(97.2, 105.8)
    ax.set_ylim(5.5, 20.6)
    ax.set_aspect(1 / np.cos(np.radians(13)))
    ax.axis("off")
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.01, shrink=0.85)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("Fatal share of incidents", fontsize=6.5)
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.07)
    fig.savefig(fig_dir / "fig_map.png")
    plt.close(fig)
    big = table[table.incidents >= 1000].sort_values("fatal_share", ascending=False)
    numbers["provinces"] = {"n": int(len(table)), "total_incidents": int(table.incidents.sum()),
                            "top": big.head(5)[["name", "incidents", "fatal_share"]].to_dict("records"),
                            "bottom": big.tail(3)[["name", "incidents", "fatal_share"]].to_dict("records"),
                            "over_20": int((table.fatal_share >= 0.20).sum())}


def main(csv, geojson, results_dir, paper_dir):
    results, paper = Path(results_dir), Path(paper_dir)
    fig_dir = paper / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    numbers = {}
    raw = pd.read_csv(csv, encoding="utf-8-sig", low_memory=False, usecols=["acc_year", "severity"]).dropna()
    by_year = pd.crosstab(raw["acc_year"].astype(int), raw["severity"])
    numbers["fatal_share_by_year"] = {int(y): float(r["fatal"] / r.sum()) for y, r in by_year.iterrows()}
    numbers["incidents_by_year"] = {int(y): int(r.sum()) for y, r in by_year.iterrows()}
    numbers["class_totals"] = {k: int(v) for k, v in by_year.sum().items()}
    confusion_figure(results, fig_dir, numbers)
    map_figure(geojson, results, fig_dir, numbers)
    model_figures(csv, results, fig_dir, numbers)
    (paper / "paper_numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    cv.log("wrote figures and paper_numbers.json")


if __name__ == "__main__":
    main(*sys.argv[1:5])
