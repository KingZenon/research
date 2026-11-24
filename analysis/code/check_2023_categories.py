"""How different are the 2023 category spellings from the other years?

Usage:  python -X utf8 check_2023_categories.py <combined_csv> <results_dir>

On the modelled incidents (after the 2023 de-duplication), for the three categorical columns that carry free-text-like labels (presumed cause, road location, accident type) this
counts, for each year, the share of incidents whose label never occurs in 2019-2022. The 2023 yearly file uses different
spellings, so most of its values are unseen when a model is trained on earlier years. Writes
results/category_recode_2023.json (used by the paper). The 2026 labels are only counted, never used for a model.
"""
import json
import sys
from pathlib import Path

import pandas as pd

import baseline_models as bm


def main(csv, results_dir):
    d = bm.load(csv, "all")          # the modelled incidents: the same de-duplication as every experiment
    d = d.dropna(subset=["acc_year"])
    out = {}
    for col in ("cause", "location", "acc_type"):
        s = d[col].astype("string").str.replace(r"\s+", "", regex=True)
        seen = set(s[d["acc_year"] <= 2022].dropna().unique())
        out[col] = {int(y): round(float((~s[d["acc_year"] == y].dropna().isin(seen)).mean()), 4) for y in range(2023, 2027)}
    path = Path(results_dir) / "category_recode_2023.json"
    path.write_text(json.dumps({"share_unseen_in_2019_2022": out}, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
