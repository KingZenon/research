# Which Thai Road Crashes Turn Fatal? — Team AHT (ICT324)

- **Public project website:** https://kingzenon.github.io/research/
- **Repository:** https://github.com/KingZenon/research
- **Course:** ICT324 Data Mining and Data Warehouse, Rangsit University International College (2026)
- **Team AHT:** Aung Ko Ko Minn · Htoo Myat Htin Lynn · Thant Mg Mg Lwin

## Overview

This repository hosts the static project website, analysis code, locked experiment outputs, and final deliverables for Team AHT's road accident severity classification study on Thailand's Ministry of Transport (TRAMS) road network (168,768 deduplicated incidents, 2019–2026), replicated on UK DfT STATS19 collisions (412,276 collisions, 2021–2024).

## Repository Layout

| Path | Contents |
| --- | --- |
| `index.html`, `styles.css`, `script.js` | Static project website (no build step or external CSS library required). |
| `project-data.js` | Research narrative, dataset audit, progress log, file links, and 21 references. |
| `results-data.js`, `final-data.js`, `figures-data.js`, `map-data.js` | Generated data files for site tables, charts, confusion matrices, and the Thailand province map. |
| `assets/deliverables/` | Final IEEE paper (`Team_AHT_ICT324_Final_Paper.pdf`) and final presentation slides (`Team_AHT_ICT324_Final_Slides.pdf`). |
| `assets/` | Public project proposal (`Project-Proposal-Public.pdf`, student IDs removed), kickoff slides, and course brief. |
| `analysis/` | Python analysis scripts (`analysis/code/`), reproduction guide (`analysis/README.md`), dataset audit and download links (`analysis/DATA.md`, `analysis/data-manifest.csv`), and locked results (`analysis/results/`, `analysis/results_recheck/`). |

Raw TRAMS and UK STATS19 CSV files are not stored in this repository; direct official download links and SHA-256 verification hashes are documented in [analysis/DATA.md](analysis/DATA.md) and [analysis/data-manifest.csv](analysis/data-manifest.csv).

## Preview & Verification

Open `index.html` directly, or serve locally:

```bash
python -m http.server 8000
```

Validate site scripts and configured local links:

```bash
node --check script.js
node --check project-data.js
node check-site.js
```

## Reproducing the Experiments

See [analysis/README.md](analysis/README.md) for full environment setup and step-by-step commands to reproduce the temporal cross-validation, SMOTE-NC and nested tuning runs, UK STATS19 replication, motorcycle subset, single-shot 2026 test evaluation, post hoc robustness re-check, and generated site/paper outputs.

## Privacy Note

In accordance with course guidelines, this public repository and website contain no private or personal data (no student IDs or email addresses).
