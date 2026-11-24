# ICT324 road accident severity website

Public site: https://kingzenon.github.io/research/  
Repository: https://github.com/KingZenon/Research

This is a static project website. Open `index.html` directly, or run `python -m http.server 8000` in this folder and visit `http://localhost:8000/`. No build step or online CSS library is required.

The data audit, dataset lock, related work, first model results (Milestone 3), analysis code and a public copy of the proposal are in place (see the progress log on the site). Values that are not measured yet stay visibly pending. **Do not enter estimated values.**

## Where to put new materials

| Material | Location | Page update |
| --- | --- | --- |
| Real TRAMS CSV/XLSX files for analysis | `incoming-data/` or `data/` (both ignored by git) | Record audited values in `project-data.js`. Do not publish the raw folder unless redistribution is allowed and privacy has been checked. Download links are in `analysis/DATA.md`. |
| Charts and figures | `assets/results/` | Add an entry to `results.figures` with title, path, alt text, and caption. |
| Final paper PDF | `assets/deliverables/` | Set the **Final paper** path in `files`. |
| Final slides PDF or PPTX | `assets/deliverables/` | Set the **Final presentation** path in `files`. |
| Analysis code and reproduction guide | `analysis/` | Added (`analysis/README.md`, `analysis/DATA.md`, `analysis/code/`, `analysis/results/`). Keep it in step with the experiments and link new result files from `files`. |
| Public dataset or access instructions | Publisher URL or an approved public file | Set the **Verified dataset or data access instructions** path in `files`. |

The kickoff slides, course brief and a public copy of the proposal (`assets/Project-Proposal-Public.pdf`, student ID numbers removed) are linked already. The original proposal PDF contains student ID numbers, so it stays out of the repository. Add files first, then update their paths in `project-data.js`. Paths are relative to `index.html`.

## Update the research content

All changeable project content lives in `project-data.js`:

To update the team section, edit the `team` list near the top of `project-data.js`. Change a member's `name` or `github` profile URL. The hero names and team cards update from the same list. Email addresses are deliberately not published, because the course rules say the page must contain no private or personal data. If you add an `email` field, the card will show it publicly.

1. Set `data.totalRecords`, `dateRange`, `targetDefinition`, `cleaningSummary`, and each audited year. Add the real Thai column names, English labels, and cleaning notes to `data.columns`.
2. Add a dated `progress` entry describing work completed, issues, and next action. Set `lastUpdated` to the newest actual update date.
3. Model scores are **not typed by hand**. Rerun the experiments, then run `analysis/code/export_site_results.py` to rebuild `results-data.js`; the results table, stat tiles and charts read that file. Write the summary, interpretation and first-draft status text in `results` in `project-data.js`.
3a. `analysis/code/export_site_results.py` now reads the combined summary (`cv_thai_all_summary.csv`), so the results table also lists the tuned models and the SMOTE-NC strategy. The final test result, the selection table, the tuning and SMOTE checks, the UK replication and the motorcycle subset come from `final-data.js`, built by `analysis/code/export_final_results.py` (step 10 in `analysis/README.md`).
3b. The confusion matrices, precision-recall chart, importance chart and province map are **not typed by hand either**. Run `analysis/code/make_site_figures.py` (step 5 in `analysis/README.md`) to rebuild `figures-data.js` and `map-data.js` and the four CSV files linked under Files. Rerun it whenever the features, settings or data change.
4. Update `highlights` (hero facts) and `milestones` (set each `state` to `done`, `current` or `upcoming`). Set `results.status` only when results are actually measured.
5. Add verified references with full citation details and URLs. The course requires at least 10 recent research references in the final paper. The dataset source is already listed.
6. Set each pending `files[].path` after placing the file. A `null` path stays visibly marked **Awaiting file**.

## Before publishing each milestone

- Run `node --check script.js` and `node --check project-data.js`.
- Run `node check-site.js` to check page targets and configured local file paths.
- Preview on desktop and phone sizes.
- Open the public URL in a signed-out browser to confirm it works without login.
- Submit the public URL on Google Classroom when ready. GitHub Pages is configured from the `main` branch at the repository root; Classroom submission remains a team action.

The website code is split by purpose: `index.html` contains page structure, `styles.css` contains the design, layout and responsive rules, `project-data.js` contains research text, `results-data.js` contains measured numbers (generated), and `script.js` renders the content, draws the charts and controls the mobile menu. When you change `styles.css`, `script.js` or the data files, change the `?v=` tag on their links in `index.html` so browsers load the new version.

