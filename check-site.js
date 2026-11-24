const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = __dirname;
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const context = { window: {} };
vm.runInNewContext(fs.readFileSync(path.join(root, "project-data.js"), "utf8"), context);

const project = context.window.PROJECT_DATA;
const problems = [];
const localPaths = [
  "styles.css",
  "script.js",
  "project-data.js",
  "results-data.js",
  "figures-data.js",
  "map-data.js",
  "final-data.js",
  ...project.files.map(item => item.path).filter(Boolean),
  ...project.results.figures.map(item => item.path)
];

for (const item of localPaths) {
  if (/^https?:\/\//.test(item)) continue;
  if (path.isAbsolute(item) || item.includes("..")) {
    problems.push(`Unsafe local path: ${item}`);
  } else if (!fs.existsSync(path.join(root, item))) {
    problems.push(`Missing local file: ${item}`);
  }
}

const requiredIds = [
  "last-updated", "data-status", "data-facts", "cleaning-steps", "related-work", "older-entries",
  "year-rows", "column-rows", "progress-entries", "results-status",
  "results-summary", "model-rows", "result-figures",
  "results-takeaways", "results-limits", "file-list", "reference-list",
  "team-names", "team-members", "hero-stats", "milestones",
  "severity-bar", "severity-legend", "chart-fatal-share", "fold-rows",
  "result-stats", "variant-legend", "model-table-note", "draft-status", "tooltip",
  "map-card", "province-map", "map-metric", "map-legend", "map-rank",
  "error-card", "cm-models", "cm-grid", "chart-pr", "chart-importance",
  "final-card", "final-stats", "final-rows", "final-cm", "selection-card", "selection-rows",
  "strategy-cards", "tuning-rows", "smote-rows", "uk-card", "uk-rows", "moto-card", "moto-rows",
  "recheck-card", "recheck-rows", "recheck-sub-rows"
];
for (const id of requiredIds) {
  if (!html.includes(`id="${id}"`)) problems.push(`Missing page target: ${id}`);
}

if (problems.length) {
  console.error(problems.join("\n"));
  process.exitCode = 1;
} else {
  console.log("Site structure and configured local file paths are valid.");
}
