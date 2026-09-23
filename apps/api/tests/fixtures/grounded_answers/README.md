# Grounded-answer evaluation fixtures

This directory contains sanitized, fictional evaluation input for local grounded-answer quality work. It is not production application data and no fixture is loaded by the API at runtime.

`v1/manifest.json` is a human-reviewable JSON manifest with `schema_version: 1`. Its logical knowledge-base IDs are stable fixture labels, never database identifiers. Each document entry gives a committed, UTF-8 `.txt` fixture path and its intended processing status.

Each case has a stable ID, a `calibration` or `holdout` split, a category, a selected fixture knowledge base, a question, and an expected outcome. `answer` cases also name their expected Ready source and a canonical fact that occurs in that source. `no_answer` cases intentionally omit those fields. Isolation and unavailable-evidence cases include fixture-only metadata that identifies where matching evidence exists outside the selected knowledge base or in a non-Ready document.

The set contains 20 cases: eight direct-evidence answerable cases, four paraphrased-evidence answerable cases, four no-supporting-evidence cases, two knowledge-base-isolation cases, and two non-Ready/unavailable-evidence cases. The calibration and holdout groups each contain 10 cases.

When adding a case, use only fictional, sanitized technical text; add a unique stable case ID; keep all fixture paths within the version directory; and update the validation test if the schema changes. Do not change holdout cases to optimize a tuning result. A changed benchmark requires a new versioned manifest directory, such as `v2`, so existing results remain comparable.
