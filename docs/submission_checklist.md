# PathSnap submission checklist

Status reflects the local project as of 18 November 2026. The report requirement is one consolidated
technical report per team, not one report per model.

## Required submission package

| Deliverable | Required content | Local evidence | Status |
|---|---|---|---|
| Public GitHub repository | Complete source, configs, reproducible data processing, feature generation, model training/comparison/evaluation, HPO, HMM/Viterbi, API/UI, visualizations and results; exclude raw data and secrets | Repository source, `configs/`, `scripts/`, `src/pathsnap/`, `ui/`, `outputs/`, `artifacts/final/` | Present locally; push awaits user approval |
| One team technical report PDF | Required report sections; actual results, comparison tables, plots and geographical visualizations; each claim verifiable in GitHub | `reports/PathSnap_Technical_Report.pdf` and `scripts/build_technical_report.py` | Prepared locally; add to GitHub after review |
| EDA evidence | Inventory, missing/invalid checks, intervals, trajectories, geographic extent, road network, route alignment, candidate diagnostics, high-miss analysis and training-only feature plots | `outputs/eda/`, `docs/data_quality.md`, `outputs/eda_operations.md` | Present |
| Interactive maps | Three Folium maps with toggleable GPS, network, reference route, legend and efficient geometry | `outputs/eda/maps/*.html`; generation code in `scripts/run_eda.py` | Present locally |
| Folium map screenshot/exported browser view | At least one genuine view of the interactive map in the report | The report has projected-metre map figures and the repository has interactive HTML, but no verified browser screenshot | Still needed |
| Model experiments | At least five models, group-aware CV, student-test comparison, metrics, configuration and timing | `outputs/experiments/model_comparison.csv`, `outputs/experiments/teammate_model_retraining/`, `scripts/run_model_suite.py` | Present; original and later suites are distinguished in report |
| Bayesian HPO | Search space, objective, trial records, best parameters and tuned/untuned comparison | `outputs/experiments/bayesian_trials.csv`, `tuning_result.json`, `tuned_vs_untuned_summary.csv` | Present for original HistGradientBoosting; active Random Forest was not HPO-tuned |
| HMM/Viterbi evaluation | State/emission/transition/gap behavior, classifier-only versus decoded results and failure analysis | `outputs/experiments/selected_decoder.json`, `outputs/experiments/teammate_model_retraining/hmm_student_test_summary.csv`, `outputs/final_evaluation/` | Present; external parameter provenance is disclosed |
| FastAPI/UI demonstration evidence | Working UI and successful API response/logs from the two-laptop run; screenshot/output must come from the actual receiving laptop | Code and runbook: `src/pathsnap/api/main.py`, `ui/app.py`, `docs/two_laptop_deployment.md` | Screenshots still need to be captured during the real two-system run |

## Important reporting and integrity notes

- `train` and `student_test` are the EDA/development partitions. The `evaluator_test` partition was
  evaluated only after the active pipeline was frozen. It must not be used for additional selection,
  tuning or threshold decisions.
- The Level 1 seven-feature HistGradientBoosting results and the later 28-feature Random Forest
  results are different pipeline versions. Do not present them as a single controlled model swap.
- Bayesian tuning was run for the original HistGradientBoosting model. It was not run for the active
  Random Forest.
- The active 28-feature candidate and HMM/Viterbi implementation was adapted from
  `RealAdhi/ML2026`, commit `97a78406269b33eda31fd17a0b00125c87246946`. The assessment guidance
  requires independent implementation. The team must verify this adaptation is allowed and disclose
  it accurately; do not claim it was independently authored.
- AI assistance was used for project documentation, result synthesis and report preparation in this
  collaboration. Disclose AI use as required by the assessment policy and ensure all team members
  understand and can verify submitted work.
- Keep raw datasets, `.env`/secrets, separate viva guides and contributor upload ZIP files out of the
  submitted GitHub tree.
- Review staged files and obtain team approval before pushing to GitHub.
