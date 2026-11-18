# PathSnap Level 1 Viva Demo

Use this as a short walkthrough of the work completed for the five-model Level 1 comparison. The detailed study guide is `output/pdf/PathSnap_Level1_Viva_Review.pdf`.

## 6-8 minute walkthrough

1. **Problem (30 seconds):** explain that PathSnap matches noisy GPS observations to a trip-specific directed road network. Arc IDs are local to each trip, and the route file is an ordered list rather than one label per GPS point.
2. **Dataset (45 seconds):** show `outputs/data_inventory.csv`. The provided working data has 80 training trips and 10 `student_test` trips. Mention that the separate evaluator set was not present locally and was not used.
3. **Pipeline (60 seconds):** show `src/pathsnap/features/candidates.py`. Explain projected meter coordinates, top-8 candidates within 60 m, seven candidate features, monotone alignment to the reference route, and candidate-miss tracking.
4. **Five models (45 seconds):** show `src/pathsnap/models/candidates.py` and `configs/experiment.yaml`: Logistic Regression, Random Forest, HistGradientBoosting, calibrated Linear SVM, and a small MLP.
5. **Results (90 seconds):** show `outputs/experiments/model_comparison.csv` and the three plots in `outputs/experiments/plots/`. Explain CV Macro F1, student-test Top-1, student-test Macro F1, Log Loss, continuity, and the 12.12% training candidate miss rate.
6. **Selection (60 seconds):** show `outputs/experiments/selected_model.json` and `outputs/experiments/paired_bootstrap.csv`. HistGradientBoosting is selected by highest trip-mean 5-fold CV Macro F1 (0.4826), but the paired interval versus the MLP includes zero. Call it the selected top-scoring model, not a decisive statistical winner.
7. **Reproduction (45 seconds):** show `scripts/prepare_data.py`, `scripts/run_model_suite.py`, and `scripts/analyze_comparison.py`.
8. **Follow-on work (45 seconds):** briefly point to `outputs/experiments/tuning_result.json` and `outputs/experiments/decoder_comparison.csv`. Explain that HPO and HMM/Viterbi are downstream of, and separate from, the five-model selection.

## Reproduction commands

Set `PATHSNAP_DATA_ROOT` to the external `Student folder` containing `train/` and `student_test/`. Keep raw data outside Git.

```bash
python -m pip install -e .
python scripts/prepare_data.py
python scripts/run_model_suite.py
python scripts/analyze_comparison.py
python scripts/plot_experiments.py
```

The five-model suite refits the estimators and can take several minutes. For a short viva, show the already generated CSVs and plots first, and rerun only if asked.

## Files to open during the demo

- `outputs/data_inventory.csv`
- `src/pathsnap/features/candidates.py`
- `src/pathsnap/models/candidates.py`
- `configs/experiment.yaml`
- `outputs/experiments/model_comparison.csv`
- `outputs/experiments/selected_model.json`
- `outputs/experiments/paired_bootstrap.csv`
- `outputs/experiments/plots/cv_macro_f1.svg`
- `outputs/experiments/plots/student_test_accuracy_macro_f1.svg`
- `outputs/experiments/plots/log_loss_comparison.svg`
- `outputs/experiments/mlp_training_history.svg` (actual full-training refit history; validation series is accuracy)

## Be precise about limits

- `student_test` is development data, not the hidden evaluator partition.
- The HGB and MLP Macro F1 difference is tiny and not statistically separated in the paired trip bootstrap.
- The MLP has lower student-test Log Loss; no single model wins every metric.
- The training candidate-miss rate is much higher than student-test, which affects measured performance.
- The MLP plot is based on a real full-training refit with the registered settings. It shows training loss and early-stopping validation accuracy; Scikit-learn does not expose validation loss here, so do not label the validation curve as loss.
- The candidate scorer's inference-time metric excludes data loading and candidate generation.
- All team members should understand the results and explain them in their own words. Follow the assessment policy for disclosure of AI assistance.
