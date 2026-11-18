# Active PathSnap model

The API now uses the Random Forest retrained from the team repository's implementation,
followed by its HMM/Viterbi route decoder. The model and decoder are wired into the regular
`/predict/{trip_id}` request; they are not a separate demo path.

## Training and selection evidence

Training used the supplied `train` partition (80 trips), with five-fold GroupKFold by trip.
`student_test` (10 trips) was used only for development comparison. No evaluator partition was
read. The model's 28 candidate features and candidate generator match the teammate implementation;
the API applies the same 60 m search radius and keeps up to 12 candidates per observation.

| Candidate model | CV reachable Top-1 mean | Student-test classifier Top-1 (all observations) | Student-test Top-1 after HMM |
|---|---:|---:|---:|
| Logistic Regression | 82.96% | 83.99% | 90.54% |
| Random Forest | 85.02% | 85.82% | **94.16%** |
| LightGBM | 84.83% | 85.84% | 92.72% |
| XGBoost | **85.15%** | **86.48%** | 93.39% |
| MLP (64-32) | 84.82% | 85.39% | 90.92% |

The HMM result shown uses the teammate repository's decoder settings (originally selected for its
XGBoost pipeline), applied unchanged to each retrained classifier. No decoder tuning was done on
student_test. For the active Random Forest, student-test Top-1 increased by 8.35 percentage points
(from 85.82% to 94.16%). Mean trip accuracy was 94.47%; mean route continuity was 97.72%.
This is a development result on 10 trips, not a guaranteed evaluator score, and it does not exceed
95%. The XGBoost classifier had the best mean reachable-observation CV Top-1, while Random Forest
with HMM had the best student-test all-observation Top-1 among these runs. Keep that distinction
clear when presenting model selection.

The training candidate miss rate was 10.83%, mostly from reference-route mismatch in trips 54 and
33. Student-test candidate miss rate was 0.20%. The HMM cannot recover a reference arc absent from
its candidate set; these misses remain in the all-observation Top-1 denominator.

## Active artifacts and code

- `artifacts/final/candidate_scorer_tuned.joblib`: active trained Random Forest and feature-order metadata.
- `outputs/experiments/selected_model.json`: active model identity and selection evidence.
- `outputs/experiments/selected_decoder.json`: active HMM parameters.
- `outputs/experiments/teammate_model_retraining/`: five-model CV, student-test HMM comparison, and per-trip result tables.
- `src/pathsnap/features/retrained_candidates.py`: the 28-feature candidate pipeline and local metre projection.
- `src/pathsnap/decoder/retrained_hmm.py`: the route-distance transition model and Viterbi decoder.
- `src/pathsnap/api/main.py`: active inference integration.

The feature and decoder implementation was adapted from the team's repository
[RealAdhi/ML2026](https://github.com/RealAdhi/ML2026), commit
`97a78406269b33eda31fd17a0b00125c87246946`. The experiment runs the repository's five model
families on the supplied PathSnap partitions and keeps the prior PathSnap model/results as an
archive. It does not copy the teammate's pre-trained model artifact.

The earlier seven-feature experiment results remain unchanged in their original comparison files.
