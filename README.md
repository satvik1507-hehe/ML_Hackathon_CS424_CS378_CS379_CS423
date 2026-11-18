# ML_Hackathon_CS424_CS378_CS379_CS423

## PathSnap

PathSnap reconstructs realistic vehicle routes by matching noisy GPS observations to a trip's
road network. Raw data stays external, and the evaluator partition must not be inspected or used
during development.

## Current status

The data pipeline, candidate features, five-model comparison, Bayesian search, HMM/Viterbi decoder,
experiment logs, and comparison plots have been run on supplied development data. The evaluator
partition was not supplied, so the final evaluation path still needs that data folder.

## Data

Set `PATHSNAP_DATA_ROOT` to the supplied `Student folder` directory. Do not copy raw data into this
repository or commit it. Expected partitions are `train/` (80 trips) and `student_test/` (10 trips).
Each trip has a `.track`, `.nodes`, `.arcs`, and `.route` file sharing its trip ID. The evaluator
partition is reserved for final evaluation; keep its labels outside development and the UI.

```bash
cp .env.example .env
# Edit PATHSNAP_DATA_ROOT in .env, then:
python -m pip install -e .
python -m pathsnap.data.inventory
python scripts/prepare_data.py
python scripts/run_model_suite.py
python scripts/analyze_comparison.py
python scripts/tune_selected_model.py --trials 15 --folds 3
python scripts/compare_tuned_cv.py
python scripts/run_decoder_comparison.py
python scripts/paired_system_analysis.py
python scripts/plot_experiments.py
```

The inventory command writes a CSV summary under `outputs/`. It reads the training and student-test
partitions only and reports file availability, row counts, timestamp gaps, coordinate bounds, and
graph sizes. Verify the observed data before settling feature or distance assumptions.

Generate a real Folium map from a permitted trip (the GPS points, full network, and reference route
are separate toggleable layers):

```bash
python -m pathsnap.visualization.map_trip \
  --data-root "$PATHSNAP_DATA_ROOT" --partition train --trip-id 00000034 \
  --output outputs/maps/trip-00000034.html
```

## Modeling approach

Map matching has two linked stages: score candidate road arcs for each GPS observation, then decode
a connected route. Arc IDs are local to a trip, so the supervised task should score observation-arc
pairs rather than learn one global class per arc ID. The route file is an ordered arc sequence, not a
GPS-row label vector. `features.candidates.align_reference_route` aligns it monotonically to GPS
observations using projected geometry and route progress; alignment distance and candidate misses
are retained as diagnostics.

| Stage | Candidate approach | Role |
|---|---|---|
| Baseline | Nearest road segment by projected point-to-segment distance | Sanity check; exposes candidate-generation and label-alignment issues |
| Candidate scorer 1 | Logistic Regression | Fast, interpretable probability baseline on observation-arc features |
| Candidate scorer 2 | Random Forest | Nonlinear interactions and robustness to feature scaling |
| Candidate scorer 3 | HistGradientBoosting | Strong tabular candidate-ranking model |
| Candidate scorer 4 | Calibrated Linear SVM | Linear margin model with probability estimates for decoder emissions |
| Candidate scorer 5 | Small regularized MLP | Neural comparison on the same observation-arc features |
| Route decoder | Gap-aware HMM with Viterbi decoding over road-arc candidates | Combines model emissions with directed graph connectivity and route plausibility |

Current scorer by the pre-registered primary metric (trip-mean CV Macro F1): HistGradientBoosting.
It narrowly leads the MLP and is not statistically separated from it by paired trip bootstrap, so
describe this as the selection under the primary metric, not a decisive win. HMM/Viterbi is the
larger measured improvement. Its transition penalty adapts to GPS gaps and directed road continuity.

Potential observation-arc features include perpendicular distance, heading agreement, along-segment
position, local road length, elapsed time, implied speed, turn angle, graph reachability, and
shortest-path distance from the previous observation's candidates. Use metric coordinates (projected
per trip or geodesic calculations), not Euclidean distances in longitude/latitude degrees.

## Experiment rules

- Keep all observations from a trip in the same fold; use group-aware cross-validation on the 80
  training trips.
- Train and compare at least five distinct models. Save configurations, seeds, fold metrics, timing,
  and the actual predictions needed to reproduce comparisons.
- Use the 10 `student_test` trips for development-stage comparisons as permitted by the brief. Do not
  tune against the evaluator partition.
- Run Bayesian hyperparameter optimization (Optuna is included as a dependency) on training-trip
  cross-validation. Retain the full trial database and compare tuned versus untuned models.
- Report candidate miss rate alongside observation-level accuracy, macro F1, log loss where
  meaningful, route continuity, and sequence-level metrics.
- Compare classifier-only outputs with HMM/Viterbi outputs. Preserve candidate-to-arc identity and
  define behavior for missing candidates, large gaps, and disconnected transitions.
- Generate plots and Folium maps from actual implementation outputs. Never invent metrics, curves,
  maps, or experiment results.

## Level 1 results

Five-fold `GroupKFold` keeps each trip together across the 80 training trips. The 10 student-test
trips are used for development comparison. Macro F1 is averaged across trips; Top-1 and Log Loss are
observation-weighted. Candidate misses remain in the evaluation denominator.

| Model | CV Macro F1 | CV Top-1 | Student Macro F1 | Student Top-1 | Student Log Loss |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | **0.4826** | **0.6187** | **0.5137** | **0.7028** | 1.2960 |
| Small MLP | 0.4824 | 0.6166 | 0.5121 | 0.7017 | **1.1925** |
| Random Forest | 0.4737 | 0.6136 | 0.5021 | 0.6998 | 1.2615 |
| Logistic Regression | 0.4621 | 0.6031 | 0.4931 | 0.6901 | 1.4649 |
| Calibrated Linear SVM | 0.4614 | 0.6022 | 0.4943 | 0.6897 | 1.3223 |

The MLP has better student-test Log Loss and slightly better student-test route continuity;
HistGradientBoosting has the highest CV Macro F1 and student-test Top-1 point estimates. The paired
trip bootstrap did not establish a Macro F1 difference from the MLP. The training candidate-miss
rate is 12.12% observation-weighted (1.58% on student_test); missed training observations are omitted
from scorer fitting because no positive candidate is available, but remain in evaluation.

## Bayesian tuning and route-decoder results

Optuna TPE ran 15 trials with 3-fold trip-group CV for HistGradientBoosting. Best settings:
`max_iter=220`, `learning_rate=0.09264`, `max_leaf_nodes=15`, `min_samples_leaf=15`,
`l2_regularization=0.001184`. Tuned student-test Macro F1 was 0.5154 versus 0.5137 untuned; paired
intervals across 10 trips include zero, so the tuning gain is small and uncertain.

| Partition | Method | Macro F1 | Top-1 | Route continuity |
|---|---|---:|---:|---:|
| Training OOF | Classifier only | 0.4826 | 0.6187 | 0.8945 |
| Training OOF | Classifier + HMM/Viterbi | 0.5421 | 0.6267 | 0.9746 |
| Student test | Classifier only | 0.5137 | 0.7028 | 0.9117 |
| Student test | Classifier + HMM/Viterbi | **0.5689** | **0.7124** | **0.9866** |

The student-test paired bootstrap supports the Macro F1 and continuity gains; the Top-1 interval
includes zero. Full per-trip results and plots are in `outputs/experiments/`; trained artifacts are
in `artifacts/final/`.

## Architecture

```text
Laptop 2: evaluator UI  -- trip ID / GPS request -->  Laptop 1: FastAPI
                                                     candidate generation
                                                     candidate scorer
                                                     HMM + Viterbi
                                                     evaluator-only scoring
```

The API owns the road network, model artifacts, and any evaluation labels. The UI receives predicted
routes and display metrics only; it must never receive reference routes or labels. Bind the API to a
reachable interface and configure the UI's `PATHSNAP_API_URL` for the host network before the demo.

## Run

Start the backend on Laptop 1:

```bash
make install
make api
```

Start the evaluator interface on Laptop 2 (install the same package and set `PATHSNAP_API_URL`):

```bash
make install
make ui
```

API docs are served at `/docs`; `/health` confirms the service is available. Prediction requires
`PATHSNAP_EVALUATION_ROOT`, the trained model artifact, and decoder configuration. The API predicts
without loading reference routes, then calculates metrics server-side; the UI receives no reference
route or labels.

## Repository map

```text
configs/       Experiment defaults and model search spaces
docs/          Architecture and data notes
outputs/       Actual inventories, experiment records, plots, and maps
reports/       Team-authored final report material
src/pathsnap/  Data, features, models, decoder, API and visualization package
ui/            Evaluator-facing client, separated from server-side labels
```

## Academic integrity and AI assistance

The supporting assessment rules require independent team implementation and understanding. Treat
this scaffold and its modeling suggestions as a starting point: verify every design choice against
the data, disclose AI assistance as required, and write conclusions from your real experiments.
Every team member should be able to explain the pipeline and reproduce the reported results.

## Level 1 viva materials

- [Level 1 viva review PDF](output/pdf/PathSnap_Level1_Viva_Review.pdf): method, five-model results, selection caveats, likely viva questions, and a demo walkthrough.
- [Level 1 demo checklist](docs/level1_viva_demo.md): concise repository tour and reproduction commands.
- Actual MLP fit history: `outputs/experiments/mlp_training_history.csv` and `outputs/experiments/plots/mlp_training_history.svg`. The validation series is accuracy because scikit-learn's early-stopping MLP does not expose validation loss.
