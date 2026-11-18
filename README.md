# PathSnap

PathSnap reconstructs realistic vehicle routes by matching noisy GPS observations to a trip's
road network. This repository is the project scaffold: raw data stays external, and every result
must come from the team's own runs. The hackathon's evaluator partition must not be inspected or
used during development.

## Current status

The project layout, configuration, raw-file readers, inventory command, API/UI boundaries, and
experiment plan are established. Candidate generation, route-label alignment, model training,
HMM decoding, and the end-to-end prediction endpoint are the next implementation milestones.
No model results or trained artifacts are included yet.

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

## Proposed modeling plan

Map matching has two linked stages: score candidate road arcs for each GPS observation, then decode
a connected route. Arc IDs are local to a trip, so the supervised task should score observation-arc
pairs rather than learn one global class per arc ID. Ground-truth route-to-observation alignment is
an explicit preprocessing decision and must be validated and documented before training.

| Stage | Candidate approach | Role |
|---|---|---|
| Baseline | Nearest road segment by projected point-to-segment distance | Sanity check; exposes candidate-generation and label-alignment issues |
| Candidate scorer 1 | Multinomial Logistic Regression | Fast, interpretable probability baseline on observation-arc features |
| Candidate scorer 2 | Random Forest | Nonlinear interactions and robustness to feature scaling |
| Candidate scorer 3 | Extra Trees | Diverse tree ensemble and fast baseline for small tabular data |
| Candidate scorer 4 | HistGradientBoosting | Strong tabular candidate-ranking model |
| Candidate scorer 5 | RBF SVM with calibrated probabilities | Nonlinear margin comparison; calibration matters for emissions |
| Candidate scorer 6 | Small regularized MLP | Neural comparison, only if trip-group validation supports it |
| Route decoder | Gap-aware HMM with Viterbi decoding over road-arc candidates | Combines model emissions with directed graph connectivity and route plausibility |

Recommended first final-model hypothesis: HistGradientBoosting for candidate scores, followed by a
gap-aware HMM/Viterbi decoder. This is a hypothesis, not a result. The project-specific novelty angle
is to combine calibrated observation-arc evidence with directed network continuity and adapt the
transition penalty to GPS time gaps. Compare this against classifier-only decoding and the nearest-
segment baseline; claim improvement only if measured on permitted held-out trips.

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

API docs are served at `/docs`; `/health` confirms the service is available. Prediction remains
unavailable until the team's trained artifacts and decoder are implemented and registered.

## Repository map

```text
configs/       Experiment defaults and model search spaces
docs/          Architecture and data notes
outputs/       Generated local inventories and experiment records (ignored by Git)
reports/       Team-authored final report material
src/pathsnap/  Data, features, models, decoder, API and visualization package
ui/            Evaluator-facing client, separated from server-side labels
```

## Academic integrity and AI assistance

The supporting assessment rules require independent team implementation and understanding. Treat
this scaffold and its modeling suggestions as a starting point: verify every design choice against
the data, disclose AI assistance as required, and write conclusions from your real experiments.
Every team member should be able to explain the pipeline and reproduce the reported results.

