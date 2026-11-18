# Architecture and implementation decisions

## Problem framing

The raw data consists of timestamped GPS observations, trip-specific road nodes, directed road arcs,
and a chronological reference route. The target is route reconstruction, not a global road-ID
classification problem. Arc IDs are meaningful only inside a trip's road graph.

The inference unit is an observation-candidate-arc pair. Candidate scoring estimates how well an
arc explains an observation; sequence decoding selects a temporally and topologically plausible arc
sequence. The reference route contains fewer arcs than GPS observations, so route-to-observation
labels are produced by monotone geometric alignment and tracked with alignment-distance and
candidate-miss diagnostics.

## Data boundaries

- `train/`: fitting, grouped cross-validation, feature decisions, Bayesian search.
- `student_test/`: development comparison and debugging allowed by the hackathon brief.
- evaluator test: final evaluation only; keep out of local development and the UI payload.

Preprocessing that learns parameters must be fit inside each training fold. Split by trip, never by
observation, to avoid leakage from nearly identical sequences.

## Proposed package boundaries

- `data`: strict parsing and inventory for raw trip files.
- `features`: projected geometry, candidate generation, observation-arc features, and route-label
  alignment.
- `models`: common candidate scorer interface and registered estimators.
- `decoder`: transition scoring and Viterbi route decoding, including GPS-gap handling.
- `evaluation`: grouped CV, permitted student-test comparison, metrics, and experiment records.
- `api`: model-loading and inference service. Any reference labels stay server-side.
- `ui`: remote client only; it renders GPS points and returned predicted route.
- `visualization`: Folium output built from actual data and predictions.

## Novelty hypothesis

Use a gap-aware transition model: transition plausibility combines directed graph connectivity,
network shortest-path distance, elapsed time, and implied speed. Increase tolerance for longer GPS
gaps rather than applying one fixed transition threshold. Candidate emission scores come from a
calibrated tabular model. This is an engineering hypothesis, not a claim of algorithmic novelty;
compare it against a fixed-transition HMM and classifier-only predictions.

## Experiment snapshot

The active API model has since been promoted to the retrained teammate Random Forest + HMM pipeline.
See [active_model.md](active_model.md) for its 28-feature inference path and student-test evidence.
The earlier experiment figures below are preserved as historical results.


The original Level 1 comparison used Logistic Regression, Random Forest, HistGradientBoosting,
calibrated Linear SVM, and a small MLP. Those results remain preserved. The active API now runs the
retrained teammate Random Forest + HMM pipeline; see [active_model.md](active_model.md) for its
training and student-test comparison.

## Delivery constraints

The two-laptop setup means the server owns the full inference path and labels; the UI calls the API
with a trip ID and displays the returned prediction. Prediction loads a trip without its reference
route, performs candidate scoring and Viterbi decoding, and only then loads the route for server-side
metrics. Do not expose the ground-truth route in the request or response. Keep raw data and secrets
out of Git; include final trained models and experiment records required by the submission guide.

