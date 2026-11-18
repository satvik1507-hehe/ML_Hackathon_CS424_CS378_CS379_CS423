# Architecture and implementation decisions

## Problem framing

The raw data consists of timestamped GPS observations, trip-specific road nodes, directed road arcs,
and a chronological reference route. The target is route reconstruction, not a global road-ID
classification problem. Arc IDs are meaningful only inside a trip's road graph.

The intended inference unit is an observation-candidate-arc pair. Candidate scoring estimates how
well an arc explains an observation; sequence decoding selects a temporally and topologically
plausible arc sequence. The candidate generator and label-alignment procedure must be validated
before any model metrics are trusted.

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

## Delivery constraints

The two-laptop setup means the server owns the full inference path and labels; the UI calls the API
with a trip ID and displays the returned prediction. Do not expose the ground-truth route in the
request or response. For a public repository, keep raw data, secrets, and large local experiment
artifacts out of Git unless the organizers explicitly require model artifacts to be submitted.

