# Reserved evaluator results

This is the final evaluation of the frozen active Random Forest + HMM/Viterbi pipeline on the
`evaluator_test` partition (10 trips, 19,674 GPS observations). The trained artifact and decoder
configuration were loaded from the existing project files. No fitting or tuning was performed on
evaluator data. For each trip, the reference route was omitted while candidate generation, scoring,
and decoding ran; it was read only after the prediction sequence was fixed.

## Metric scope

- `summary.json` gives observation-weighted Top-1 accuracy, mean-trip metrics, pooled route
  continuity, candidate misses, and mean inference time.
- `per_trip_metrics.csv` gives each trip's metrics. Macro F1, macro precision, macro recall, and
  route F1 are computed within each trip because road-arc IDs are trip-local.
- Classifier log loss normalizes scorer values across each observation's candidates. A missing
  reference candidate receives the fixed `1e-12` probability floor. HMM log loss is not reported,
  because Viterbi emits a best sequence rather than calibrated per-observation probabilities.
- Candidate misses stay in the Top-1 denominator. Route continuity is the share of adjacent
  predictions that use the same arc or connect end-to-start in the directed network.

## Graphs

Every PNG is rendered at 200 dpi and has a same-numbered CSV containing its plotted values.

1. `01_overall_model_comparison`: observation-weighted Top-1, mean-trip Macro F1, and pooled route
   continuity for Random Forest alone versus Random Forest + HMM/Viterbi.
2. `02_top1_by_trip`: all-observation Top-1 for each trip, before and after sequence decoding.
3. `03_macro_f1_by_trip`: per-trip Macro F1 before and after sequence decoding.
4. `04_candidate_misses_by_trip`: count and share of observations whose aligned reference arc was
   absent from the generated candidate set.
5. `05_route_continuity_by_trip`: directed-network continuity before and after decoding.
6. `06_prediction_time_by_trip`: candidate generation, scoring, and decoding time per trip on this
   machine; treat as local runtime evidence, not a hardware-independent benchmark.
7. `07_classifier_log_loss_by_trip`: candidate-normalized Random Forest log loss per trip.

## Reproduce

```bash
.venv/bin/python scripts/evaluate_reserved.py \
  --data-root "/absolute/path/to/evaluator_test" \
  --output outputs/final_evaluation
```

The script saves only summary metrics and figures; it does not persist raw evaluator data, aligned
labels, or observation-level predicted routes. The current outputs are local and have not been
committed or pushed.
