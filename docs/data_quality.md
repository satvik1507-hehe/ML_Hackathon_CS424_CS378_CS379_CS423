# PathSnap data quality and EDA

This report is based on the generated tables in [`outputs/eda/`](../outputs/eda/). The audit reads
only `train` (80 trips) and `student_test` (10 trips). It does not open an evaluator partition.
Distances and speeds use the per-trip projected metric coordinates used by the existing candidate
code. The 55 m/s value below is an explicit screening threshold for review, not a claim that every
flagged observation is invalid.

## Inventory

| Partition | Trips | Present trip files | GPS rows | Road nodes | Directed arcs |
|---|---:|---:|---:|---:|---:|
| train | 80 | 320 / 320 | 208,417 | 1,933,485 | 3,876,904 |
| student_test | 10 | 40 / 40 | 19,160 | 90,689 | 185,041 |

The per-trip counts and file-presence flags are in [`dataset_inventory.csv`](../outputs/eda/dataset_inventory.csv).

## Findings, assumptions, and decisions

| Finding from EDA CSVs | Assumption | Decision |
|---|---|---|
| Across 12,889,782 expected numeric cells, `missing`, `extra`, `nonfinite`, and `malformed_rows` are all zero; all 90 trips have the four expected files. | A raw value is valid for this check when its file row has the expected number of parseable, finite numeric fields. | No cell or row is removed or imputed. The per-trip audit remains available in [`missing_invalid_by_trip.csv`](../outputs/eda/missing_invalid_by_trip.csv). |
| No GPS or road-node coordinate is outside longitude/latitude bounds. | Standard longitude and latitude ranges are a basic validity screen, not a location-specific map check. | Keep coordinates as supplied; use per-trip projected metres for distances. |
| There are 572 duplicate GPS coordinate rows in train and 6 in student_test. | Repeated positions can be legitimate stationary samples or sensor quantization; coordinates alone do not establish a bad row. | Retain duplicates and their timestamps. Do not deduplicate the trajectory. |
| All trips have nondecreasing timestamps and there are zero nonpositive timestamp intervals. Train has 201,048 / 208,337 exactly-one-second gaps (96.50%); student_test has 18,886 / 19,150 (98.62%). | A long positive interval is a sampling gap, not a missing record unless another source proves it. | Keep observations and gaps; expose the gap and use it in the existing gap-aware feature/decoder path. The largest observed gaps are listed in [`largest_gap_trips.csv`](../outputs/eda/largest_gap_trips.csv). |
| Maximum gaps are 153,064 s in train and 210 s in student_test. | Large gaps deserve review because movement during the gap is under-observed. | Do not interpolate or discard points in EDA. Keep the raw gap and flag it for interpretation. |
| At the 55 m/s screening threshold, 989 train steps and 4 student_test steps are flagged; maximum implied speeds are 180.54 m/s and 72.97 m/s respectively. | A flagged step can reflect GPS noise, a long gap, or a timestamp/position discontinuity; the threshold is only a review heuristic. | Preserve values, report flags, and avoid treating the flags as validated travel speeds. See [`step_speed_samples.csv`](../outputs/eda/step_speed_samples.csv) and [`geographic_extents_speed.csv`](../outputs/eda/geographic_extents_speed.csv). |
| All road arc endpoints and reference arc IDs are in range; there are zero isolated road nodes in the per-trip network summary. | A weak component counts connectivity after ignoring arc direction; it does not imply that all components are reachable in the directed graph. | Report weak components per trip and preserve the directed arcs. See [`road_network_summary.csv`](../outputs/eda/road_network_summary.csv). |
| Reference route adjacency has 14 disconnected consecutive arc pairs across 11 trips: 13 pairs in 10 train trips and 1 pair in 1 student_test trip. | The `.route` file is treated as its given ordered list of arc IDs. A broken endpoint between adjacent entries is a data continuity issue; EDA does not silently repair it. | Keep the supplied route order and report the broken steps. This route is reference geometry for alignment and evaluation, not an observation-by-observation GPS label file. See [`reference_route_summary.csv`](../outputs/eda/reference_route_summary.csv). |
| At top_k=8, candidate misses are 25,268 / 208,417 observations (12.12%) in train and 302 / 19,160 (1.58%) in student_test. Increasing top_k to 16 lowers these to 11.10% and 0.41%. | A miss follows the existing candidate/alignment definition: the aligned route label is beyond the 100 m label radius or is absent from the selected top-k candidates. | Keep the existing definition and code; report the sweep rather than changing the model or metric. See [`candidate_topk_sweep.csv`](../outputs/eda/candidate_topk_sweep.csv). |
| Train 54 has a 90.61% top-8 miss rate; 16,158 of its 19,313 aligned observations are over 100 m from the reference route (median 138.59 m; p95 210.05 m). Train 33 has a 67.06% miss rate; 2,978 of 5,844 observations are over 100 m (median 122.41 m; p95 142.06 m). | The 100 m rule defines the current alignment miss. High nearest-reference distances are evidence of GPS/reference-route offset or noise, though they do not identify which source is wrong. | Attribute the majority of each miss rate to GPS-to-reference alignment distance, not to candidate ranking: within-radius rank misses are 1,342 for trip 54 and 941 for trip 33. Trip 54 also has a 79 s maximum gap and 402 duplicate-coordinate rows; trip 33 has a 3 s maximum gap, 115 duplicates, and one disconnected reference-route step. The metric distance evidence points to a route/GPS spatial mismatch for both; the gap is an additional concern for 54. See [`high_miss_investigation.csv`](../outputs/eda/high_miss_investigation.csv). |
| Training candidate rows used for scorer-fit feature plots contain 183,149 positives among 1,149,445 rows (15.93%). | Candidate rows for observations marked as misses are excluded from the existing scorer-fit training rows. | Plot the seven existing features by positive/negative class using covered training rows only; student_test and evaluator data are not used for feature distributions or correlations. See [`candidate_class_balance.csv`](../outputs/eda/candidate_class_balance.csv), [`feature_distribution_bins.csv`](../outputs/eda/feature_distribution_bins.csv), and [`feature_correlation.csv`](../outputs/eda/feature_correlation.csv). |

## Plot and map index

Each PNG is saved at 200 dpi. Plot data is stored in CSV files in the same output directory; every
figure identifies its partition in its title or caption. The six representative GPS tracks and all
distance calculations use projected metres. Static network overlays and interactive maps use a
deterministic sample of nearby road arcs for readability; road node/arc counts and connectivity
summaries use every source arc. Interactive maps in `outputs/eda/maps/` are limited to one normal
train trip, train trip 54, and one student_test trip; each has toggleable, colour-coded GPS, nearby
road, and reference-route layers, plus a predicted-route layer when the saved model is available.
GPS map markers are sampled and the full GPS trace remains a polyline.
