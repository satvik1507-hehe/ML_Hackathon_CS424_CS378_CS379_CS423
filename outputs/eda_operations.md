# PathSnap EDA and preprocessing audit

Generated from the supplied `train/` and `student_test/` folders. Raw trip files remain outside Git.

## Raw file and cell audit

| Partition | Trips | Files checked | Numeric cells read | Missing cells | Malformed rows | Extra cells | Non-finite cells | Blank rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 80 | 320 | 12,277,365 | 0 | 0 | 0 | 0 | 0 |
| student_test | 10 | 40 | 612,417 | 0 | 0 | 0 | 0 | 0 |

No raw data cells or raw rows were dropped by cleaning: all expected cells parsed, and there were no missing, malformed, extra, blank, or non-finite cells. Duplicate GPS coordinates were retained: 572 rows in train and 6 in student_test.

## Timestamp, graph, and route checks

| Diagnostic | train | student_test |
|---|---:|---:|
| GPS intervals exactly 1 second | 201,048/208,337 (96.50%) | 18,886/19,150 (98.62%) |
| Non-positive timestamp intervals | 0 | 0 |
| Maximum interval in any trip | 153,064 s | 210 s |
| Trips with max gap > 10 s / > 60 s / > 300 s | 40 / 22 / 11 | 2 / 1 / 0 |
| Invalid road-arc endpoint references | 0 | 0 |
| Invalid reference-route arc IDs | 0 | 0 |
| Isolated road nodes | 0 | 0 |
| Reference-route disconnected adjacent arc steps | 13 | 1 |

All timestamps are nondecreasing. A trip can still contain a very long positive GPS gap; the longest observed gap was 153,064 seconds in train and 210 seconds in student_test. These gaps are preserved and represented by elapsed-time features / gap-aware route decoding rather than deleting valid records.

## Operations and records affected

1. **Read and validate:** load whitespace-delimited `.track` (3 columns), `.nodes` (2), `.arcs` (2 integer IDs), `.route` (1 integer ID). Fail on absent files, wrong column count, non-finite values, empty GPS/nodes, invalid arc endpoints, or invalid route IDs. No raw row was removed in the provided data.
2. **Check duplicates and ordering:** count repeated GPS coordinates and non-positive timestamp intervals; do not drop repeated coordinates. All timestamp intervals were positive.
3. **Project coordinates:** project each trip from WGS84 longitude/latitude into its corresponding UTM zone for meter-based geometry calculations.
4. **Align route labels:** monotonically align GPS observations to occurrences of arcs in the ordered route, using a 100 m search radius. Keep aligned distance diagnostics.
5. **Generate candidates:** query up to the nearest 8 directed road arcs within 60 m for each observation; fall back to the nearest arc if the radius returns none.
6. **Build candidate features:** cross-track distance, heading cosine, along-arc fraction, arc length, GPS speed, elapsed GPS gap, and GPS step distance.
7. **Training-only exclusion:** if the aligned reference arc is absent from the candidate set, the observation has no positive candidate. Exclude all its candidate rows from scorer fitting, but keep that observation in evaluation. Train excluded 25,268 of 208,417 observations (12.12%), which corresponds to 96,702 candidate rows. Fit uses 1,149,445 candidate rows. Student_test has 302 missed observations (1.58%); these remain in its evaluation denominator and are never used for fitting.
8. **No label leakage at inference:** prediction constructs candidates without reading `.route`; route labels are loaded after the predicted sequence is fixed for server-side evaluation.

## Partition-specific raw row counts

| Partition | track observations | nodes | directed arcs | route arc entries |
|---|---:|---:|---:|---:|
| train | 208,417 | 1,933,485 | 3,876,904 | 31,336 |
| student_test | 19,160 | 90,689 | 185,041 | 3,477 |

These counts describe only the provided development data. The evaluator partition was not available in the working folder.
