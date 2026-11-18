# Two-laptop demo and evaluation runbook

This project separates inference from the evaluator-facing interface. Laptop 1 runs FastAPI,
loads the model and trip files, predicts the route, and calculates metrics. Laptop 2 runs Streamlit
and sends only the selected trip ID to the API. Reference routes and labels stay on Laptop 1.

## Data mode for this checkout

The supplied `Student folder` contains `train/` and `student_test/`, but no reserved evaluator
partition. For a local rehearsal, point the API at `student_test/` and describe the run as a
development demo. Do not call it a hidden-evaluator result. At the actual evaluation, set
`PATHSNAP_EVALUATION_ROOT` on Laptop 1 to the provided reserved-evaluation folder. The UI and API
commands stay the same.

The API predicts with `include_route=False`; only after Viterbi has fixed the predicted sequence does
it load the reference route to calculate metrics. The API response contains GPS points, predicted
route geometry, and metric values, but no reference route or label sequence.

## Before the demo

1. Put the same project checkout on both laptops. Raw data and the trained artifact are needed only
   on Laptop 1. Do not copy raw data or `.env` into GitHub.
2. Connect both laptops to the same trusted Wi-Fi network.
3. On Laptop 1, check that these files exist:
   `artifacts/final/candidate_scorer_tuned.joblib` and
   `outputs/experiments/selected_decoder.json`.
4. On Laptop 1, verify the chosen partition folder contains trip subfolders with `.track`, `.nodes`,
   `.arcs`, and `.route` files. For rehearsal, that folder is
   `/Users/satviklokesh/Downloads/Student folder/student_test`.

## Laptop 1: start the FastAPI model server

Open Terminal in the repository and run:

```bash
make install
export PATHSNAP_EVALUATION_ROOT="/absolute/path/to/Student folder/student_test"
make api
```

For the actual evaluation, replace the `student_test` path with the reserved evaluator partition
path supplied for the evaluation. Keep that path on Laptop 1.

The server listens on all laptop network interfaces at port `8000`. Keep this terminal visible during
the demo: Uvicorn prints the incoming requests, including `GET /health`, `GET /trips`, and
`POST /predict/{trip_id}`. These access logs are the required evidence that Laptop 1 received the
inference request.

Find Laptop 1's local Wi-Fi IP address on macOS with:

```bash
ipconfig getifaddr en0
```

If that prints nothing, check the active Wi-Fi interface in System Settings or run `ifconfig` and
use the active interface's LAN address. Do not use a public IP address.

## Laptop 2: start the evaluator interface

Open Terminal in the repository on Laptop 2. Replace `192.168.1.20` with Laptop 1's LAN IP:

```bash
make install
export PATHSNAP_API_URL="http://192.168.1.20:8000"
make ui
```

Streamlit prints a local address, normally `http://localhost:8501`. Open it on Laptop 2. The browser
loads the Streamlit interface; Streamlit makes the API request to Laptop 1.

## Check the connection from Laptop 2

Before opening the UI, these requests can verify the two machines can communicate. Run them on
Laptop 2, replacing the address as above:

```bash
curl "http://192.168.1.20:8000/health"
curl "http://192.168.1.20:8000/trips"
```

The first should return `{"status":"ok","service":"pathsnap-api"}`. The second returns trip IDs
available to the API. Confirm that Laptop 1's terminal shows both GET requests. From Laptop 2, a
direct prediction request is:

```bash
curl -X POST "http://192.168.1.20:8000/predict/00000007"
```

Use a trip ID returned by `/trips`; `00000007` is only an example. The JSON response should contain
`gps_points`, `predicted_route`, `predicted_route_segments`, and `metrics`. It must not contain a
reference route or reference labels. The API terminal should show the `POST /predict/...` request.

## Live presentation sequence

1. Show both laptops: Laptop 1's API terminal and Laptop 2's PathSnap page.
2. Explain the separation: Laptop 2 sends a trip ID; Laptop 1 owns data, features, the Random Forest,
   HMM/Viterbi, and server-side scoring.
3. On Laptop 2, select an available trip and press **Reconstruct route**.
4. Keep Laptop 1's terminal visible as `POST /predict/{trip_id}` arrives and completes.
5. On Laptop 2, point to the blue GPS track, orange predicted route, layer controls, and the returned
   metrics. State which partition the demo uses.
6. Explain that route labels never travel to Laptop 2. The API first fixes its prediction and only
   then reads the reference route to calculate server-side metrics.

For a `student_test` rehearsal, call it a development-set demo. Do not claim its metrics are the
reserved-evaluator score. During the actual evaluation, use the provided reserved partition and show
that the UI offers all ten supplied trip IDs.

## Required screenshots and evidence

The submission guidelines ask for screenshots of the working UI and successful API responses on the
receiving laptop. Capture:

- Laptop 2 showing a selected trip, the route map, and returned metrics.
- Laptop 1's terminal showing the corresponding `POST /predict/{trip_id}` request.
- If requested, a successful `/health` or `/predict/{trip_id}` response in Laptop 2's Terminal.

Use genuine captures from the actual run. The API output and UI should come from the same code and
model artifacts as the submitted repository.

## Troubleshooting

- **Laptop 2 cannot reach `/health`:** verify both laptops are on the same Wi-Fi, the API terminal is
  still running, Laptop 1's IP is correct, and the API is listening on port `8000`.
- **`/trips` returns 503:** `PATHSNAP_EVALUATION_ROOT` is missing, points to the parent data folder
  instead of the partition folder, or points to a folder with no trip directories.
- **Prediction returns 503:** check that the model and decoder files listed above are present and
  that the trip folder has all four required files.
- **The UI still calls `127.0.0.1`:** set `PATHSNAP_API_URL` in Laptop 2's same Terminal before
  running `make ui`.
- **Firewall prompt appears:** allow the Python process to accept connections on the local network.

Stop each process with `Ctrl-C` in its own Terminal after the demo.
