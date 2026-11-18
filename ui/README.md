# PathSnap evaluator UI

The Streamlit client runs on Laptop 2 and calls the FastAPI service on Laptop 1. It lists trip IDs
from the API, requests a route prediction, and displays GPS observations, the predicted route, and
the evaluation metrics returned by the server. Reference routes and labels remain on the API server;
the UI never reads raw trip files.

Configure `PATHSNAP_API_URL` to the Laptop 1 API address, for example
`http://192.168.1.20:8000`, then start the client with `make ui`. Start the API with `make api` and
configure its evaluation data root and model settings first. The map has toggleable GPS and
predicted-route layers, avoids connecting across predicted route breaks, and samples GPS markers
for long trips while retaining the full track line.
