# Evaluator UI

This client runs on Laptop 2 and communicates only with the Laptop 1 API. It must not read raw
reference routes or evaluator labels. Configure `PATHSNAP_API_URL` to the reachable server address.

The current interface is intentionally a placeholder until the inference API and route response
schema are implemented. UI output should render GPS points, returned predicted geometry, and only
the evaluator-visible metrics returned by the server.

