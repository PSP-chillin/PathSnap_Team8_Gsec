# PathSnap evaluator interface (Laptop 2)

React app the evaluator uses during the demo. It lists the 10 evaluation trips it
gets from the API, sends the selected trip ID to the FastAPI server on Laptop 1,
and shows the predicted route on a map with the accuracy figures the API computes.

The UI never receives or draws the reference (ground-truth) route: the rules
require it to stay on Laptop 1. If the API ever sends a field such as
`reference_route` or `ground_truth`, the UI ignores it and shows a warning.

## Requirements

- Node.js 18 or newer (`node --version`)
- Internet access for the OpenStreetMap background tiles (a phone hotspot is fine)

## Setup

```bash
cd ui
npm install
```

## Running

| Command | Use |
|---|---|
| `npm run dev` | **Evaluation / real API.** Opens on http://localhost:5173 |
| `npm run dev:mock` | Demo data generated in the browser, for building the UI without the API. Shows a yellow banner. **Never use for the evaluation.** |
| `npm run build` then `npm run preview` | Production build served on http://localhost:4173 |

### Pointing the UI at Laptop 1

Either type the address into **API address** in the page header and press **Connect**
(it is remembered in this browser), or create `ui/.env.local`:

```
VITE_API_URL=http://192.168.43.10:8000
```

Find Laptop 1's address with `ip addr` (Fedora), `ipconfig` (Windows) or
`ipconfig getifaddr en0` (macOS). The green pill in the header shows the API is reachable.

## API contract (for the Laptop 1 backend)

All responses are JSON. Coordinates are **objects with `lat` and `lon`**. The dataset
files list longitude first, so convert before sending.

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /trips`

The trips the evaluator can pick (the 10 evaluation trips on the day).

```json
{ "trips": ["00000015", "00000017"] }
```

Objects are also accepted: `{ "trips": [{ "trip_id": "00000015", "n_points": 1315 }] }`.

### `POST /predict`

Request body:

```json
{ "trip_id": "00000015" }
```

Response:

```json
{
  "trip_id": "00000015",
  "gps_points": [{ "lat": 52.10, "lon": 4.90 }, { "lat": 52.11, "lon": 4.91 }],
  "predicted_route": [
    [{ "lat": 52.10, "lon": 4.90 }, { "lat": 52.11, "lon": 4.91 }],
    [{ "lat": 52.11, "lon": 4.91 }, { "lat": 52.12, "lon": 4.93 }]
  ],
  "metrics": { "accuracy": 0.93, "route_continuity": 0.97, "n_points": 1315 },
  "inference_time_ms": 840
}
```

- `predicted_route`: one list of points per predicted road segment, in travel order
  (start node and end node of each arc). A single flat list of points also works.
- `metrics`: any numbers the API computes against its reference data. Fractions (0–1)
  for keys containing accuracy, precision, recall, F1, continuity, rate or score are
  shown as percentages. `accuracy` is shown as the headline figure if present.
- Errors: raise `HTTPException` with a clear `detail`; the UI shows it to the evaluator.

### CORS

The page runs on a different address from the API, so the backend must allow it:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Demo-day checklist

1. Both laptops on the same network (phone hotspot if college Wi-Fi blocks laptop-to-laptop traffic).
2. Laptop 1: `uvicorn <module>:app --host 0.0.0.0 --port 8000`. The default host only accepts local connections.
3. Laptop 1 firewall allows port 8000. Fedora: `sudo firewall-cmd --add-port=8000/tcp`.
4. Laptop 2: `cd ui && npm run dev`, open http://localhost:5173, enter Laptop 1's address, press **Connect**.
5. Header pill is green and the 10 trips are listed.
6. Press **Run all 10 trips** once to confirm every trip works, then hand over to the evaluator.
7. Keep the API terminal on Laptop 1 visible: the evaluator checks its request log.

## What the screen shows

- **Evaluation trips**: the list from `GET /trips`, each with its result badge, plus
  **Run all** and the mean of the headline metric.
- **Map**: GPS points (blue, input), predicted route (orange), start and end markers.
  Layers can be toggled at the top right. Very long trips draw a subset of GPS points
  for speed; the legend says when.
- **Result**: the API's metrics, inference time reported by the API, and round-trip
  time measured by the UI. Connection problems come with a checklist of likely fixes.
- **API requests**: every request this screen sent, with status and duration.

## Code layout

```
ui/
├── index.html
├── package.json          # pinned dependency versions
├── vite.config.js
├── .env.example          # copy to .env.local to set VITE_API_URL
├── .env.mock             # loaded by `npm run dev:mock`
└── src/
    ├── main.jsx          # entry point
    ├── App.jsx           # state: API address, trips, current run, results, request log
    ├── api.js            # fetch wrapper, error types, response normalisation
    ├── config.js         # API address and timeouts
    ├── format.js         # metric and duration formatting
    ├── mock.js           # synthetic data for dev:mock only
    ├── styles.css
    └── components/
        ├── Header.jsx      # API status and address
        ├── TripList.jsx    # trip picker, run all, summary
        ├── MapView.jsx     # Leaflet map, layers, legend
        ├── ResultPanel.jsx # metrics, timings, errors
        └── RequestLog.jsx  # requests sent from this screen
```
