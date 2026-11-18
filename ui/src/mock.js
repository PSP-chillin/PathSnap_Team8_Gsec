// Fake API used only by `npm run dev:mock`, so the UI can be built and tested
// before the real API exists. Trips are synthetic (IDs "mock-01" … "mock-10"),
// generated in the browser around Bengaluru; nothing here comes from the
// dataset or from a model. The page shows a banner whenever this is active.

import { ApiError } from "./api.js";

const TRIP_IDS = Array.from({ length: 10 }, (_, i) => `mock-${String(i + 1).padStart(2, "0")}`);

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildTrip(index) {
  const rand = seededRandom(1000 + index);
  let lat = 12.88 + rand() * 0.1;
  let lon = 77.55 + rand() * 0.12;
  let heading = rand() * Math.PI * 2;

  const corners = [[lat, lon]];
  const legs = 8 + Math.floor(rand() * 6);
  for (let i = 0; i < legs; i += 1) {
    heading += (rand() - 0.5) * 1.6;
    const step = 0.002 + rand() * 0.004;
    lat += Math.cos(heading) * step;
    lon += Math.sin(heading) * step;
    corners.push([lat, lon]);
  }

  const gpsPoints = [];
  for (let i = 0; i < corners.length - 1; i += 1) {
    const [a, b] = [corners[i], corners[i + 1]];
    const samples = 12 + Math.floor(rand() * 10);
    for (let k = 0; k < samples; k += 1) {
      const t = k / samples;
      gpsPoints.push({
        lat: a[0] + (b[0] - a[0]) * t + (rand() - 0.5) * 0.0004,
        lon: a[1] + (b[1] - a[1]) * t + (rand() - 0.5) * 0.0004,
      });
    }
  }

  const predictedRoute = corners.slice(0, -1).map((corner, i) => [
    { lat: corner[0], lon: corner[1] },
    { lat: corners[i + 1][0], lon: corners[i + 1][1] },
  ]);

  return {
    trip_id: TRIP_IDS[index],
    gps_points: gpsPoints,
    predicted_route: predictedRoute,
    metrics: {
      accuracy: 0.84 + rand() * 0.13,
      route_continuity: 0.9 + rand() * 0.1,
      n_points: gpsPoints.length,
      n_predicted_arcs: predictedRoute.length,
    },
    inference_time_ms: Math.round(300 + rand() * 900),
  };
}

function wait(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new ApiError("Request cancelled.", { kind: "cancelled" }));
      },
      { once: true },
    );
  });
}

export function createMockClient({ onLog } = {}) {
  let counter = 0;

  async function fake(method, path, produce, { signal, quiet = false, delayMs = 350 } = {}) {
    const started = performance.now();
    const entry = { id: `mock-${++counter}`, at: new Date(), method, path, status: null, ms: null, ok: false, mock: true };
    try {
      await wait(delayMs, signal);
      const data = produce();
      entry.status = 200;
      entry.ok = true;
      return data;
    } catch (error) {
      if (error instanceof ApiError && error.status) entry.status = error.status;
      throw error;
    } finally {
      entry.ms = Math.round(performance.now() - started);
      if (!quiet) onLog?.(entry);
    }
  }

  return {
    health: (options = {}) => fake("GET", "/health", () => ({ status: "ok", mode: "mock" }), { quiet: true, delayMs: 40, ...options }),
    listTrips: (options = {}) => fake("GET", "/trips", () => ({ trips: TRIP_IDS }), options),
    predict: (tripId, options = {}) =>
      fake(
        "POST",
        "/predict",
        () => {
          const index = TRIP_IDS.indexOf(tripId);
          if (index < 0) {
            throw new ApiError(`POST /predict failed with 404: unknown trip ${tripId}`, { status: 404, kind: "http" });
          }
          return buildTrip(index);
        },
        { delayMs: 500 + Math.random() * 700, ...options },
      ),
  };
}
