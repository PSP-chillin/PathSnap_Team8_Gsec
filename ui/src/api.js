// Talks to the FastAPI server on Laptop 1 and turns its responses into the
// shapes the UI draws. The expected contract is documented in ui/README.md.

import { HEALTH_TIMEOUT_MS, REQUEST_TIMEOUT_MS } from "./config.js";

export class ApiError extends Error {
  /** kind: "network" | "timeout" | "http" | "format" | "cancelled" */
  constructor(message, { status = null, kind = "http" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;
  }
}

let requestCounter = 0;

async function request({ baseUrl, method, path, body, signal, timeoutMs = REQUEST_TIMEOUT_MS, quiet = false, onLog }) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const forwardAbort = () => controller.abort();
  signal?.addEventListener("abort", forwardAbort, { once: true });

  const started = performance.now();
  const entry = { id: ++requestCounter, at: new Date(), method, path, status: null, ms: null, ok: false };

  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    entry.status = response.status;
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    }
    if (!response.ok) {
      throw new ApiError(`${method} ${path} failed with ${response.status}: ${describeError(data, response)}`, {
        status: response.status,
        kind: "http",
      });
    }
    entry.ok = true;
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (timedOut) {
      throw new ApiError(`No response from ${path} within ${Math.round(timeoutMs / 1000)} s.`, { kind: "timeout" });
    }
    if (controller.signal.aborted) throw new ApiError("Request cancelled.", { kind: "cancelled" });
    throw new ApiError(`Cannot reach the API at ${baseUrl}.`, { kind: "network" });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", forwardAbort);
    entry.ms = Math.round(performance.now() - started);
    if (!quiet) onLog?.(entry);
  }
}

function describeError(data, response) {
  // FastAPI puts the reason in {"detail": ...}.
  if (data && typeof data === "object" && "detail" in data) {
    return typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
  }
  if (typeof data === "string" && data.trim()) return data.trim().slice(0, 200);
  return response.statusText || "no details";
}

export function createApiClient({ baseUrl, onLog }) {
  const call = (method, path, options = {}) => request({ baseUrl, method, path, onLog, ...options });
  return {
    health: (options = {}) => call("GET", "/health", { timeoutMs: HEALTH_TIMEOUT_MS, quiet: true, ...options }),
    listTrips: (options = {}) => call("GET", "/trips", options),
    predict: (tripId, options = {}) => call("POST", "/predict", { ...options, body: { trip_id: tripId } }),
  };
}

// ---------- response normalisation ----------

// Points must arrive as {lat, lon} objects (also accepts latitude/longitude or
// lng). A bare [a, b] pair is read as [lat, lon]. The raw dataset files list
// longitude first, so the API must convert before sending.
function toLatLng(point) {
  if (Array.isArray(point) && point.length >= 2) return [Number(point[0]), Number(point[1])];
  if (point && typeof point === "object") {
    return [Number(point.lat ?? point.latitude), Number(point.lon ?? point.lng ?? point.longitude)];
  }
  return null;
}

function isValidLatLng(latLng) {
  return (
    Array.isArray(latLng) &&
    Number.isFinite(latLng[0]) &&
    Number.isFinite(latLng[1]) &&
    Math.abs(latLng[0]) <= 90 &&
    Math.abs(latLng[1]) <= 180
  );
}

function isPointLike(value) {
  if (Array.isArray(value)) return typeof value[0] === "number";
  return Boolean(value) && typeof value === "object" && ("lat" in value || "latitude" in value);
}

// predicted_route may be one flat list of points (one line) or a list of
// segments (one per predicted road segment, each a list of points).
function normalizeRoute(route) {
  if (!Array.isArray(route) || route.length === 0) return [];
  if (isPointLike(route[0])) {
    const line = route.map(toLatLng).filter(isValidLatLng);
    return line.length >= 2 ? [line] : [];
  }
  return route
    .map((segment) => (Array.isArray(segment) ? segment : segment?.points ?? []))
    .map((segment) => segment.map(toLatLng).filter(isValidLatLng))
    .filter((segment) => segment.length >= 2);
}

function centroid(points) {
  if (points.length === 0) return null;
  let lat = 0;
  let lon = 0;
  for (const [a, b] of points) {
    lat += a;
    lon += b;
  }
  return [lat / points.length, lon / points.length];
}

function haversineKm([lat1, lon1], [lat2, lon2]) {
  const rad = Math.PI / 180;
  const dLat = (lat2 - lat1) * rad;
  const dLon = (lon2 - lon1) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

const GROUND_TRUTH_KEYS = ["reference_route", "true_route", "ground_truth", "gt_route", "route_truth", "truth"];

export function normalizeTrips(data) {
  const list = Array.isArray(data) ? data : Array.isArray(data?.trips) ? data.trips : null;
  if (!list) {
    throw new ApiError('GET /trips returned an unexpected shape; expected {"trips": [...]}.', { kind: "format" });
  }
  const seen = new Set();
  const trips = [];
  for (const item of list) {
    const id = typeof item === "string" || typeof item === "number" ? item : item?.trip_id ?? item?.id;
    if (id == null || seen.has(String(id))) continue;
    seen.add(String(id));
    const points = Number(item?.n_points);
    trips.push({ id: String(id), points: Number.isFinite(points) ? points : null });
  }
  return trips;
}

export function normalizePrediction(data, requestedId) {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new ApiError("POST /predict returned something other than a JSON object.", { kind: "format" });
  }

  const rawPoints = Array.isArray(data.gps_points) ? data.gps_points : [];
  const gpsPoints = rawPoints.map(toLatLng).filter(isValidLatLng);
  const routeSegments = normalizeRoute(data.predicted_route);
  const metrics =
    data.metrics && typeof data.metrics === "object" && !Array.isArray(data.metrics) ? data.metrics : {};
  const inference = Number(data.inference_time_ms);

  const warnings = [];
  const dropped = rawPoints.length - gpsPoints.length;
  if (dropped > 0) warnings.push(`${dropped} GPS point(s) had missing or out-of-range coordinates and were not drawn.`);
  if (gpsPoints.length === 0) warnings.push("The response has no GPS points to draw.");
  if (routeSegments.length === 0) warnings.push("The response has no predicted route to draw.");
  if (Object.keys(metrics).length === 0) warnings.push("The response has no metrics.");

  const gpsCentre = centroid(gpsPoints);
  const routeCentre = centroid(routeSegments.flat());
  if (gpsCentre && routeCentre) {
    const gapKm = haversineKm(gpsCentre, routeCentre);
    if (gapKm > 5) {
      warnings.push(
        `GPS points and predicted route are ${Math.round(gapKm)} km apart. One of them probably has latitude and longitude swapped.`,
      );
    }
  }

  // Ground truth must stay on Laptop 1. It is never drawn; flag it if sent.
  const leaked = GROUND_TRUTH_KEYS.filter((key) => key in data);
  if (leaked.length) {
    warnings.push(
      `The response includes ground-truth data (${leaked.join(", ")}). The rules require it to stay on Laptop 1; ask the backend to remove it.`,
    );
  }

  return {
    tripId: String(data.trip_id ?? requestedId),
    gpsPoints,
    routeSegments,
    metrics,
    inferenceMs: Number.isFinite(inference) ? Math.round(inference) : null,
    warnings,
  };
}
