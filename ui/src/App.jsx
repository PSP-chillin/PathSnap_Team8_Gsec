import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Header from "./components/Header.jsx";
import MapView from "./components/MapView.jsx";
import RequestLog from "./components/RequestLog.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import TripList from "./components/TripList.jsx";
import { createApiClient, normalizePrediction, normalizeTrips } from "./api.js";
import { DEFAULT_API_URL, USE_MOCK, clearApiUrl, loadApiUrl, normalizeBaseUrl, saveApiUrl } from "./config.js";
import { headlineMetric } from "./format.js";
import { createMockClient } from "./mock.js";

const HEALTH_INTERVAL_MS = 30000;
const LOG_LIMIT = 25;

export default function App() {
  const [apiUrl, setApiUrl] = useState(loadApiUrl);
  const [log, setLog] = useState([]);
  const addLog = useCallback((entry) => setLog((previous) => [entry, ...previous].slice(0, LOG_LIMIT)), []);

  const client = useMemo(
    () => (USE_MOCK ? createMockClient({ onLog: addLog }) : createApiClient({ baseUrl: apiUrl, onLog: addLog })),
    [apiUrl, addLog],
  );

  const [health, setHealth] = useState({ state: "checking" });
  const [healthTick, setHealthTick] = useState(0);
  const [trips, setTrips] = useState({ state: "loading", items: [] });
  const [selectedId, setSelectedId] = useState(null);
  const [run, setRun] = useState({ state: "idle" });
  const [results, setResults] = useState({});
  const [batch, setBatch] = useState(null);

  const controllerRef = useRef(null);
  const batchStopRef = useRef(false);
  const autoRunDoneRef = useRef(false);

  // ---- API health: on start, every 30 s, and when the status pill is clicked ----
  useEffect(() => {
    let cancelled = false;
    async function check() {
      const started = performance.now();
      try {
        await client.health();
        if (!cancelled) setHealth({ state: "online", ms: Math.round(performance.now() - started) });
      } catch (error) {
        if (cancelled) return;
        if (error.kind === "http") {
          // The server answered, it just has no /health route.
          setHealth({ state: "online", ms: Math.round(performance.now() - started), note: "no /health route" });
        } else {
          setHealth({ state: "offline", detail: error.message });
        }
      }
    }
    setHealth({ state: "checking" });
    check();
    const timer = setInterval(check, HEALTH_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [client, healthTick]);

  // ---- trip list ----
  const loadTrips = useCallback(async () => {
    setTrips({ state: "loading", items: [] });
    try {
      const data = await client.listTrips();
      setTrips({ state: "ready", items: normalizeTrips(data) });
    } catch (error) {
      setTrips({ state: "error", items: [], error });
    }
  }, [client]);

  useEffect(() => {
    // A new API address starts from a clean slate, including stopping any "run all" batch.
    batchStopRef.current = true;
    controllerRef.current?.abort();
    setResults({});
    setRun({ state: "idle" });
    setSelectedId(null);
    loadTrips();
  }, [loadTrips]);

  // ---- predictions ----
  const runTrip = useCallback(
    async (tripId) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;

      setSelectedId(tripId);
      setRun({ state: "running", tripId });
      const started = performance.now();
      try {
        const raw = await client.predict(tripId, { signal: controller.signal });
        const result = normalizePrediction(raw, tripId);
        const roundTripMs = Math.round(performance.now() - started);
        setRun({ state: "done", tripId, result, roundTripMs });
        setResults((previous) => ({
          ...previous,
          [tripId]: { status: "ok", headline: headlineMetric(result.metrics), inferenceMs: result.inferenceMs, roundTripMs },
        }));
        return "ok";
      } catch (error) {
        if (error.kind === "cancelled") return "cancelled";
        setRun({ state: "error", tripId, error });
        setResults((previous) => ({ ...previous, [tripId]: { status: "error", message: error.message } }));
        return "error";
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
      }
    },
    [client],
  );

  const runAll = useCallback(async () => {
    const ids = trips.items.map((trip) => trip.id);
    batchStopRef.current = false;
    setBatch({ done: 0, total: ids.length });
    for (let i = 0; i < ids.length; i += 1) {
      if (batchStopRef.current) break;
      const outcome = await runTrip(ids[i]);
      if (outcome === "cancelled") break;
      setBatch({ done: i + 1, total: ids.length });
    }
    setBatch(null);
  }, [trips.items, runTrip]);

  const stop = useCallback(() => {
    batchStopRef.current = true;
    controllerRef.current?.abort();
    setRun((previous) => (previous.state === "running" ? { state: "cancelled", tripId: previous.tripId } : previous));
  }, []);

  // Optional deep link for testing: ?trip=<id> runs that trip once the list loads.
  useEffect(() => {
    if (autoRunDoneRef.current || trips.state !== "ready") return;
    const wanted = new URLSearchParams(window.location.search).get("trip");
    if (wanted && trips.items.some((trip) => trip.id === wanted)) {
      autoRunDoneRef.current = true;
      runTrip(wanted);
    }
  }, [trips, runTrip]);

  // ---- API address ----
  const changeApiUrl = useCallback((value) => {
    const clean = normalizeBaseUrl(value);
    saveApiUrl(clean);
    setApiUrl(clean);
  }, []);

  const resetApiUrl = useCallback(() => {
    clearApiUrl();
    setApiUrl(normalizeBaseUrl(DEFAULT_API_URL));
  }, []);

  const runningId = run.state === "running" ? run.tripId : null;
  const mapResult = run.state === "done" ? run.result : null;
  const mapMessage =
    run.state === "error"
      ? `No route to show for trip ${run.tripId}. See the error below.`
      : run.state === "cancelled"
        ? `Trip ${run.tripId} was stopped before the API answered.`
        : undefined;

  return (
    <div className="app">
      <Header
        apiUrl={apiUrl}
        health={health}
        onRecheck={() => setHealthTick((tick) => tick + 1)}
        onApiUrlChange={changeApiUrl}
        onApiUrlReset={resetApiUrl}
      />

      {USE_MOCK && (
        <div className="mock-banner" role="note">
          Demo data mode: responses are generated in this browser, not by the API. Start the UI with
          <code>npm run dev</code> for the evaluation.
        </div>
      )}

      <main className="workspace">
        <aside className="sidebar">
          <TripList
            trips={trips}
            results={results}
            selectedId={selectedId}
            runningId={runningId}
            batch={batch}
            disabled={batch != null}
            onSelect={runTrip}
            onRunAll={runAll}
            onCancel={stop}
            onReload={loadTrips}
          />
        </aside>

        <section className="stage">
          <MapView result={mapResult} runningId={runningId} emptyMessage={mapMessage} />
          <div className="details">
            <ResultPanel run={run} apiUrl={apiUrl} />
            <RequestLog entries={log} mock={USE_MOCK} />
          </div>
        </section>
      </main>
    </div>
  );
}
