import { useEffect, useState } from "react";
import { REQUEST_TIMEOUT_MS } from "../config.js";
import { formatDuration, formatMetric, headlineMetric, prettyLabel } from "../format.js";

function useElapsed(active) {
  const [started, setStarted] = useState(null);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) {
      setStarted(null);
      return undefined;
    }
    const begin = Date.now();
    setStarted(begin);
    setNow(begin);
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, [active]);
  return started == null ? 0 : now - started;
}

function hintsFor(error, apiUrl) {
  switch (error.kind) {
    case "network":
      return [
        `Check the API address in the header: ${apiUrl}`,
        "On Laptop 1, start the API with --host 0.0.0.0 (the default only accepts local connections).",
        "Open the API port in Laptop 1's firewall. Fedora: sudo firewall-cmd --add-port=8000/tcp",
        "Put both laptops on the same network. A phone hotspot avoids college Wi-Fi blocking.",
        `The API must allow requests from this page (CORS origin ${window.location.origin}).`,
      ];
    case "timeout":
      return [
        "The API is still working or has stalled. Check its log on Laptop 1, then try again.",
        `The UI waits up to ${Math.round(REQUEST_TIMEOUT_MS / 1000)} s (VITE_REQUEST_TIMEOUT_MS).`,
      ];
    case "format":
      return ["The response doesn't match the contract in ui/README.md."];
    case "http":
      if (error.status === 404) return ["The API doesn't know this route or trip ID. Check it against ui/README.md."];
      if (error.status === 422) return ['The API rejected the request body. It should accept {"trip_id": "..."}.'];
      return ["The API raised an error. Its log on Laptop 1 shows the details."];
    default:
      return [];
  }
}

export default function ResultPanel({ run, apiUrl }) {
  const elapsed = useElapsed(run.state === "running");

  if (run.state === "idle") {
    return (
      <section className="panel">
        <h2>Result</h2>
        <p className="muted">
          Select a trip. The UI sends its ID to the API on Laptop 1, which predicts the route and scores it
          against reference data that stays on Laptop 1.
        </p>
      </section>
    );
  }

  if (run.state === "running") {
    return (
      <section className="panel">
        <h2>Trip {run.tripId}</h2>
        <p className="running-line">
          <span className="spinner" /> Waiting for the API… {(elapsed / 1000).toFixed(1)} s
        </p>
      </section>
    );
  }

  if (run.state === "cancelled") {
    return (
      <section className="panel">
        <h2>Trip {run.tripId}</h2>
        <p className="muted">Stopped before the API answered.</p>
      </section>
    );
  }

  if (run.state === "error") {
    const hints = hintsFor(run.error, apiUrl);
    return (
      <section className="panel">
        <h2>Trip {run.tripId}</h2>
        <div className="notice notice-error">
          <strong>Prediction failed.</strong>
          <p>{run.error.message}</p>
          {hints.length > 0 && (
            <ul>
              {hints.map((hint) => (
                <li key={hint}>{hint}</li>
              ))}
            </ul>
          )}
        </div>
      </section>
    );
  }

  const { result, roundTripMs } = run;
  const headline = headlineMetric(result.metrics);
  const otherMetrics = Object.entries(result.metrics).filter(([key]) => key !== headline?.key);

  return (
    <section className="panel">
      <h2>Trip {result.tripId}</h2>

      {headline && (
        <div className="headline">
          <span className="headline-value">{formatMetric(headline.key, headline.value)}</span>
          <span className="headline-label">{prettyLabel(headline.key)}</span>
        </div>
      )}

      <dl className="metrics">
        {otherMetrics.map(([key, value]) => (
          <div key={key} className="metric">
            <dt>{prettyLabel(key)}</dt>
            <dd>{formatMetric(key, value)}</dd>
          </div>
        ))}
        <div className="metric">
          <dt>GPS points drawn</dt>
          <dd>{result.gpsPoints.length.toLocaleString()}</dd>
        </div>
        <div className="metric">
          <dt>Route segments drawn</dt>
          <dd>{result.routeSegments.length.toLocaleString()}</dd>
        </div>
        <div className="metric">
          <dt>Inference time (API)</dt>
          <dd>{formatDuration(result.inferenceMs)}</dd>
        </div>
        <div className="metric">
          <dt>Round trip (UI)</dt>
          <dd>{formatDuration(roundTripMs)}</dd>
        </div>
      </dl>

      {result.warnings.length > 0 && (
        <div className="notice notice-warn">
          <strong>Check the response</strong>
          <ul>
            {result.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="muted small">Accuracy is computed by the API on Laptop 1. The reference route is never sent to this screen.</p>
    </section>
  );
}
