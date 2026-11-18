import { formatMetric, isPercentMetric, prettyLabel } from "../format.js";

function TripBadge({ result, running }) {
  if (running) return <span className="badge badge-running"><span className="spinner spinner-small" /> Running</span>;
  if (!result) return <span className="badge">Not run</span>;
  if (result.status === "error") return <span className="badge badge-error">Error</span>;
  if (result.headline) {
    return <span className="badge badge-ok">{formatMetric(result.headline.key, result.headline.value)}</span>;
  }
  return <span className="badge badge-ok">Done</span>;
}

function Summary({ trips, results }) {
  const done = trips.filter((trip) => results[trip.id]?.status === "ok");
  const failed = trips.filter((trip) => results[trip.id]?.status === "error").length;
  if (done.length === 0 && failed === 0) return null;

  const headlines = done.map((trip) => results[trip.id].headline).filter(Boolean);
  const sameKey = headlines.length > 0 && headlines.every((h) => h.key === headlines[0].key);
  let mean = null;
  if (sameKey && isPercentMetric(headlines[0].key, headlines[0].value)) {
    const value = headlines.reduce((sum, h) => sum + h.value, 0) / headlines.length;
    mean = { key: headlines[0].key, value };
  }

  return (
    <div className="summary">
      <div>
        <span className="summary-value">{done.length}</span>/{trips.length} trips predicted
        {failed > 0 && <span className="summary-failed"> · {failed} failed</span>}
      </div>
      {mean && (
        <div>
          Mean {prettyLabel(mean.key).toLowerCase()}:{" "}
          <span className="summary-value">{formatMetric(mean.key, mean.value)}</span>
        </div>
      )}
    </div>
  );
}

export default function TripList({ trips, results, selectedId, runningId, batch, disabled, onSelect, onRunAll, onCancel, onReload }) {
  return (
    <div className="trip-panel">
      <div className="panel-head">
        <h2>Evaluation trips</h2>
        <button type="button" className="btn btn-small btn-ghost" onClick={onReload} disabled={disabled || trips.state === "loading"}>
          Reload
        </button>
      </div>

      {trips.state === "loading" && (
        <p className="muted"><span className="spinner spinner-small" /> Loading trips from the API…</p>
      )}

      {trips.state === "error" && (
        <div className="notice notice-error">
          <strong>Couldn't load the trip list.</strong>
          <p>{trips.error.message}</p>
          <p>Fix the connection shown in the header, then press Reload.</p>
        </div>
      )}

      {trips.state === "ready" && trips.items.length === 0 && (
        <p className="muted">The API returned no trips.</p>
      )}

      {trips.state === "ready" && trips.items.length > 0 && (
        <>
          <p className="muted trip-hint">Select a trip to send it to the API for prediction.</p>
          <ul className="trip-list">
            {trips.items.map((trip, index) => (
              <li key={trip.id}>
                <button
                  type="button"
                  className={`trip ${trip.id === selectedId ? "trip-selected" : ""}`}
                  onClick={() => onSelect(trip.id)}
                  disabled={disabled}
                  aria-pressed={trip.id === selectedId}
                >
                  <span className="trip-index">{index + 1}</span>
                  <span className="trip-main">
                    <span className="trip-id">{trip.id}</span>
                    {trip.points != null && <span className="trip-meta">{trip.points.toLocaleString()} GPS points</span>}
                  </span>
                  <TripBadge result={results[trip.id]} running={trip.id === runningId} />
                </button>
              </li>
            ))}
          </ul>

          <div className="batch">
            {batch ? (
              <>
                <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={batch.total} aria-valuenow={batch.done}>
                  <div className="progress-bar" style={{ width: `${(batch.done / batch.total) * 100}%` }} />
                </div>
                <div className="batch-row">
                  <span>Running {Math.min(batch.done + 1, batch.total)} of {batch.total}…</span>
                  <button type="button" className="btn btn-small btn-ghost" onClick={onCancel}>Stop</button>
                </div>
              </>
            ) : (
              <button type="button" className="btn btn-block" onClick={onRunAll} disabled={disabled}>
                Run all {trips.items.length} trips
              </button>
            )}
          </div>

          <Summary trips={trips.items} results={results} />
        </>
      )}
    </div>
  );
}
