import { useEffect, useState } from "react";
import { DEFAULT_API_URL, USE_MOCK, normalizeBaseUrl } from "../config.js";

function StatusPill({ health, onRecheck }) {
  const label =
    health.state === "online"
      ? `API online · ${health.ms} ms${health.note ? ` · ${health.note}` : ""}`
      : health.state === "offline"
        ? "Can't reach API"
        : "Checking API…";
  return (
    <button
      type="button"
      className={`status-pill status-${health.state}`}
      onClick={onRecheck}
      title={health.detail || "Click to check again"}
    >
      <span className="status-dot" aria-hidden="true" />
      {label}
    </button>
  );
}

export default function Header({ apiUrl, health, onRecheck, onApiUrlChange, onApiUrlReset }) {
  const [draft, setDraft] = useState(apiUrl);
  useEffect(() => setDraft(apiUrl), [apiUrl]);

  const isDefault = normalizeBaseUrl(apiUrl) === normalizeBaseUrl(DEFAULT_API_URL);

  function submit(event) {
    event.preventDefault();
    onApiUrlChange(draft);
  }

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 32 32" width="28" height="28">
            <rect width="32" height="32" rx="7" fill="currentColor" />
            <path d="M7 24 L13 14 L19 18 L25 8" stroke="#fff" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
        <div>
          <h1>PathSnap</h1>
          <p className="brand-sub">Team 8 · Evaluator interface (Laptop 2)</p>
        </div>
      </div>

      <div className="topbar-controls">
        {USE_MOCK ? (
          <span className="status-pill status-mock">
            <span className="status-dot" aria-hidden="true" />
            Demo data (no API)
          </span>
        ) : (
          <>
            <StatusPill health={health} onRecheck={onRecheck} />
            <form className="api-form" onSubmit={submit}>
              <label htmlFor="api-url">API address</label>
              <input
                id="api-url"
                type="text"
                spellCheck="false"
                autoComplete="off"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="http://192.168.x.x:8000"
              />
              <button type="submit" className="btn btn-small" disabled={normalizeBaseUrl(draft) === apiUrl}>
                Connect
              </button>
              {!isDefault && (
                <button type="button" className="btn btn-small btn-ghost" onClick={onApiUrlReset}>
                  Reset
                </button>
              )}
            </form>
          </>
        )}
      </div>
    </header>
  );
}
