import { formatClock, formatDuration } from "../format.js";

export default function RequestLog({ entries, mock }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>API requests</h2>
        <span className="muted small">{mock ? "Demo data, not sent anywhere" : "Sent from this screen"}</span>
      </div>
      {entries.length === 0 ? (
        <p className="muted">No requests yet.</p>
      ) : (
        <div className="table-wrap">
          <table className="log">
            <thead>
              <tr>
                <th scope="col">Time</th>
                <th scope="col">Request</th>
                <th scope="col">Status</th>
                <th scope="col" className="num">Duration</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.id}>
                  <td className="mono">{formatClock(entry.at)}</td>
                  <td className="mono">
                    {entry.method} {entry.path}
                  </td>
                  <td>
                    <span className={`code ${entry.ok ? "code-ok" : "code-bad"}`}>{entry.status ?? "no response"}</span>
                  </td>
                  <td className="num mono">{formatDuration(entry.ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
