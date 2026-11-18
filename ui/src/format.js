// Display helpers for metrics and durations.

const PERCENT_KEYS = /(acc|precision|recall|f1|continuity|ratio|rate|score|overlap|iou|match)/i;
const HEADLINE_KEYS = ["accuracy", "point_accuracy", "top1_accuracy", "observation_accuracy", "route_accuracy"];

export function isPercentMetric(key, value) {
  return typeof value === "number" && PERCENT_KEYS.test(key) && value >= 0 && value <= 100;
}

export function formatMetric(key, value) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (isPercentMetric(key, value)) {
      // Fractions (0–1) are shown as percentages; values above 1 are assumed to be percentages already.
      return `${(value <= 1 ? value * 100 : value).toFixed(1)}%`;
    }
    if (/_ms$/i.test(key)) return formatDuration(value);
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(3);
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (value == null) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function prettyLabel(key) {
  return String(key)
    .replace(/_ms$/i, "")
    .replace(/_/g, " ")
    .replace(/\btop ?1\b/gi, "Top-1")
    .replace(/\b(f1|gps|hmm|id|api|iou)\b/gi, (word) => word.toUpperCase())
    .replace(/^n /i, "Number of ")
    .replace(/^./, (first) => first.toUpperCase());
}

export function headlineMetric(metrics) {
  if (!metrics) return null;
  for (const key of HEADLINE_KEYS) {
    if (typeof metrics[key] === "number") return { key, value: metrics[key] };
  }
  const first = Object.entries(metrics).find(([, value]) => typeof value === "number");
  return first ? { key: first[0], value: first[1] } : null;
}

export function formatDuration(ms) {
  if (ms == null || !Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}

export function formatClock(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
