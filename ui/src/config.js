// Runtime settings. The API address comes from (in order): the value saved in
// the page header, VITE_API_URL in .env.local, then localhost.

const STORAGE_KEY = "pathsnap.apiUrl";

export const DEFAULT_API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const USE_MOCK = String(import.meta.env.VITE_USE_MOCK).toLowerCase() === "true";
export const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS) || 180000;
export const HEALTH_TIMEOUT_MS = 5000;

export function normalizeBaseUrl(url) {
  const trimmed = String(url ?? "").trim().replace(/\/+$/, "");
  if (!trimmed) return DEFAULT_API_URL.replace(/\/+$/, "");
  return /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
}

export function loadApiUrl() {
  try {
    return normalizeBaseUrl(window.localStorage.getItem(STORAGE_KEY) || DEFAULT_API_URL);
  } catch {
    return normalizeBaseUrl(DEFAULT_API_URL);
  }
}

export function saveApiUrl(url) {
  try {
    window.localStorage.setItem(STORAGE_KEY, url);
  } catch {
    // Storage can be unavailable (private window); the address still applies for this session.
  }
}

export function clearApiUrl() {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing stored, nothing to clear.
  }
}
