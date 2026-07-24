// lib/api.js
// ----------
// Thin fetch wrapper shared by every page. One place to change the base
// URL, one place that throws a readable error when the backend returns
// a non-2xx status (FastAPI's error body is always {"detail": "..."}).
//
// Every list endpoint returns {"data": [...], "meta": {...}}.
// Every single-item endpoint returns {"data": {...}}.
// This client returns the parsed JSON body as-is (i.e. still wrapped in
// "data") -- callers unwrap it themselves so it's obvious which shape
// they're getting back.

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  if (res.status === 204) return null;

  const body = await res.json().catch(() => null);

  if (!res.ok) {
    const message = body?.detail
      ? (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))
      : `Request failed (${res.status})`;
    throw new Error(message);
  }

  return body;
}

export const api = {
  get: (path) => request(path),
  post: (path, data) => request(path, { method: 'POST', body: JSON.stringify(data) }),
  put: (path, data) => request(path, { method: 'PUT', body: JSON.stringify(data) }),
  patch: (path, data) => request(path, { method: 'PATCH', body: JSON.stringify(data) }),
  delete: (path) => request(path, { method: 'DELETE' }),
};