import { useSyncExternalStore } from "react";

/** Per-tab ArangoDB connection override, sent to the API as X-Arango-* headers. */
export interface ConnectionOverride {
  useServerDefault: boolean;
  endpoint: string;
  database: string;
  username: string;
  password: string;
}

const KEY = "ael.connection";
const EMPTY: ConnectionOverride = {
  useServerDefault: true,
  endpoint: "",
  database: "",
  username: "",
  password: "",
};

let current: ConnectionOverride = load();
const listeners = new Set<() => void>();

function load(): ConnectionOverride {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? { ...EMPTY, ...JSON.parse(raw) } : EMPTY;
  } catch {
    return EMPTY;
  }
}

export function getConnection(): ConnectionOverride {
  return current;
}

export function setConnection(next: ConnectionOverride): void {
  current = next;
  try {
    sessionStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* storage unavailable: keep in memory only */
  }
  listeners.forEach((fn) => fn());
}

export function useConnection(): ConnectionOverride {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    getConnection,
    getConnection,
  );
}

export function connectionHeaders(conn: ConnectionOverride = current): Record<string, string> {
  if (conn.useServerDefault) return {};
  const headers: Record<string, string> = {};
  if (conn.endpoint) headers["X-Arango-Endpoint"] = conn.endpoint;
  if (conn.database) headers["X-Arango-Database"] = conn.database;
  if (conn.username) headers["X-Arango-Username"] = conn.username;
  if (conn.password) headers["X-Arango-Password"] = conn.password;
  return headers;
}

/** Stable key so React Query refetches when the connection changes (no secrets). */
export function connectionKey(conn: ConnectionOverride): string {
  return conn.useServerDefault ? "default" : `${conn.endpoint}|${conn.database}|${conn.username}`;
}
