import { useSyncExternalStore } from "react";

/**
 * The ArangoDB database the app works in, sent to the API as `X-Arango-Database`.
 * Credentials are never handled here: on the Arango Platform the gateway forwards the signed-in
 * user's token, and elsewhere the server uses its own.
 */

const KEY = "ael.database";

/** The database of a database-scoped platform mount (`/_service/uds/_db/<db>/...`). */
export function mountDatabase(pathname: string): string {
  const match = /\/_service\/uds\/_db\/([^/]+)\//.exec(`${pathname}/`);
  return match ? decodeURIComponent(match[1]) : "";
}

let current: string = load();
const listeners = new Set<() => void>();

function load(): string {
  try {
    const stored = localStorage.getItem(KEY);
    if (stored) return stored;
  } catch {
    /* storage unavailable */
  }
  return typeof window === "undefined" ? "" : mountDatabase(window.location.pathname);
}

/** The chosen database, or "" for the server's default. */
export function getDatabase(): string {
  return current;
}

export function setDatabase(next: string): void {
  current = next;
  try {
    if (next) localStorage.setItem(KEY, next);
    else localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable: keep in memory only */
  }
  listeners.forEach((fn) => fn());
}

export function useDatabase(): string {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    getDatabase,
    getDatabase,
  );
}

export function databaseHeaders(database: string = current): Record<string, string> {
  return database ? { "X-Arango-Database": database } : {};
}
