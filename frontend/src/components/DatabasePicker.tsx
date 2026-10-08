import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Database, User, XCircle } from "lucide-react";
import { useEffect } from "react";

import { setDatabase, useDatabase } from "../lib/database";
import { useConnectionInfo, useDatabases } from "../lib/queries";
import { Select, Spinner } from "./ui";

/** Who the app is signed in as, and which database it works in. */
export function DatabasePicker() {
  const info = useConnectionInfo();
  const databases = useDatabases();
  const chosen = useDatabase();
  const qc = useQueryClient();
  const choose = (name: string) => {
    setDatabase(name);
    qc.invalidateQueries();
  };

  const accessible = databases.data ?? [];
  const connected = !!info.data?.connected;
  const current = info.data?.database ?? chosen;

  // Users often can't open the server's default database (usually _system). Until they pick
  // one, switch to the first database they can access so the app connects on its own.
  const fallback = !info.isLoading && !connected && !chosen && accessible.length > 0;
  useEffect(() => {
    if (fallback && !accessible.includes(current)) choose(accessible[0]);
  });

  if (info.isLoading) return <Spinner />;

  const options = accessible.includes(current) || !current ? accessible : [current, ...accessible];
  const picker = options.length > 0 && (
    <label className="flex items-center gap-1.5">
      <Database className="size-4 text-arango" aria-hidden />
      <span className="sr-only">Database</span>
      <Select
        value={current}
        onChange={(e) => choose(e.target.value)}
        className="w-auto py-1"
        aria-label="Database"
      >
        {options.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </Select>
    </label>
  );

  if (!connected) {
    return (
      <div className="flex items-center gap-3 text-sm">
        {picker}
        <span className="flex items-center gap-2 text-danger" role="status">
          <XCircle className="size-4 shrink-0" aria-hidden />
          {info.data?.error ?? (info.error as Error | null)?.message ?? "Not connected"}
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 text-sm">
      {info.data?.user && (
        <span className="flex items-center gap-1.5 text-muted" title="Signed in to ArangoDB as">
          <User className="size-4" aria-hidden />
          {info.data.user}
        </span>
      )}
      {picker || (
        <span className="flex items-center gap-1.5 font-medium">
          <Database className="size-4 text-arango" aria-hidden />
          {current}
        </span>
      )}
      <CheckCircle2
        className="size-4 text-arango"
        aria-label={`connected, ArangoDB ${info.data?.server_version ?? ""}`}
      />
    </div>
  );
}
