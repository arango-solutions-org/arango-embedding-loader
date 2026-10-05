import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Database, X, XCircle } from "lucide-react";
import { useState } from "react";

import { setConnection, useConnection, type ConnectionOverride } from "../lib/connection";
import { useConnectionInfo } from "../lib/queries";
import { Badge, Button, Checkbox, Field, Input, Spinner } from "./ui";

export function ConnectionStatus({ onOpen }: { onOpen: () => void }) {
  const { data, isLoading } = useConnectionInfo();
  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex items-center gap-2 rounded-md border border-line bg-white px-3 py-1.5 text-sm hover:bg-panel"
    >
      <Database className="size-4 text-arango" aria-hidden />
      {isLoading ? (
        <Spinner />
      ) : data?.connected ? (
        <>
          <span className="font-medium">{data.database}</span>
          <span className="text-muted">@ {String(data.connection.endpoint)}</span>
          <CheckCircle2 className="size-4 text-arango" aria-label="connected" />
        </>
      ) : (
        <>
          <span>Not connected</span>
          <XCircle className="size-4 text-danger" aria-label="not connected" />
        </>
      )}
    </button>
  );
}

export function ConnectionPanel({ onClose }: { onClose: () => void }) {
  const conn = useConnection();
  const [draft, setDraft] = useState<ConnectionOverride>(conn);
  const qc = useQueryClient();
  const info = useConnectionInfo();
  const set = (patch: Partial<ConnectionOverride>) => setDraft((d) => ({ ...d, ...patch }));

  const apply = () => {
    setConnection(draft);
    qc.invalidateQueries();
  };

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/30" onClick={onClose}>
      <aside
        className="flex h-full w-full max-w-md flex-col gap-5 overflow-y-auto bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
        aria-label="ArangoDB connection"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">ArangoDB connection</h2>
          <Button variant="ghost" onClick={onClose} aria-label="Close">
            <X className="size-4" />
          </Button>
        </div>
        <Checkbox
          checked={draft.useServerDefault}
          onChange={(v) => set({ useServerDefault: v })}
          label="Use server default connection"
          hint="Configured on the API server via ARANGO_* environment variables."
        />
        <fieldset className="flex flex-col gap-4" disabled={draft.useServerDefault}>
          <Field label="Endpoint" hint="Blank fields fall back to the server default.">
            <Input
              placeholder="https://my-deployment.arangodb.cloud:8529"
              value={draft.endpoint}
              onChange={(e) => set({ endpoint: e.target.value })}
            />
          </Field>
          <Field label="Database">
            <Input
              placeholder="_system"
              value={draft.database}
              onChange={(e) => set({ database: e.target.value })}
            />
          </Field>
          <Field label="Username">
            <Input
              placeholder="root"
              value={draft.username}
              autoComplete="username"
              onChange={(e) => set({ username: e.target.value })}
            />
          </Field>
          <Field label="Password" hint="Kept in this browser tab only (sessionStorage).">
            <Input
              type="password"
              autoComplete="current-password"
              value={draft.password}
              onChange={(e) => set({ password: e.target.value })}
            />
          </Field>
        </fieldset>
        <div className="flex gap-2">
          <Button onClick={apply}>Save &amp; test</Button>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="rounded-md border border-line bg-panel p-4 text-sm">
          {info.isFetching ? (
            <Spinner label="Testing connection…" />
          ) : info.data?.connected ? (
            <div className="flex flex-col gap-1">
              <span className="flex items-center gap-2 font-medium text-arango">
                <CheckCircle2 className="size-4" /> Connected
              </span>
              <span>
                ArangoDB {info.data.server_version} · database <b>{info.data.database}</b>
              </span>
              <span>
                Vector indexes:{" "}
                {info.data.vector_index_supported ? (
                  <Badge tone="green">supported</Badge>
                ) : (
                  <Badge tone="amber">requires 3.12.4+</Badge>
                )}
              </span>
            </div>
          ) : (
            <span className="flex items-start gap-2 text-danger">
              <XCircle className="mt-0.5 size-4 shrink-0" />
              {info.data?.error ?? (info.error as Error | null)?.message ?? "Not connected"}
            </span>
          )}
        </div>
      </aside>
    </div>
  );
}
