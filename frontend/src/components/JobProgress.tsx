import { useEffect, useState } from "react";

import { api, apiBase, unwrap, type Job } from "../api/client";
import { formatBytes, formatDuration, formatNumber } from "../lib/spec";
import { Badge, Button, ProgressBar } from "./ui";

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

export function statusTone(status: Job["status"]): "green" | "red" | "amber" | "gray" {
  if (status === "succeeded") return "green";
  if (status === "failed") return "red";
  if (status === "cancelled") return "amber";
  return "gray";
}

/** Subscribes to a job's SSE stream and keeps the latest snapshot. */
export function useJob(initial: Job): Job {
  const [job, setJob] = useState(initial);
  useEffect(() => {
    if (TERMINAL.has(initial.status)) return;
    const source = new EventSource(`${apiBase}/api/v1/jobs/${initial.id}/events`);
    source.addEventListener("job", (event) => {
      const next = JSON.parse((event as MessageEvent).data) as Job;
      setJob(next);
      if (TERMINAL.has(next.status)) source.close();
    });
    source.onerror = () => source.close();
    return () => source.close();
  }, [initial.id, initial.status]);
  return job;
}

export function JobProgress({ initial }: { initial: Job }) {
  const job = useJob(initial);
  const running = !TERMINAL.has(job.status);
  const c = job.counts;
  const vi = job.result?.vector_index as { status?: string; error?: string } | undefined;
  const samples = job.result?.samples as unknown[] | undefined;
  const analysis = job.result?.analysis as Analysis | undefined;
  const prescan = job.result?.prescan as { status: string; reason?: string } | undefined;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Badge tone={statusTone(job.status)}>{job.status}</Badge>
        <span className="text-sm text-muted">
          Job <span className="font-mono">{job.id}</span> → <b>{job.collection}</b> ·{" "}
          {formatDuration(job.started_at, job.finished_at)}
        </span>
        {running && (
          <Button
            variant="danger"
            className="ml-auto"
            onClick={() =>
              unwrap(api.DELETE("/api/v1/jobs/{job_id}", { params: { path: { job_id: job.id } } }))
            }
          >
            Cancel
          </Button>
        )}
      </div>
      <ProgressBar value={job.status === "succeeded" ? 1 : job.progress} />
      <dl className="grid grid-cols-4 gap-3 text-sm sm:grid-cols-8">
        {(
          [
            ["Read", c.read],
            ["Embedded", c.embedded],
            ["Written", c.written],
            ["Updated", c.updated],
            ["Skipped", c.skipped],
            ["Failed", c.failed],
            ["Over limit", c.over_limit],
            ["Chunks", c.chunks],
          ] as const
        ).map(([label, value]) => (
          <div key={label} className="rounded-md bg-panel px-3 py-2">
            <dt className="text-xs text-muted">{label}</dt>
            <dd className={`text-lg font-semibold ${label === "Failed" && value ? "text-danger" : ""}`}>
              {formatNumber(value)}
            </dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-muted">
        {job.total != null && <>Total {formatNumber(job.total)} documents · </>}
        {job.bytes_total != null && (
          <>
            {formatBytes(job.bytes_read)} / {formatBytes(job.bytes_total)}
          </>
        )}
      </p>
      {job.message && (
        <p className={`text-sm ${job.status === "failed" ? "text-danger" : ""}`}>{job.message}</p>
      )}
      {vi && (
        <p className="text-sm">
          Vector index:{" "}
          <Badge tone={vi.status === "failed" ? "red" : "green"}>{vi.status}</Badge>{" "}
          {vi.error && <span className="text-danger">{vi.error}</span>}
        </p>
      )}
      {job.errors.length > 0 && (
        <details className="rounded-md border border-line">
          <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
            {job.errors.length} document error(s)
            {job.counts.failed > job.errors.length && ` (showing first ${job.errors.length})`}
          </summary>
          <table className="w-full text-xs">
            <tbody>
              {job.errors.map((e, i) => (
                <tr key={i} className="border-t border-line">
                  <td className="px-3 py-1.5 font-mono">{e.index ?? ""}</td>
                  <td className="px-3 py-1.5 font-mono">{e.key ?? ""}</td>
                  <td className="px-3 py-1.5 text-danger">{e.error}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
      {prescan?.status === "skipped" && (
        <p className="text-xs text-muted">Length pre-scan skipped: {prescan.reason}</p>
      )}
      {analysis && <AnalysisReport analysis={analysis} />}
      {samples && samples.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-medium">Dry-run output (vectors truncated)</h3>
          <pre className="max-h-80 overflow-auto rounded-md bg-panel p-3 font-mono text-xs">
            {JSON.stringify(samples, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

interface FieldAnalysis {
  provider: string;
  model: string;
  long_input: string;
  token_count: "exact" | "estimated";
  max_tokens: number | null;
  documents: number;
  missing_text: number;
  over_limit?: number;
  over_limit_pct?: number;
  tokens?: { median: number; p90: number; p99: number; max: number };
  longest?: { document: string; tokens: number }[];
  note?: string;
}

interface Analysis {
  documents: number;
  over_limit_documents: number;
  fields: Record<string, FieldAnalysis>;
}

function AnalysisReport({ analysis }: { analysis: Analysis }) {
  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-medium">
        Text lengths: {formatNumber(analysis.over_limit_documents)} of{" "}
        {formatNumber(analysis.documents)} documents over the model limit
      </h3>
      {Object.entries(analysis.fields).map(([target, f]) => (
        <div key={target} className="rounded-md bg-panel p-3 text-sm">
          <div className="mb-2">
            <b>{target}</b>{" "}
            <span className="text-muted">
              {f.provider}/{f.model}
              {f.max_tokens != null && <> · limit {formatNumber(f.max_tokens)} tokens</>}
              {" "}· long_input: {f.long_input}
              {f.token_count === "estimated" && <> · token counts estimated</>}
            </span>
          </div>
          {f.note && <p className="text-muted">{f.note}</p>}
          {f.tokens && (
            <dl className="grid grid-cols-3 gap-2 sm:grid-cols-6">
              {(
                [
                  ["Over limit", `${formatNumber(f.over_limit ?? 0)} (${f.over_limit_pct ?? 0}%)`],
                  ["Median", f.tokens.median],
                  ["p90", f.tokens.p90],
                  ["p99", f.tokens.p99],
                  ["Max", f.tokens.max],
                  ["No text", f.missing_text],
                ] as const
              ).map(([label, value]) => (
                <div key={label}>
                  <dt className="text-xs text-muted">{label}</dt>
                  <dd className="font-semibold">
                    {typeof value === "number" ? formatNumber(value) : value}
                  </dd>
                </div>
              ))}
            </dl>
          )}
          {f.longest && f.longest.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs text-muted">
                Longest documents over the limit
              </summary>
              <ul className="mt-1 font-mono text-xs">
                {f.longest.map((d) => (
                  <li key={d.document}>
                    {d.document}: {formatNumber(d.tokens)} tokens
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}
