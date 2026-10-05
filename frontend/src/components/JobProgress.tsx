import { useEffect, useState } from "react";

import { api, unwrap, type Job } from "../api/client";
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
    const source = new EventSource(`/api/v1/jobs/${initial.id}/events`);
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
      <dl className="grid grid-cols-3 gap-3 text-sm sm:grid-cols-6">
        {(
          [
            ["Read", c.read],
            ["Embedded", c.embedded],
            ["Written", c.written],
            ["Updated", c.updated],
            ["Skipped", c.skipped],
            ["Failed", c.failed],
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
