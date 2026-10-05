import { useQuery } from "@tanstack/react-query";
import { Fragment, useState } from "react";

import { api, unwrap } from "../api/client";
import { JobProgress, statusTone } from "../components/JobProgress";
import { Badge, Card, EmptyState, ErrorPanel, ProgressBar } from "../components/ui";
import { formatDuration, formatNumber } from "../lib/spec";

export function Jobs() {
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => unwrap(api.GET("/api/v1/jobs")),
    refetchInterval: 3000,
  });
  const [open, setOpen] = useState<string | null>(null);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Jobs</h1>
        <p className="mt-1 text-sm text-muted">Recent load and re-embed jobs on this server.</p>
      </div>
      <Card>
        <ErrorPanel error={jobs.error} />
        {jobs.data?.length === 0 && <EmptyState>No jobs yet.</EmptyState>}
        {!!jobs.data?.length && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Status</th>
                  <th className="py-2 pr-4 font-medium">Kind</th>
                  <th className="py-2 pr-4 font-medium">Collection</th>
                  <th className="w-40 py-2 pr-4 font-medium">Progress</th>
                  <th className="py-2 pr-4 font-medium">Written</th>
                  <th className="py-2 pr-4 font-medium">Failed</th>
                  <th className="py-2 pr-4 font-medium">Duration</th>
                  <th className="py-2 font-medium">Started</th>
                </tr>
              </thead>
              <tbody>
                {jobs.data.map((j) => (
                  <Fragment key={j.id}>
                    <tr
                      className="cursor-pointer border-t border-line hover:bg-panel"
                      onClick={() => setOpen(open === j.id ? null : j.id)}
                    >
                      <td className="py-2 pr-4"><Badge tone={statusTone(j.status)}>{j.status}</Badge></td>
                      <td className="py-2 pr-4">{j.kind.replace("_", " ")}</td>
                      <td className="py-2 pr-4 font-medium">{j.database}/{j.collection}</td>
                      <td className="py-2 pr-4"><ProgressBar value={j.status === "succeeded" ? 1 : j.progress} /></td>
                      <td className="py-2 pr-4">{formatNumber(j.counts.written)}</td>
                      <td className={`py-2 pr-4 ${j.counts.failed ? "text-danger" : ""}`}>{formatNumber(j.counts.failed)}</td>
                      <td className="py-2 pr-4">{formatDuration(j.started_at, j.finished_at)}</td>
                      <td className="py-2 text-muted">{new Date(j.created_at).toLocaleTimeString()}</td>
                    </tr>
                    {open === j.id && (
                      <tr>
                        <td colSpan={8} className="bg-white px-2 py-4">
                          <JobProgress initial={j} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
