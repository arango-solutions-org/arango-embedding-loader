import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Eye, Pause, Play, Plus, RefreshCw, Sparkles, Trash2 } from "lucide-react";
import { useState } from "react";

import { api, unwrap, type Job, type VectorIndexSpec } from "../api/client";
import { JobProgress } from "../components/JobProgress";
import { ModelPicker } from "../components/ModelPicker";
import {
  Badge,
  Button,
  Card,
  Checkbox,
  EmptyState,
  ErrorPanel,
  Field,
  Input,
  ProgressBar,
  Select,
  Spinner,
} from "../components/ui";
import { useCollections, useProfileStatus } from "../lib/queries";
import { buildSpecs, DEFAULT_MODEL_CHOICE, formatNumber, type ModelChoice } from "../lib/spec";

function ProfileForm({ initialName, onDone }: { initialName?: string; onDone: (name: string) => void }) {
  const qc = useQueryClient();
  const [name, setName] = useState(initialName ?? "");
  const [source, setSource] = useState("text");
  const [targetField, setTargetField] = useState("embedding");
  const [keyField, setKeyField] = useState("id");
  const [choice, setChoice] = useState<ModelChoice>(DEFAULT_MODEL_CHOICE);
  const [vectorIndex, setVectorIndex] = useState(true);
  const [metric, setMetric] = useState<VectorIndexSpec["metric"]>("cosine");
  const [autoWatch, setAutoWatch] = useState(true);
  const [interval, setIntervalS] = useState("10");

  const save = useMutation({
    mutationFn: () =>
      unwrap(
        api.PUT("/api/v1/collections/{name}/profile", {
          params: { path: { name } },
          body: {
            embeddings: buildSpecs([{ path: source, target: targetField }], null, choice, "skip"),
            key_field: keyField || null,
            on_duplicate: "update",
            batch_size: 64,
            vector_index: vectorIndex ? ({ create: true, metric } as VectorIndexSpec) : null,
            auto_watch: autoWatch,
            watch_interval_s: Number(interval) || 10,
          },
        }),
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["collections"] });
      qc.invalidateQueries({ queryKey: ["profile"] });
      onDone(name);
    },
  });

  return (
    <Card title="Set up an auto-embedding collection">
      <div className="flex flex-col gap-5">
        <p className="text-sm text-muted">
          Attach an embedding profile to a collection. Every load into it is embedded automatically,
          and the optional watcher embeds documents inserted by any other client.
        </p>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Collection" hint="Created if it doesn't exist.">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="pubmed" />
          </Field>
          <Field label="Field to embed" hint="Dot-path, e.g. text or content.body">
            <Input className="font-mono" value={source} onChange={(e) => setSource(e.target.value)} />
          </Field>
          <Field label="Store vector in">
            <Input className="font-mono" value={targetField} onChange={(e) => setTargetField(e.target.value)} />
          </Field>
          <Field label="Key field" hint="Optional: becomes the document _key.">
            <Input className="font-mono" value={keyField} onChange={(e) => setKeyField(e.target.value)} />
          </Field>
        </div>
        <div>
          <h3 className="mb-3 text-sm font-semibold">Embedding model</h3>
          <ModelPicker value={choice} onChange={setChoice} />
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          <Checkbox checked={vectorIndex} onChange={setVectorIndex} label="Create vector index" hint="Built once vectors exist." />
          <Field label="Metric">
            <Select value={metric} disabled={!vectorIndex} onChange={(e) => setMetric(e.target.value as VectorIndexSpec["metric"])}>
              <option value="cosine">cosine</option>
              <option value="l2">l2</option>
              <option value="innerProduct">innerProduct</option>
            </Select>
          </Field>
          <div />
          <Checkbox
            checked={autoWatch}
            onChange={setAutoWatch}
            label="Watch for new documents"
            hint="Background watcher embeds docs inserted by other clients."
          />
          <Field label="Watch interval (seconds)">
            <Input type="number" min={1} value={interval} disabled={!autoWatch} onChange={(e) => setIntervalS(e.target.value)} />
          </Field>
        </div>
        <p className="text-xs text-muted">
          API keys entered in provider options are not stored with the profile; configure them on
          the server (e.g. OPENAI_API_KEY) for watchers and profile-based loads.
        </p>
        <ErrorPanel error={save.error} />
        <div className="flex gap-2">
          <Button onClick={() => save.mutate()} loading={save.isPending} disabled={!name || !source || !targetField}>
            Save profile
          </Button>
        </div>
      </div>
    </Card>
  );
}

function ProfileDetail({ name }: { name: string }) {
  const qc = useQueryClient();
  const status = useProfileStatus(name);
  const [job, setJob] = useState<Job | null>(null);
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["profile"] });
    qc.invalidateQueries({ queryKey: ["collections"] });
  };

  const watcher = useMutation({
    mutationFn: (action: "start" | "stop") =>
      unwrap(
        action === "start"
          ? api.POST("/api/v1/collections/{name}/profile/watcher/start", { params: { path: { name } } })
          : api.POST("/api/v1/collections/{name}/profile/watcher/stop", { params: { path: { name } } }),
      ),
    onSuccess: refresh,
  });
  const reembed = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/api/v1/collections/{name}/embed", {
          params: { path: { name } },
          body: { only_missing: true, filter: {}, analyze: false },
        }),
      ),
    onSuccess: (j) => {
      setJob(j);
      refresh();
    },
  });
  const remove = useMutation({
    mutationFn: () =>
      unwrap(api.DELETE("/api/v1/collections/{name}/profile", { params: { path: { name } } })),
    onSuccess: refresh,
  });

  if (status.isLoading) return <Spinner label="Loading profile…" />;
  if (status.error) {
    return (
      <EmptyState>
        <b>{name}</b> has no auto-embed profile yet.
      </EmptyState>
    );
  }
  const s = status.data!;
  const spec = s.profile.embeddings[0];
  const target = spec.target_field ?? "embedding";
  const embedded = s.embedded_documents[target] ?? 0;
  const vectorIndexes = s.indexes.filter((i) => (i as { type?: string }).type === "vector");

  return (
    <div className="flex flex-col gap-5">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-md bg-panel p-4">
          <div className="text-xs text-muted">Embedded</div>
          <div className="text-2xl font-semibold">
            {formatNumber(embedded)} <span className="text-base font-normal text-muted">/ {formatNumber(s.total_documents)}</span>
          </div>
          <ProgressBar value={s.total_documents ? embedded / s.total_documents : 0} />
        </div>
        <div className="rounded-md bg-panel p-4">
          <div className="text-xs text-muted">Pending</div>
          <div className="text-2xl font-semibold">{s.pending_documents < 0 ? "?" : formatNumber(s.pending_documents)}</div>
          <div className="text-xs text-muted">not yet processed by the loader</div>
        </div>
        <div className="rounded-md bg-panel p-4">
          <div className="flex items-center gap-2 text-xs text-muted">
            Watcher {s.watcher.running ? <Badge tone="green">running</Badge> : <Badge>stopped</Badge>}
          </div>
          <div className="mt-1 text-sm">
            {formatNumber(s.watcher.embedded)} embedded · {formatNumber(s.watcher.failed)} failed
          </div>
          {s.watcher.last_error && <div className="mt-1 text-xs text-danger">{s.watcher.last_error}</div>}
        </div>
      </div>

      <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
        {s.profile.embeddings.map((e) => (
          <div key={e.target_field ?? e.source_field ?? ""} className="contents">
            <dt className="text-muted">Embeds</dt>
            <dd>
              <span className="font-mono">{e.source_field ?? e.template}</span> →{" "}
              <span className="font-mono">{e.target_field}</span> with{" "}
              <b>{e.preset ? `${e.preset} preset` : `${e.provider} / ${e.model ?? "default"}`}</b>
            </dd>
          </div>
        ))}
        <dt className="text-muted">Key field</dt>
        <dd className="font-mono">{s.profile.key_field ?? "auto"}</dd>
        <dt className="text-muted">Vector index</dt>
        <dd>
          {vectorIndexes.length
            ? vectorIndexes.map((i) => {
                const idx = i as { name?: string; params?: { metric?: string; dimension?: number } };
                return (
                  <Badge key={idx.name} tone="green">
                    {idx.name} · {idx.params?.metric} · {idx.params?.dimension}d
                  </Badge>
                );
              })
            : s.profile.vector_index?.create
              ? "created after the first load"
              : "none"}
        </dd>
      </dl>

      <div className="flex flex-wrap gap-2">
        {s.watcher.running ? (
          <Button variant="secondary" onClick={() => watcher.mutate("stop")} loading={watcher.isPending}>
            <Pause className="size-4" /> Stop watcher
          </Button>
        ) : (
          <Button variant="secondary" onClick={() => watcher.mutate("start")} loading={watcher.isPending}>
            <Play className="size-4" /> Start watcher
          </Button>
        )}
        <Button variant="secondary" onClick={() => reembed.mutate()} loading={reembed.isPending}>
          <Sparkles className="size-4" /> Embed missing now
        </Button>
        <Button variant="danger" className="ml-auto" onClick={() => remove.mutate()} loading={remove.isPending}>
          <Trash2 className="size-4" /> Remove profile
        </Button>
      </div>
      <ErrorPanel error={watcher.error ?? reembed.error ?? remove.error} />
      {job && <JobProgress initial={job} />}
    </div>
  );
}

export function Collections() {
  const collections = useCollections();
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Collections & auto-embed profiles</h1>
          <p className="mt-1 text-sm text-muted">
            Set up a collection once; documents loaded into it get embeddings automatically.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => collections.refetch()}>
            <RefreshCw className="size-4" /> Refresh
          </Button>
          <Button onClick={() => { setCreating(true); setSelected(null); }}>
            <Plus className="size-4" /> New auto-embed collection
          </Button>
        </div>
      </div>

      {creating && (
        <ProfileForm
          initialName={selected ?? undefined}
          onDone={(name) => {
            setCreating(false);
            setSelected(name);
          }}
        />
      )}

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <Card title="Collections">
          <ErrorPanel error={collections.error} />
          {collections.isLoading && <Spinner label="Loading…" />}
          {collections.data?.length === 0 && <EmptyState>No collections yet.</EmptyState>}
          <ul className="-mx-2 flex flex-col">
            {collections.data?.map((c) => (
              <li key={c.name}>
                <button
                  type="button"
                  onClick={() => { setSelected(c.name); setCreating(false); }}
                  className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-2 text-left text-sm ${
                    selected === c.name ? "bg-arango-50 font-medium text-arango" : "hover:bg-panel"
                  }`}
                >
                  <span className="truncate">{c.name}</span>
                  <span className="flex shrink-0 items-center gap-2">
                    {c.has_profile && <Badge tone="green">auto-embed</Badge>}
                    <span className="text-xs text-muted">{formatNumber(c.count)}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </Card>
        <Card
          title={selected ?? "Select a collection"}
          actions={
            selected && (
              <Button variant="secondary" onClick={() => setCreating(true)}>
                <Eye className="size-4" /> Edit profile
              </Button>
            )
          }
        >
          {selected ? <ProfileDetail key={selected} name={selected} /> : <EmptyState>Pick a collection to see its embedding status.</EmptyState>}
        </Card>
      </div>
    </div>
  );
}
