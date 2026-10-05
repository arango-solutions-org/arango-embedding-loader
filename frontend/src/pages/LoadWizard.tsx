import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, ClipboardPaste, Database, FileUp, Link2, Upload } from "lucide-react";
import { useMemo, useRef, useState, type DragEvent, type ReactNode } from "react";

import {
  api,
  postForm,
  unwrap,
  type Job,
  type PreviewResponse,
  type VectorIndexSpec,
} from "../api/client";
import { FieldLabeler } from "../components/FieldLabeler";
import { JobProgress } from "../components/JobProgress";
import { ModelPicker } from "../components/ModelPicker";
import {
  Badge,
  Button,
  Card,
  Checkbox,
  Code,
  EmptyState,
  ErrorPanel,
  Field,
  Input,
  Segmented,
  Select,
  Textarea,
} from "../components/ui";
import { useCollections } from "../lib/queries";
import {
  buildSpecs,
  DEFAULT_MODEL_CHOICE,
  formatBytes,
  formatNumber,
  vectorIndexSpec,
  type LabeledField,
  type ModelChoice,
  type OnMissing,
} from "../lib/spec";

type SourceMode = "file" | "paste" | "url" | "collection";
type OnDuplicate = "error" | "update" | "replace" | "ignore";

const STEPS = ["Source", "Label fields", "Embedding model", "Target", "Run"] as const;

function Stepper({ step, onJump }: { step: number; onJump: (s: number) => void }) {
  return (
    <ol className="flex flex-wrap items-center gap-2 text-sm">
      {STEPS.map((label, i) => {
        const state = i < step ? "done" : i === step ? "current" : "todo";
        return (
          <li key={label} className="flex items-center gap-2">
            <button
              type="button"
              disabled={state === "todo"}
              onClick={() => onJump(i)}
              className={`flex items-center gap-2 rounded-full px-3 py-1 ${
                state === "current"
                  ? "bg-arango text-white"
                  : state === "done"
                    ? "bg-arango-50 text-arango"
                    : "text-muted"
              }`}
            >
              <span className="flex size-5 items-center justify-center rounded-full border border-current text-xs">
                {state === "done" ? <Check className="size-3" /> : i + 1}
              </span>
              {label}
            </button>
            {i < STEPS.length - 1 && <span className="h-px w-6 bg-line" aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}

function DropZone({ file, onFile }: { file: File | null; onFile: (f: File) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={`flex flex-col items-center gap-3 rounded-lg border-2 border-dashed px-6 py-10 text-center ${
        over ? "border-arango bg-arango-50" : "border-line bg-panel"
      }`}
    >
      <Upload className="size-8 text-arango" aria-hidden />
      {file ? (
        <p className="text-sm">
          <b>{file.name}</b> · {formatBytes(file.size)}
        </p>
      ) : (
        <p className="text-sm">Drop a JSON, JSONL/NDJSON or CSV file here (gzip OK)</p>
      )}
      <Button variant="secondary" onClick={() => input.current?.click()}>
        <FileUp className="size-4" /> {file ? "Choose another file" : "Choose file"}
      </Button>
      <input
        ref={input}
        type="file"
        className="hidden"
        accept=".json,.jsonl,.ndjson,.csv,.tsv,.gz"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
    </div>
  );
}

function pastedFile(text: string): File {
  const trimmed = text.trim();
  const name = trimmed.startsWith("[") ? "pasted.json" : "pasted.jsonl";
  return new File([trimmed], name, { type: "application/json" });
}

function Nav({
  onBack,
  onNext,
  nextLabel = "Next",
  nextDisabled,
  loading,
  extra,
}: {
  onBack?: () => void;
  onNext?: () => void;
  nextLabel?: ReactNode;
  nextDisabled?: boolean;
  loading?: boolean;
  extra?: ReactNode;
}) {
  return (
    <div className="mt-6 flex items-center gap-2 border-t border-line pt-4">
      {onBack && (
        <Button variant="secondary" onClick={onBack}>
          Back
        </Button>
      )}
      {extra}
      {onNext && (
        <Button className="ml-auto" onClick={onNext} disabled={nextDisabled} loading={loading}>
          {nextLabel}
        </Button>
      )}
    </div>
  );
}

export function LoadWizard() {
  const qc = useQueryClient();
  const collections = useCollections();
  const [step, setStep] = useState(0);

  // Step 1 - source
  const [mode, setMode] = useState<SourceMode>("file");
  const [file, setFile] = useState<File | null>(null);
  const [pasted, setPasted] = useState("");
  const [urls, setUrls] = useState("");
  const [existing, setExisting] = useState("");
  const [preview, setPreview] = useState<PreviewResponse | null>(null);

  // Step 2 - labels
  const [labeled, setLabeled] = useState<LabeledField[]>([]);
  const [useTemplate, setUseTemplate] = useState(false);
  const [template, setTemplate] = useState("");
  const [templateTarget, setTemplateTarget] = useState("embedding");
  const [keyField, setKeyField] = useState("");
  const [onMissing, setOnMissing] = useState<OnMissing>("skip");

  // Step 3 - model
  const [choice, setChoice] = useState<ModelChoice>(DEFAULT_MODEL_CHOICE);

  // Step 4 - target
  const [target, setTarget] = useState("");
  const [createCollection, setCreateCollection] = useState(true);
  const [onDuplicate, setOnDuplicate] = useState<OnDuplicate>("error");
  const [batchSize, setBatchSize] = useState("64");
  const [storeMetadata, setStoreMetadata] = useState(true);
  const [vectorIndex, setVectorIndex] = useState(true);
  const [metric, setMetric] = useState<VectorIndexSpec["metric"]>("cosine");
  const [dryRun, setDryRun] = useState(false);
  const [onlyMissing, setOnlyMissing] = useState(true);
  const [saveProfile, setSaveProfile] = useState(false);
  const [autoWatch, setAutoWatch] = useState(false);

  const [job, setJob] = useState<Job | null>(null);

  const targetCollection = mode === "collection" ? existing : target;
  const targetInfo = collections.data?.find((c) => c.name === targetCollection);

  const previewMutation = useMutation({
    mutationFn: async (): Promise<PreviewResponse> => {
      if (mode === "collection") {
        const res = await unwrap(
          api.GET("/api/v1/collections/{name}/sample", {
            params: { path: { name: existing }, query: { limit: 50 } },
          }),
        );
        return {
          filename: null,
          format: "collection",
          sample: res.sample,
          fields: res.fields.filter(
            (f) => !f.path.endsWith("_embedding") && !f.path.includes("_meta"),
          ),
          estimated_total: res.count,
          size_bytes: null,
        };
      }
      if (mode === "url") {
        const first = urls.split(/\s+/).find(Boolean) ?? "";
        return unwrap(
          api.POST("/api/v1/preview/url", {
            body: { url: first, format: "auto", limit: 50 } as never,
          }),
        );
      }
      const form = new FormData();
      form.append("file", mode === "paste" ? pastedFile(pasted) : file!);
      return postForm<PreviewResponse>("/api/v1/preview?limit=50", form);
    },
    onSuccess: (data) => {
      setPreview(data);
      const fields = data.fields as { path: string; embeddable: boolean }[];
      // Sensible defaults: label the best text field, use `id` as key if present.
      const best = fields.find((f) => f.embeddable);
      setLabeled(best ? [{ path: best.path, target: "embedding" }] : []);
      setKeyField(fields.some((f) => f.path === "id") && mode !== "collection" ? "id" : "");
      if (!target && data.filename) {
        setTarget(
          data.filename
            .replace(/\.(gz)$/i, "")
            .replace(/\.[^.]+$/, "")
            .replace(/[^A-Za-z0-9_-]/g, "_")
            .replace(/^[^A-Za-z]+/, "c_")
            .slice(0, 60),
        );
      }
      setStep(1);
    },
  });

  const specs = useMemo(
    () =>
      buildSpecs(labeled, useTemplate ? template : null, choice, onMissing, templateTarget),
    [labeled, useTemplate, template, choice, onMissing, templateTarget],
  );

  const run = useMutation({
    mutationFn: async (): Promise<Job> => {
      const vi = vectorIndexSpec(vectorIndex, metric);
      const batch = Number(batchSize) || 64;
      if (saveProfile && !dryRun) {
        await unwrap(
          api.PUT("/api/v1/collections/{name}/profile", {
            params: { path: { name: targetCollection } },
            body: {
              embeddings: specs,
              key_field: keyField || null,
              on_duplicate: onDuplicate === "error" ? "update" : onDuplicate,
              batch_size: batch,
              vector_index: vi,
              auto_watch: autoWatch,
              watch_interval_s: 10,
            },
          }),
        );
      }
      if (mode === "collection") {
        return unwrap(
          api.POST("/api/v1/collections/{name}/embed", {
            params: { path: { name: existing } },
            body: {
              embeddings: specs,
              only_missing: onlyMissing,
              batch_size: batch,
              store_metadata: storeMetadata,
              vector_index: vi,
              filter: {},
            },
          }),
        );
      }
      const options = {
        collection: target,
        create_collection: createCollection,
        embeddings: specs,
        key_field: keyField || null,
        on_duplicate: onDuplicate,
        batch_size: batch,
        store_metadata: storeMetadata,
        vector_index: vi,
        dry_run: dryRun,
      };
      if (mode === "url") {
        return unwrap(
          api.POST("/api/v1/load/url", {
            body: { ...options, urls: urls.split(/\s+/).filter(Boolean) } as never,
          }),
        );
      }
      const form = new FormData();
      form.append("file", mode === "paste" ? pastedFile(pasted) : file!);
      form.append("config", JSON.stringify(options));
      return postForm<Job>("/api/v1/load/file", form);
    },
    onSuccess: (data) => {
      setJob(data);
      setStep(4);
      qc.invalidateQueries({ queryKey: ["collections"] });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const sourceReady =
    (mode === "file" && !!file) ||
    (mode === "paste" && pasted.trim().length > 1) ||
    (mode === "url" && urls.trim().length > 0) ||
    (mode === "collection" && !!existing);

  const labelsReady = useTemplate
    ? template.includes("{") && !!templateTarget
    : labeled.length > 0 && labeled.every((l) => l.target.trim());

  const targetReady = /^[A-Za-z][A-Za-z0-9_-]{0,255}$/.test(targetCollection);

  const reset = () => {
    setJob(null);
    setPreview(null);
    setStep(0);
    run.reset();
  };

  const fieldPaths = (preview?.fields as { path: string }[] | undefined)?.map((f) => f.path) ?? [];

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold">Load documents with embeddings</h1>
        <p className="mt-1 text-sm text-muted">
          Label the field(s) to embed, pick any embedding model, and the vectors are stored in the
          same ArangoDB documents.
        </p>
      </div>
      <Stepper step={step} onJump={(s) => s < step && !job && setStep(s)} />

      {step === 0 && (
        <Card title="1 · Choose a source">
          <div className="flex flex-col gap-5">
            <Segmented
              value={mode}
              onChange={(m) => {
                setMode(m);
                setPreview(null);
              }}
              options={[
                { value: "file", label: <><FileUp className="size-4" /> Upload file</> },
                { value: "paste", label: <><ClipboardPaste className="size-4" /> Paste JSON</> },
                { value: "url", label: <><Link2 className="size-4" /> From URL</> },
                { value: "collection", label: <><Database className="size-4" /> Existing collection</> },
              ]}
            />
            {mode === "file" && <DropZone file={file} onFile={setFile} />}
            {mode === "paste" && (
              <Field label="Documents" hint="A JSON array of objects, or one JSON object per line (JSONL).">
                <Textarea
                  rows={10}
                  value={pasted}
                  placeholder={'{"id": 1234, "text": "title + abstract + keywords"}\n{"id": 1235, "text": "..."}'}
                  onChange={(e) => setPasted(e.target.value)}
                />
              </Field>
            )}
            {mode === "url" && (
              <Field
                label="File URLs"
                hint={<>One per line. HTTP(S) or public <Code>gs://bucket/path</Code> URLs; streamed, never staged.</>}
              >
                <Textarea
                  rows={4}
                  value={urls}
                  placeholder="https://storage.googleapis.com/my-bucket/data/part-0001.jsonl"
                  onChange={(e) => setUrls(e.target.value)}
                />
              </Field>
            )}
            {mode === "collection" && (
              <Field label="Collection" hint="Embed documents that are already stored in ArangoDB.">
                <Select value={existing} onChange={(e) => setExisting(e.target.value)}>
                  <option value="">Select a collection…</option>
                  {collections.data?.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({formatNumber(c.count)} docs)
                    </option>
                  ))}
                </Select>
                <ErrorPanel error={collections.error} />
              </Field>
            )}
            <ErrorPanel error={previewMutation.error} />
          </div>
          <Nav
            onNext={() => previewMutation.mutate()}
            nextLabel="Preview fields"
            nextDisabled={!sourceReady}
            loading={previewMutation.isPending}
          />
        </Card>
      )}

      {step === 1 && preview && (
        <Card
          title="2 · Label the field(s) to embed"
          actions={
            <span className="text-sm text-muted">
              {preview.format.toUpperCase()}
              {preview.estimated_total != null && <> · {formatNumber(preview.estimated_total)} documents</>}
              {preview.size_bytes != null && <> · {formatBytes(preview.size_bytes)}</>}
            </span>
          }
        >
          <div className="flex flex-col gap-5">
            <Segmented
              value={useTemplate ? "template" : "fields"}
              onChange={(v) => setUseTemplate(v === "template")}
              options={[
                { value: "fields", label: "Embed field(s)" },
                { value: "template", label: "Combine fields with a template" },
              ]}
            />
            {useTemplate ? (
              <div className="grid gap-4 sm:grid-cols-3">
                <Field
                  className="sm:col-span-2"
                  label="Template"
                  hint={<>Use <Code>{"{field.path}"}</Code> placeholders, e.g. <Code>{"{title}. {abstract}"}</Code></>}
                >
                  <Input
                    className="font-mono"
                    value={template}
                    placeholder={fieldPaths.slice(0, 2).map((p) => `{${p}}`).join("\n")}
                    onChange={(e) => setTemplate(e.target.value)}
                  />
                </Field>
                <Field label="Store vector in">
                  <Input
                    className="font-mono"
                    value={templateTarget}
                    onChange={(e) => setTemplateTarget(e.target.value)}
                  />
                </Field>
                <div className="flex flex-wrap gap-1.5 sm:col-span-3">
                  {fieldPaths.map((p) => (
                    <button
                      key={p}
                      type="button"
                      className="rounded border border-line bg-panel px-2 py-0.5 font-mono text-xs hover:border-arango"
                      onClick={() => setTemplate((t) => `${t}${t ? " " : ""}{${p}}`)}
                    >
                      {`{${p}}`}
                    </button>
                  ))}
                </div>
              </div>
            ) : preview.fields.length ? (
              <FieldLabeler
                fields={preview.fields as never}
                labeled={labeled}
                onChange={setLabeled}
              />
            ) : (
              <EmptyState>No fields detected in the sample.</EmptyState>
            )}

            <div className="grid gap-4 sm:grid-cols-2">
              {mode !== "collection" && (
                <Field label="Document key (_key)" hint="Optional: use this field's value as the ArangoDB _key.">
                  <Select value={keyField} onChange={(e) => setKeyField(e.target.value)}>
                    <option value="">Auto-generate keys</option>
                    {fieldPaths.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </Select>
                </Field>
              )}
              <Field label="When the field is missing or empty">
                <Select value={onMissing} onChange={(e) => setOnMissing(e.target.value as OnMissing)}>
                  <option value="skip">Store the document without a vector</option>
                  <option value="null">Store the document with a null vector</option>
                  <option value="error">Reject the document</option>
                </Select>
              </Field>
            </div>

            <details className="rounded-md border border-line">
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium">
                Sample documents ({preview.sample.length})
              </summary>
              <pre className="max-h-72 overflow-auto border-t border-line bg-panel p-3 font-mono text-xs">
                {JSON.stringify(preview.sample.slice(0, 5), null, 2)}
              </pre>
            </details>
          </div>
          <Nav onBack={() => setStep(0)} onNext={() => setStep(2)} nextDisabled={!labelsReady} />
        </Card>
      )}

      {step === 2 && (
        <Card title="3 · Choose an embedding model">
          <ModelPicker value={choice} onChange={setChoice} />
          <Nav onBack={() => setStep(1)} onNext={() => setStep(3)} />
        </Card>
      )}

      {step === 3 && (
        <Card title="4 · Target collection & options">
          <div className="flex flex-col gap-5">
            <div className="grid gap-4 sm:grid-cols-2">
              {mode === "collection" ? (
                <Field label="Collection">
                  <Input value={existing} disabled />
                </Field>
              ) : (
                <Field
                  label="Target collection"
                  hint={
                    targetInfo ? (
                      <>
                        Existing collection · {formatNumber(targetInfo.count)} documents
                        {targetInfo.has_profile && " · has an auto-embed profile"}
                      </>
                    ) : (
                      "A new collection will be created."
                    )
                  }
                >
                  <Input
                    list="collection-names"
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                    placeholder="pubmed"
                  />
                  <datalist id="collection-names">
                    {collections.data?.map((c) => <option key={c.name} value={c.name} />)}
                  </datalist>
                </Field>
              )}
              {mode !== "collection" && (
                <Field label="If a document key already exists">
                  <Select value={onDuplicate} onChange={(e) => setOnDuplicate(e.target.value as OnDuplicate)}>
                    <option value="error">Report an error for that document</option>
                    <option value="update">Update (merge) the existing document</option>
                    <option value="replace">Replace the existing document</option>
                    <option value="ignore">Keep the existing document</option>
                  </Select>
                </Field>
              )}
              <Field label="Batch size" hint="Documents embedded and written per batch.">
                <Input type="number" min={1} max={10000} value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
              </Field>
              <Field label="Vector index metric">
                <Select
                  value={metric}
                  disabled={!vectorIndex}
                  onChange={(e) => setMetric(e.target.value as VectorIndexSpec["metric"])}
                >
                  <option value="cosine">cosine</option>
                  <option value="l2">l2 (euclidean)</option>
                  <option value="innerProduct">innerProduct</option>
                </Select>
              </Field>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Checkbox
                checked={vectorIndex}
                onChange={setVectorIndex}
                label="Create an ArangoDB vector index after loading"
                hint="Requires ArangoDB 3.12.4+ with --experimental-vector-index."
              />
              <Checkbox
                checked={storeMetadata}
                onChange={setStoreMetadata}
                label="Store embedding metadata"
                hint="Adds <target>_meta with provider, model, dimension and timestamp."
              />
              {mode === "collection" ? (
                <Checkbox
                  checked={onlyMissing}
                  onChange={setOnlyMissing}
                  label="Only documents without a vector"
                />
              ) : (
                <>
                  <Checkbox checked={createCollection} onChange={setCreateCollection} label="Create the collection if missing" />
                  <Checkbox
                    checked={dryRun}
                    onChange={setDryRun}
                    label="Dry run"
                    hint="Embed but don't write; shows sample output."
                  />
                </>
              )}
            </div>
            <div className="rounded-md border border-line bg-panel p-4">
              <Checkbox
                checked={saveProfile}
                onChange={setSaveProfile}
                disabled={dryRun}
                label={<b>Save as this collection's auto-embed profile</b>}
                hint="Future loads into the collection are embedded automatically - no field labels needed."
              />
              {saveProfile && (
                <div className="mt-3 pl-6">
                  <Checkbox
                    checked={autoWatch}
                    onChange={setAutoWatch}
                    label="Also watch the collection"
                    hint="Embeds documents inserted by any client (arangoimport, drivers, web UI)."
                  />
                </div>
              )}
            </div>
            <div>
              <h3 className="mb-2 text-sm font-medium">Request preview</h3>
              <pre className="max-h-56 overflow-auto rounded-md bg-panel p-3 font-mono text-xs">
                {JSON.stringify(
                  { collection: targetCollection, key_field: keyField || undefined, embeddings: specs.map((s) => ({ ...s, provider_options: Object.keys(s.provider_options ?? {}).length ? "…" : undefined })) },
                  null,
                  2,
                )}
              </pre>
            </div>
            <ErrorPanel error={run.error} />
          </div>
          <Nav
            onBack={() => setStep(2)}
            onNext={() => run.mutate()}
            nextLabel={dryRun ? "Run dry run" : mode === "collection" ? "Embed collection" : "Start load"}
            nextDisabled={!targetReady}
            loading={run.isPending}
            extra={
              targetInfo?.has_profile && mode !== "collection" ? (
                <Badge tone="green">Collection has a profile; labels above take precedence</Badge>
              ) : undefined
            }
          />
        </Card>
      )}

      {step === 4 && job && (
        <Card
          title="5 · Run"
          actions={
            <Button variant="secondary" onClick={reset}>
              New load
            </Button>
          }
        >
          <JobProgress initial={job} />
        </Card>
      )}
    </div>
  );
}
