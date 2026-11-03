import type { EmbeddingFieldSpec, Preset, VectorIndexSpec } from "../api/client";

export type ModelMode = "preset" | "custom";

export interface ModelChoice {
  mode: ModelMode;
  preset: string;
  provider: string;
  model: string;
  dimensions: string;
  options: Record<string, unknown>;
}

export const DEFAULT_MODEL_CHOICE: ModelChoice = {
  mode: "preset",
  preset: "pubmedbert",
  provider: "sentence_transformers",
  model: "",
  dimensions: "",
  options: {},
};

export interface LabeledField {
  path: string;
  target: string;
}

export type OnMissing = "skip" | "null" | "error";

export type LongInputMode = NonNullable<EmbeddingFieldSpec["long_input"]>;

/** Choices for `long_input`, in the order shown in the GUI. */
export const LONG_INPUT_MODES: { value: LongInputMode; label: string }[] = [
  { value: "truncate_end", label: "Keep the first tokens (truncate the end)" },
  { value: "truncate_start", label: "Keep the last tokens (truncate the start)" },
  { value: "truncate_middle", label: "Keep the start and the end (truncate the middle)" },
  { value: "chunk_mean", label: "Embed overlapping chunks, store their mean" },
  { value: "chunk_docs", label: "Store each chunk as its own document" },
  { value: "skip", label: "Don't embed the document" },
  { value: "error", label: "Fail the job" },
];

export interface LongInputSettings {
  mode: LongInputMode;
  overlap: string;
  maxTokens: string;
  tokenChars: string;
  warn: boolean;
}

export const DEFAULT_LONG_INPUT: LongInputSettings = {
  mode: "truncate_end",
  overlap: "25",
  maxTokens: "",
  tokenChars: "",
  warn: true,
};

/** Spec fields for the long-input settings; blank numbers fall back to server defaults. */
export function longInputFields(s: LongInputSettings): Partial<EmbeddingFieldSpec> {
  const num = (v: string) => (v.trim() && Number(v) > 0 ? Number(v) : undefined);
  const overlap = s.overlap.trim() === "" ? undefined : Math.max(0, Number(s.overlap) || 0);
  return {
    long_input: s.mode,
    warn_on_long_input: s.warn,
    ...(overlap !== undefined ? { chunk_overlap_tokens: overlap } : {}),
    ...(num(s.maxTokens) ? { max_tokens: num(s.maxTokens) } : {}),
    ...(num(s.tokenChars) ? { token_chars: num(s.tokenChars) } : {}),
  };
}

export function defaultTarget(path: string, isOnly: boolean): string {
  return isOnly ? "embedding" : `${path.replace(/\./g, "_")}_embedding`;
}

/** Remove empty option values so server-side defaults (env vars) apply. */
export function cleanOptions(options: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(options).filter(([, v]) => v !== "" && v !== undefined && v !== null),
  );
}

export function modelSpecFields(choice: ModelChoice): Partial<EmbeddingFieldSpec> {
  const dims = choice.dimensions.trim() ? Number(choice.dimensions) : undefined;
  const base: Partial<EmbeddingFieldSpec> = {
    provider_options: cleanOptions(choice.options),
    ...(dims ? { dimensions: dims } : {}),
  };
  if (choice.mode === "preset") {
    return { ...base, preset: choice.preset, ...(choice.model ? { model: choice.model } : {}) };
  }
  return { ...base, provider: choice.provider, ...(choice.model ? { model: choice.model } : {}) };
}

export function buildSpecs(
  fields: LabeledField[],
  template: string | null,
  choice: ModelChoice,
  onMissing: OnMissing,
  templateTarget = "embedding",
  longInput: LongInputSettings = DEFAULT_LONG_INPUT,
): EmbeddingFieldSpec[] {
  const common = {
    ...modelSpecFields(choice),
    on_missing: onMissing,
    ...longInputFields(longInput),
  };
  if (template) {
    return [{ template, target_field: templateTarget, ...common } as EmbeddingFieldSpec];
  }
  return fields.map(
    (f) => ({ source_field: f.path, target_field: f.target, ...common }) as EmbeddingFieldSpec,
  );
}

export function resolvedProvider(choice: ModelChoice, presets: Preset[] | undefined): string {
  if (choice.mode === "custom") return choice.provider;
  return presets?.find((p) => p.name === choice.preset)?.provider ?? "";
}

export function resolvedModel(choice: ModelChoice, presets: Preset[] | undefined): string {
  if (choice.model) return choice.model;
  if (choice.mode === "preset") return presets?.find((p) => p.name === choice.preset)?.model ?? "";
  return "";
}

export function vectorIndexSpec(
  enabled: boolean,
  metric: VectorIndexSpec["metric"],
): VectorIndexSpec | null {
  return enabled ? ({ create: true, metric } as VectorIndexSpec) : null;
}

export function formatBytes(n: number | null | undefined): string {
  if (n == null) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

export function formatNumber(n: number | null | undefined): string {
  return n == null ? "–" : n.toLocaleString();
}

export function formatDuration(start?: string | null, end?: string | null): string {
  if (!start) return "–";
  const ms = (end ? Date.parse(end) : Date.now()) - Date.parse(start);
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}
