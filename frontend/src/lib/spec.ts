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
): EmbeddingFieldSpec[] {
  const common = { ...modelSpecFields(choice), on_missing: onMissing };
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
