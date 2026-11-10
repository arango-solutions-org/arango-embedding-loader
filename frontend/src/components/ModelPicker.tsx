import { useMutation } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, FlaskConical } from "lucide-react";
import { useState } from "react";

import { api, unwrap } from "../api/client";
import { useModels, usePresets, useProviders } from "../lib/queries";
import {
  cleanOptions,
  DEFAULT_LONG_INPUT,
  longInputRequest,
  resolvedModel,
  resolvedProvider,
  type LongInputSettings,
  type ModelChoice,
} from "../lib/spec";
import { ProviderOptionsForm } from "./ProviderOptionsForm";
import { Badge, Button, ErrorPanel, Field, Input, Segmented, Select } from "./ui";

export function ModelPicker({
  value,
  onChange,
  longInput = DEFAULT_LONG_INPUT,
}: {
  value: ModelChoice;
  onChange: (v: ModelChoice) => void;
  /** Applied to the test embedding exactly as a load would apply it. */
  longInput?: LongInputSettings;
}) {
  const providers = useProviders();
  const presets = usePresets();
  const providerName = resolvedProvider(value, presets.data);
  const provider = providers.data?.find((p) => p.name === providerName);
  const models = useModels(providerName);
  const [showOptions, setShowOptions] = useState(false);
  const set = (patch: Partial<ModelChoice>) => onChange({ ...value, ...patch });

  const test = useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/api/v1/providers/{name}/embed", {
          params: { path: { name: providerName } },
          body: {
            texts: ["The quick brown fox jumps over the lazy dog."],
            model: resolvedModel(value, presets.data) || null,
            dimensions: value.dimensions ? Number(value.dimensions) : null,
            provider_options: cleanOptions(value.options),
            input_type: "document",
            include_vectors: false,
            ...longInputRequest(longInput),
          },
        }),
      ),
  });

  const presetProviderAvailable = (name: string) =>
    providers.data?.find((p) => p.name === name)?.available ?? true;

  return (
    <div className="flex flex-col gap-4">
      <Segmented
        value={value.mode}
        onChange={(mode) => set({ mode, model: "", options: {} })}
        options={[
          { value: "preset", label: "Preset" },
          { value: "custom", label: "Choose provider & model" },
        ]}
      />
      {value.mode === "preset" ? (
        <Field label="Preset" hint={presets.data?.find((p) => p.name === value.preset)?.description}>
          <Select value={value.preset} onChange={(e) => set({ preset: e.target.value, options: {} })}>
            {presets.data?.map((p) => (
              <option key={p.name} value={p.name} disabled={!presetProviderAvailable(p.provider)}>
                {p.label ?? p.name} — {p.model}
                {p.dimensions ? ` (${p.dimensions}d)` : ""}
                {presetProviderAvailable(p.provider) ? "" : " · unavailable"}
              </option>
            ))}
          </Select>
        </Field>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Provider" hint={provider?.description}>
            <Select
              value={value.provider}
              onChange={(e) => set({ provider: e.target.value, model: "", options: {} })}
            >
              {providers.data?.map((p) => (
                <option key={p.name} value={p.name} disabled={!p.available}>
                  {p.label}
                  {p.available ? "" : " · unavailable"}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Model" hint="Pick a known model or type any model id.">
            <Input
              list="model-options"
              placeholder={provider?.default_model ?? "model id"}
              value={value.model}
              onChange={(e) => set({ model: e.target.value })}
            />
            <datalist id="model-options">
              {models.data?.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.dimension ? `${m.dimension} dims` : ""}
                </option>
              ))}
            </datalist>
          </Field>
          {provider?.supports_dimensions && (
            <Field label="Output dimensions" hint="Optional; only for models that support it.">
              <Input
                type="number"
                min={1}
                placeholder="model default"
                value={value.dimensions}
                onChange={(e) => set({ dimensions: e.target.value })}
              />
            </Field>
          )}
        </div>
      )}

      {provider && !provider.available && (
        <ErrorPanel error={`${provider.label} is unavailable: ${provider.reason ?? ""}`} />
      )}

      {provider && (
        <div className="rounded-md border border-line">
          <button
            type="button"
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium hover:bg-panel"
            onClick={() => setShowOptions((s) => !s)}
            aria-expanded={showOptions}
          >
            {showOptions ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
            {provider.label} options
            <span className="font-normal text-muted">(API keys, endpoints, device…)</span>
          </button>
          {showOptions && (
            <div className="border-t border-line p-4">
              <ProviderOptionsForm
                schema={provider.options_schema}
                value={value.options}
                onChange={(options) => set({ options })}
              />
            </div>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="secondary"
          onClick={() => test.mutate()}
          loading={test.isPending}
          disabled={!provider?.available}
        >
          <FlaskConical className="size-4" /> Test embedding
        </Button>
        {test.isPending && (
          <span className="text-xs text-muted">Local models download on first use; this can take a minute.</span>
        )}
        {test.data && (
          <span className="flex items-center gap-2 text-sm">
            <Badge tone="green">{test.data.dimension} dimensions</Badge>
            <span className="text-muted">
              {test.data.model} · {Math.round(test.data.elapsed_ms)} ms
            </span>
          </span>
        )}
      </div>
      <ErrorPanel error={test.error} />
    </div>
  );
}
