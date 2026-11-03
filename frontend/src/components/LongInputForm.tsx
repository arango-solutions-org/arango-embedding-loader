import { LONG_INPUT_MODES, type LongInputMode, type LongInputSettings } from "../lib/spec";
import { Checkbox, Field, Input, Select } from "./ui";

/** `long_input` and its options: what to do with texts longer than the model's token limit. */
export function LongInputForm({
  value,
  onChange,
}: {
  value: LongInputSettings;
  onChange: (v: LongInputSettings) => void;
}) {
  const set = <K extends keyof LongInputSettings>(key: K, v: LongInputSettings[K]) =>
    onChange({ ...value, [key]: v });
  const chunked = value.mode === "chunk_mean" || value.mode === "chunk_docs";

  return (
    <div className="mt-5 flex flex-col gap-4 rounded-md border border-line p-4">
      <h3 className="text-sm font-medium">Texts longer than the model's limit</h3>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field
          label="Long input"
          hint="Tokens are counted with the model's tokenizer when available, otherwise estimated."
        >
          <Select
            value={value.mode}
            onChange={(e) => set("mode", e.target.value as LongInputMode)}
          >
            {LONG_INPUT_MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.value}: {m.label}
                {m.value === "truncate_end" ? " (default)" : ""}
              </option>
            ))}
          </Select>
        </Field>
        {chunked && (
          <Field
            label="Chunk overlap (tokens)"
            hint="Tokens shared by consecutive chunks, rounded to the nearest word."
          >
            <Input
              type="number"
              min={0}
              placeholder="25"
              value={value.overlap}
              onChange={(e) => set("overlap", e.target.value)}
            />
          </Field>
        )}
      </div>
      <Checkbox
        checked={value.warn}
        onChange={(v) => set("warn", v)}
        label="Log each long text as a warning"
        hint="When off, they are logged at info level. With 'error' they are always logged as errors."
      />
      <details>
        <summary className="cursor-pointer text-sm text-muted">Advanced</summary>
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          <Field
            label="Max tokens"
            hint="Token limit to apply. Defaults to the model's limit; needed when it is unknown (e.g. Ollama)."
          >
            <Input
              type="number"
              min={1}
              placeholder="model's limit"
              value={value.maxTokens}
              onChange={(e) => set("maxTokens", e.target.value)}
            />
          </Field>
          <Field
            label="Characters per token"
            hint="Used to estimate token counts for models without a local tokenizer."
          >
            <Input
              type="number"
              min={0.1}
              step={0.1}
              placeholder="4"
              value={value.tokenChars}
              onChange={(e) => set("tokenChars", e.target.value)}
            />
          </Field>
        </div>
      </details>
    </div>
  );
}
