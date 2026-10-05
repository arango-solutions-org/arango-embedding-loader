import { Checkbox, Field, Input } from "./ui";

interface PropSchema {
  type?: string;
  format?: string;
  description?: string;
  default?: unknown;
  title?: string;
  anyOf?: PropSchema[];
}

function resolveType(schema: PropSchema): { type: string; secret: boolean } {
  const variants = schema.anyOf ?? [schema];
  const main = variants.find((v) => v.type && v.type !== "null") ?? schema;
  return { type: main.type ?? "string", secret: main.format === "password" };
}

/** Renders a form for a provider's `options_schema` (JSON Schema from pydantic). */
export function ProviderOptionsForm({
  schema,
  value,
  onChange,
}: {
  schema: Record<string, unknown> | undefined;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
}) {
  const properties = (schema?.properties ?? {}) as Record<string, PropSchema>;
  const entries = Object.entries(properties);
  if (!entries.length) return null;
  const set = (key: string, v: unknown) => onChange({ ...value, [key]: v });

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {entries.map(([key, prop]) => {
        const { type, secret } = resolveType(prop);
        const label = prop.title ?? key;
        if (type === "boolean") {
          const current = (value[key] as boolean | undefined) ?? (prop.default as boolean) ?? false;
          return (
            <div key={key} className="flex items-end pb-2">
              <Checkbox
                checked={current}
                onChange={(v) => set(key, v)}
                label={label}
                hint={prop.description}
              />
            </div>
          );
        }
        return (
          <Field key={key} label={label} hint={prop.description}>
            <Input
              type={secret ? "password" : type === "number" || type === "integer" ? "number" : "text"}
              autoComplete="off"
              placeholder={prop.default != null ? String(prop.default) : "server default"}
              value={(value[key] as string | number | undefined) ?? ""}
              onChange={(e) =>
                set(
                  key,
                  (type === "number" || type === "integer") && e.target.value !== ""
                    ? Number(e.target.value)
                    : e.target.value,
                )
              }
            />
          </Field>
        );
      })}
    </div>
  );
}
