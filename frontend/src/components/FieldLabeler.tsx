import { Sparkles } from "lucide-react";

import type { FieldPathInfo } from "../api/client";
import { defaultTarget, type LabeledField } from "../lib/spec";
import { Badge, Input } from "./ui";

function preview(value: unknown): string {
  if (value == null) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 90 ? `${text.slice(0, 87)}…` : text;
}

/** Table of detected fields; the user ticks the field(s) to embed and names the target. */
export function FieldLabeler({
  fields,
  labeled,
  onChange,
}: {
  fields: FieldPathInfo[];
  labeled: LabeledField[];
  onChange: (v: LabeledField[]) => void;
}) {
  const selected = new Map(labeled.map((l) => [l.path, l]));

  const toggle = (path: string, on: boolean) => {
    let next = on
      ? [...labeled, { path, target: "" }]
      : labeled.filter((l) => l.path !== path);
    // Keep default target names in sync with how many fields are labeled.
    next = next.map((l) => {
      const wasDefault =
        !l.target ||
        l.target === defaultTarget(l.path, true) ||
        l.target === defaultTarget(l.path, false);
      return wasDefault ? { ...l, target: defaultTarget(l.path, next.length === 1) } : l;
    });
    onChange(next);
  };

  const rename = (path: string, target: string) =>
    onChange(labeled.map((l) => (l.path === path ? { ...l, target } : l)));

  return (
    <div className="overflow-x-auto rounded-md border border-line">
      <table className="w-full text-sm">
        <thead className="bg-panel text-left">
          <tr>
            <th className="w-16 px-3 py-2 font-medium">Embed</th>
            <th className="px-3 py-2 font-medium">Field</th>
            <th className="px-3 py-2 font-medium">Type</th>
            <th className="px-3 py-2 font-medium">Coverage</th>
            <th className="px-3 py-2 font-medium">Example</th>
            <th className="w-40 px-3 py-2 font-medium whitespace-nowrap">Store vector in</th>
          </tr>
        </thead>
        <tbody>
          {fields.map((f) => {
            const sel = selected.get(f.path);
            return (
              <tr key={f.path} className={sel ? "bg-arango-50" : "border-t border-line"}>
                <td className="px-3 py-2">
                  <input
                    type="checkbox"
                    aria-label={`Embed ${f.path}`}
                    className="size-4 accent-arango"
                    checked={!!sel}
                    onChange={(e) => toggle(f.path, e.target.checked)}
                  />
                </td>
                <td className="px-3 py-2 font-mono">
                  <span className="flex items-center gap-2">
                    {f.path}
                    {f.embeddable && (
                      <Badge tone="green">
                        <Sparkles className="size-3" /> text
                      </Badge>
                    )}
                  </span>
                </td>
                <td className="px-3 py-2 text-muted">{f.types.join(", ")}</td>
                <td className="px-3 py-2">{Math.round(f.coverage * 100)}%</td>
                <td className="px-3 py-2 text-muted" title={preview(f.example)}>
                  <div className="max-w-[12rem] truncate lg:max-w-[22rem]">{preview(f.example)}</div>
                </td>
                <td className="px-3 py-2">
                  {sel && (
                    <Input
                      aria-label={`Target field for ${f.path}`}
                      className="font-mono"
                      value={sel.target}
                      onChange={(e) => rename(f.path, e.target.value)}
                    />
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
