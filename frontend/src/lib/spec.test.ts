import { describe, expect, it } from "vitest";

import {
  buildSpecs,
  cleanOptions,
  DEFAULT_LONG_INPUT,
  DEFAULT_MODEL_CHOICE,
  defaultTarget,
  formatBytes,
  longInputFields,
} from "./spec";

describe("spec helpers", () => {
  it("builds one spec per labeled field with the preset", () => {
    const specs = buildSpecs(
      [{ path: "text", target: "embedding" }],
      null,
      DEFAULT_MODEL_CHOICE,
      "skip",
    );
    expect(specs).toEqual([
      {
        source_field: "text",
        target_field: "embedding",
        preset: "pubmedbert",
        provider_options: {},
        on_missing: "skip",
        long_input: "truncate_end",
        warn_on_long_input: true,
        chunk_overlap_tokens: 25,
      },
    ]);
  });

  it("builds a template spec for custom providers", () => {
    const specs = buildSpecs([], "{title} {abstract}", {
      ...DEFAULT_MODEL_CHOICE,
      mode: "custom",
      provider: "openai",
      model: "text-embedding-3-small",
      dimensions: "512",
      options: { api_key: "", base_url: "http://x" },
    }, "null");
    expect(specs[0]).toMatchObject({
      template: "{title} {abstract}",
      provider: "openai",
      model: "text-embedding-3-small",
      dimensions: 512,
      provider_options: { base_url: "http://x" },
    });
  });

  it("derives default target names", () => {
    expect(defaultTarget("text", true)).toBe("embedding");
    expect(defaultTarget("content.body", false)).toBe("content_body_embedding");
  });

  it("cleans empty options and formats bytes", () => {
    expect(cleanOptions({ a: "", b: 0, c: false, d: null })).toEqual({ b: 0, c: false });
    expect(formatBytes(1536)).toBe("1.5 KB");
  });
});

describe("long input", () => {
  it("adds the long-input settings to every spec", () => {
    const specs = buildSpecs(
      [{ path: "text", target: "embedding" }],
      null,
      DEFAULT_MODEL_CHOICE,
      "skip",
      "embedding",
      { mode: "chunk_docs", overlap: "10", maxTokens: "", tokenChars: "3.5", warn: false },
    );
    expect(specs[0]).toMatchObject({
      long_input: "chunk_docs",
      chunk_overlap_tokens: 10,
      token_chars: 3.5,
      warn_on_long_input: false,
    });
    expect(specs[0]).not.toHaveProperty("max_tokens");
  });

  it("defaults to truncate_end with 25 overlap tokens", () => {
    expect(longInputFields(DEFAULT_LONG_INPUT)).toEqual({
      long_input: "truncate_end",
      warn_on_long_input: true,
      chunk_overlap_tokens: 25,
    });
  });
});
