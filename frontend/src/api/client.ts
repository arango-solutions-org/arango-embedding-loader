import createClient, { type Middleware } from "openapi-fetch";

import { connectionHeaders } from "../lib/connection";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];
export type Job = Schemas["Job"];
export type ProviderInfo = Schemas["ProviderInfo"];
export type Preset = Schemas["Preset"];
export type FieldPathInfo = Schemas["FieldPathInfo"];
export type PreviewResponse = Schemas["PreviewResponse"];
export type EmbeddingFieldSpec = Schemas["EmbeddingFieldSpec"];
export type ProfileStatus = Schemas["ProfileStatus"];
export type ProfileConfig = Schemas["ProfileConfig"];
export type CollectionInfo = Schemas["CollectionInfo"];
export type ModelInfo = Schemas["ModelInfo"];
export type VectorIndexSpec = Schemas["VectorIndexSpec"];

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    public detail: unknown,
  ) {
    super(formatDetail(detail) || code);
  }
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const item = d as { loc?: unknown[]; msg?: string };
        return item.loc ? `${item.loc.slice(1).join(".")}: ${item.msg}` : JSON.stringify(d);
      })
      .join("; ");
  }
  return detail ? JSON.stringify(detail) : "";
}

const withConnection: Middleware = {
  onRequest({ request }) {
    for (const [k, v] of Object.entries(connectionHeaders())) request.headers.set(k, v);
    return request;
  },
};

export const api = createClient<paths>({ baseUrl: "" });
api.use(withConnection);

/** Unwrap an openapi-fetch result, throwing ApiError on failure. */
export async function unwrap<T>(
  promise: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await promise;
  if (error !== undefined || !response.ok) {
    const body = (error ?? {}) as { error?: string; detail?: unknown };
    throw new ApiError(response.status, body.error ?? `HTTP ${response.status}`, body.detail);
  }
  return data as T;
}

/** Multipart requests (file uploads) go through fetch directly. */
export async function postForm<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(path, { method: "POST", body: form, headers: connectionHeaders() });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(response.status, body.error ?? `HTTP ${response.status}`, body.detail);
  }
  return body as T;
}
