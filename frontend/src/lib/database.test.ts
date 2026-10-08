import { describe, expect, it } from "vitest";

import { databaseHeaders, mountDatabase } from "./database";

describe("mountDatabase", () => {
  it("reads the database of a database-scoped platform mount", () => {
    expect(mountDatabase("/_service/uds/_db/ncbi_embeddings/embedder/")).toBe("ncbi_embeddings");
    expect(mountDatabase("/_service/uds/_db/my%20db/embedder")).toBe("my db");
  });

  it("is empty for global mounts and local development", () => {
    expect(mountDatabase("/_service/uds/_global/embedder/")).toBe("");
    expect(mountDatabase("/")).toBe("");
  });
});

describe("databaseHeaders", () => {
  it("sends only the database, never credentials", () => {
    expect(databaseHeaders("ncbi_graph")).toEqual({ "X-Arango-Database": "ncbi_graph" });
    expect(databaseHeaders("")).toEqual({});
  });
});
