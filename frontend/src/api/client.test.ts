import { describe, expect, it } from "vitest";

import { apiBaseFor } from "./client";

describe("apiBaseFor", () => {
  it("prefixes calls with the platform mount", () => {
    expect(apiBaseFor("/_service/uds/_db/_system/arango-embedding-loader/")).toBe(
      "/_service/uds/_db/_system/arango-embedding-loader",
    );
    expect(apiBaseFor("/_service/uds/_db/_system/arango-embedding-loader")).toBe(
      "/_service/uds/_db/_system/arango-embedding-loader",
    );
    expect(apiBaseFor("/_service/uds/_db/_system/arango-embedding-loader/index.html")).toBe(
      "/_service/uds/_db/_system/arango-embedding-loader",
    );
  });

  it("is empty when served from the root", () => {
    expect(apiBaseFor("/")).toBe("");
    expect(apiBaseFor("/index.html")).toBe("");
    expect(apiBaseFor("")).toBe("");
  });
});
