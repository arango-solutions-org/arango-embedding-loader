import { useQuery } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import { connectionKey, useConnection } from "./connection";

export function useProviders() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: () => unwrap(api.GET("/api/v1/providers")),
    staleTime: 60_000,
  });
}

export function usePresets() {
  return useQuery({
    queryKey: ["presets"],
    queryFn: () => unwrap(api.GET("/api/v1/presets")),
    staleTime: 300_000,
  });
}

export function useModels(provider: string) {
  return useQuery({
    queryKey: ["models", provider],
    queryFn: () =>
      unwrap(api.GET("/api/v1/providers/{name}/models", { params: { path: { name: provider } } })),
    enabled: !!provider,
    staleTime: 60_000,
  });
}

export function useConnectionInfo() {
  const conn = useConnection();
  return useQuery({
    queryKey: ["connection", connectionKey(conn)],
    queryFn: () => unwrap(api.GET("/api/v1/connection")),
    staleTime: 30_000,
  });
}

export function useCollections() {
  const conn = useConnection();
  return useQuery({
    queryKey: ["collections", connectionKey(conn)],
    queryFn: () => unwrap(api.GET("/api/v1/collections", { params: { query: {} } })),
    retry: false,
  });
}

export function useProfileStatus(collection: string | null) {
  const conn = useConnection();
  return useQuery({
    queryKey: ["profile", connectionKey(conn), collection],
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/collections/{name}/profile", {
          params: { path: { name: collection! } },
        }),
      ),
    enabled: !!collection,
    retry: false,
    refetchInterval: 5000,
  });
}
