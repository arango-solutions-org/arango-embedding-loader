import { useQuery } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import { useDatabase } from "./database";

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
  const database = useDatabase();
  return useQuery({
    queryKey: ["connection", database],
    queryFn: () => unwrap(api.GET("/api/v1/connection")),
    staleTime: 30_000,
  });
}

export function useDatabases() {
  return useQuery({
    queryKey: ["databases"],
    queryFn: () => unwrap(api.GET("/api/v1/databases")),
    staleTime: 60_000,
    retry: false,
  });
}

export function useCollections() {
  const database = useDatabase();
  return useQuery({
    queryKey: ["collections", database],
    queryFn: () => unwrap(api.GET("/api/v1/collections", { params: { query: {} } })),
    retry: false,
  });
}

export function useProfileStatus(collection: string | null) {
  const database = useDatabase();
  return useQuery({
    queryKey: ["profile", database, collection],
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
