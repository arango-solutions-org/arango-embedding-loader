import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const target = process.env.VITE_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: Number(process.env.VITE_PORT ?? 5173),
    proxy: {
      "/api": { target, changeOrigin: true },
      "/docs": { target, changeOrigin: true },
      "/redoc": { target, changeOrigin: true },
      "/openapi.json": { target, changeOrigin: true },
    },
  },
  test: { environment: "node" },
});
