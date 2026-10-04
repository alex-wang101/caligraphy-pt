import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API_URL points the dev proxy at a different API server (default: port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": process.env.API_URL ?? "http://127.0.0.1:8000" },
  },
});
