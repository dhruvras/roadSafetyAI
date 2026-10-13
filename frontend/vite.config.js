import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the website calls /api and /files on its own address; Vite forwards them to FastAPI,
// so the browser never sees a second origin and no CORS setup is needed.
const api = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": api, "/files": api } },
});
