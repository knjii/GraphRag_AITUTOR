import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В режиме api (npm run dev:api) фронт ходит в демо-сервис web/server/demo_api.py
// через прокси. Адрес меняется переменной RAG_API_URL, например на туннель к серверу.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.RAG_API_URL ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
