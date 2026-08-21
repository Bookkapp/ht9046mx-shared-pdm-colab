import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  build: {
    sourcemap: false,
    target: 'es2022',
    // ECharts core + line/scatter/heatmap is lazy-loaded only on analytical
    // routes and is ~207 KiB gzip in the production build.
    chunkSizeWarningLimit: 650,
  },
})
