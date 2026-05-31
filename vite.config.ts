import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Evergreen-only target → drop Vite's inline module-preload polyfill <script>,
  // which a strict `script-src 'self'` CSP (a later unit) would block (→ blank page).
  build: { modulePreload: { polyfill: false } },
})
