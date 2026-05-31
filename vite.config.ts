/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Evergreen-only target → drop Vite's inline module-preload polyfill <script>,
  // which a strict `script-src 'self'` CSP (a later unit) would block (→ blank page).
  build: { modulePreload: { polyfill: false } },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    coverage: { provider: 'v8', reporter: ['text', 'lcov'] },
  },
})
