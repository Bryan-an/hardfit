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
    // Scope to source so Vitest's default glob doesn't pick up the Playwright
    // e2e specs under e2e/ (they import @playwright/test and run separately).
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    coverage: { provider: 'v8', reporter: ['text', 'lcov'] },
  },
})
