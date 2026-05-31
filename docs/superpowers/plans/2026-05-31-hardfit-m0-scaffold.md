# HardFit M0 — Scaffold + Toolchain + Supply-Chain Hardening + CI + Empty Deploy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a deployable, CI-gated, supply-chain-hardened Vite + React 19 + TypeScript skeleton for HardFit that builds, lints, type-checks, unit-tests, e2e-smoke-tests, and auto-deploys to Cloudflare Pages — the foundation every later milestone builds on.

**Architecture:** A single static SPA package at the repo root. pnpm (pinned via Corepack) is the only package manager; dependency install scripts are blocked by default and new releases are subject to a cooldown. Quality gates (typecheck → audit → lint → unit → license → build → e2e) run in one GitHub Actions workflow; Cloudflare Pages auto-deploys `main` via git integration. No app logic yet beyond a placeholder page — M1 adds the first real feature slice.

**Tech Stack:** Node ≥22.12, pnpm 10 (Corepack), Vite 8, React 19, TypeScript 6 (strictest), Tailwind CSS v4, Biome 2, Vitest 4 + React Testing Library, Playwright + @axe-core/playwright, license-checker, GitHub Actions, Cloudflare Pages.

> **Conventions for the executor**
> - The repo root is `/Users/bryan-andagoya/Development/personal/hard-fit-project/hard-fit`. Run every command from there.
> - Versions in code blocks are current as of 2026-05-31. Where a command pins "@latest", let it resolve; where a literal version appears, treat it as a floor and re-pin via the lockfile.
> - This is a fresh repo: there is no existing code, only `docs/`. Preserve `docs/` at every step.
> - Use **opus** for any subagent dispatched to execute these tasks (project preference).

---

## File Structure

Files created in M0 and their single responsibility:

- `.gitignore` — ignore `node_modules`, `dist`, Playwright artifacts, local env. *(from Vite scaffold, extended)*
- `.nvmrc` — pin Node major version for contributors + CI.
- `package.json` — deps, scripts, `packageManager` (Corepack pin), `engines`.
- `pnpm-workspace.yaml` — **pnpm supply-chain settings** (`minimumReleaseAge`, `strictDepBuilds`, `onlyBuiltDependencies`).
- `.npmrc` — `save-exact=true` (no caret ranges).
- `pnpm-lock.yaml` — committed lockfile with integrity hashes.
- `tsconfig.json` / `tsconfig.node.json` — TypeScript strictest config.
- `vite.config.ts` — Vite plugins (React, Tailwind) + Vitest `test` config.
- `vitest.setup.ts` — RTL/jest-dom setup for component tests.
- `biome.json` — lint + format config + scripts.
- `playwright.config.ts` — e2e config with a `webServer` that serves the built app.
- `src/main.tsx`, `src/App.tsx`, `src/index.css` — placeholder app shell (Tailwind wired).
- `src/lib/version.ts` + `src/lib/version.test.ts` — a tiny pure module proving the Vitest harness works (replaced by real engine code in M1).
- `e2e/smoke.spec.ts` — Playwright smoke + axe check that the page renders and is accessible.
- `public/_headers` — Cloudflare CSP + security headers.
- `public/_redirects` — explicit SPA fallback.
- `.github/workflows/ci.yml` — quality gate on push/PR.
- `.github/dependabot.yml` — npm + GitHub Actions update PRs.
- `README.md` — how to run/build/test/deploy.
- `CLAUDE.md` — project memory for future Claude Code sessions.

---

## Task 1: Initialize git and pin the Node/pnpm toolchain

**Files:**
- Create: `.nvmrc`
- Create/Modify: `package.json` (created in Task 2; `packageManager` + `engines` added here conceptually, applied in Task 2 — this task only pins Node + enables Corepack)

- [ ] **Step 1: Initialize the git repository**

Run:
```bash
git init
git add docs && git commit -m "chore: initial commit (strategy docs)"
```
Expected: a repo on branch `main` (or `master`) with one commit containing `docs/`.

- [ ] **Step 2: Pin the Node version**

Create `.nvmrc`:
```
22
```

- [ ] **Step 3: Enable Corepack and verify pnpm is available**

Run:
```bash
corepack enable
node --version
```
Expected: Node prints `v22.x` (≥ 22.12). If Node is below 22.12, install/switch to Node 22 first (`nvm install 22 && nvm use 22`), because Vite 8 requires it.

- [ ] **Step 4: Commit**

```bash
git add .nvmrc
git commit -m "chore: pin Node 22 and enable Corepack"
```

---

## Task 2: Scaffold the Vite + React + TypeScript app (preserving docs/)

**Files:**
- Create: `package.json`, `index.html`, `vite.config.ts`, `tsconfig.json`, `tsconfig.node.json`, `src/main.tsx`, `src/App.tsx`, `.gitignore`, etc. *(from create-vite)*

- [ ] **Step 1: Scaffold into a temp dir, then move into the repo root**

`create-vite` refuses to write into a non-empty directory, so scaffold into a temp folder and copy the result in (this preserves `docs/`):
```bash
pnpm create vite@latest scaffold-tmp --template react-ts
rsync -a scaffold-tmp/ ./
rm -rf scaffold-tmp
```
Expected: `package.json`, `index.html`, `src/`, `vite.config.ts`, `tsconfig*.json`, `.gitignore` now exist at the repo root; `docs/` is untouched.

- [ ] **Step 2: Pin pnpm via Corepack (writes the `packageManager` field)**

Run:
```bash
corepack use pnpm@latest
```
Expected: `package.json` now contains a `"packageManager": "pnpm@10.x.x+sha512..."` line, and pnpm is activated for this project.

- [ ] **Step 3: Add an `engines` floor to `package.json`**

Add this top-level key to `package.json` (next to `"packageManager"`):
```json
"engines": {
  "node": ">=22.12.0"
}
```

- [ ] **Step 4: Install dependencies**

Run:
```bash
pnpm install
```
Expected: installs successfully and creates `pnpm-lock.yaml`. (You may see a notice that build scripts were ignored — that's expected and is hardened explicitly in Task 3.)

- [ ] **Step 5: Verify the dev server boots, then build**

Run:
```bash
pnpm build
```
Expected: `vite build` completes and emits `dist/`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold Vite + React + TypeScript app"
```

---

## Task 3: Harden the dependency supply chain (pnpm)

**Files:**
- Create: `pnpm-workspace.yaml`
- Create: `.npmrc`
- Modify: `pnpm-lock.yaml` (regenerated)

- [ ] **Step 1: Create `pnpm-workspace.yaml` with supply-chain settings**

```yaml
# pnpm reads project settings from this file in pnpm 10.
# No `packages:` key is needed for a single-package repo.

# Block installing any version published less than 1 day (1440 min) ago,
# so a freshly-compromised release can't be pulled before detection.
minimumReleaseAge: 1440
minimumReleaseAgeExclude: []

# Fail the install if any dependency wants to run a lifecycle (install) script
# that is NOT explicitly allowlisted below. Forces an explicit, reviewed decision.
strictDepBuilds: true

# The ONLY dependencies permitted to run build/install scripts.
# Start empty; Step 3 fills this in after reviewing what the install reports.
onlyBuiltDependencies: []
```

- [ ] **Step 2: Create `.npmrc` to pin exact versions**

```
save-exact=true
```

- [ ] **Step 3: Reinstall, observe blocked build scripts, and allowlist only the legitimate ones**

Run:
```bash
rm -rf node_modules
pnpm install
```
Expected: with `strictDepBuilds: true`, the install **fails** and lists packages whose build scripts were blocked (typically `esbuild`, and — once Tailwind is added in Task 5 — `@tailwindcss/oxide`). **Review each listed package** (these are native-binary builders, legitimate for this stack), then add the reviewed ones to `onlyBuiltDependencies` in `pnpm-workspace.yaml`. After scaffold (before Tailwind/Biome), expect at least:
```yaml
onlyBuiltDependencies:
  - esbuild
```
Re-run `pnpm install` until it succeeds. **Re-run this review step whenever a new dependency triggers a blocked-script failure in later tasks** — you will add `@tailwindcss/oxide` in Task 5 and `workerd` (for `wrangler`) in Task 8. The final reviewed allowlist will be roughly: `esbuild`, `@tailwindcss/oxide`, `workerd`.

- [ ] **Step 4: Verify the hardening is actually active (behaviorally, not via `pnpm config`)**

Do **not** use `pnpm config get minimumReleaseAge` to verify — that reads npmrc-style config, but the setting lives in `pnpm-workspace.yaml`, so it reports `undefined` even when the cooldown is active (a false negative). Verify the two controls by behavior instead:
1. **Script-block proof:** the failed install in Step 3 (before you allowlisted `esbuild`) *is* the proof that `strictDepBuilds` blocks un-allowlisted lifecycle scripts. To re-confirm at any time, temporarily remove an entry from `onlyBuiltDependencies` and run `pnpm install` — it must fail listing that package.
2. **pnpm reads the file:** confirm you're on a version that supports these keys:
```bash
pnpm --version
```
Expected: `10.16.0` or higher (required for `minimumReleaseAge`). If pnpm parsed `pnpm-workspace.yaml` without complaint and installs succeed, the settings are in effect; pnpm will silently refuse to resolve any dependency version published within the last 24h.

- [ ] **Step 5: Commit**

```bash
git add pnpm-workspace.yaml .npmrc pnpm-lock.yaml
git commit -m "chore(security): harden pnpm supply chain (cooldown + script allowlist + exact pins)"
```

---

## Task 4: Enforce TypeScript strictest settings

**Files:**
- Modify: `tsconfig.json`

- [ ] **Step 1: Add the strictest compiler options**

Replace the `compilerOptions` in `tsconfig.json` (the app config; keep `tsconfig.node.json` as scaffolded) so it includes these keys (merge with the scaffolded ones — do not drop `jsx`, `lib`, `moduleResolution`, etc.):
```jsonc
{
  "compilerOptions": {
    // ...keep scaffolded keys (target, lib, jsx: "react-jsx", module, moduleResolution: "bundler", etc.)...
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noImplicitOverride": true,
    "noFallthroughCasesInSwitch": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    "skipLibCheck": true
  }
}
```

- [ ] **Step 2: Add a typecheck script to `package.json`**

Add to `"scripts"`:
```json
"typecheck": "tsc --noEmit"
```

- [ ] **Step 3: Run the typecheck**

Run:
```bash
pnpm typecheck
```
Expected: PASS (no errors). If the scaffolded `src/` has unused-variable errors under the stricter flags, fix them minimally (remove unused imports/vars).

- [ ] **Step 4: Commit**

```bash
git add tsconfig.json package.json src
git commit -m "chore(ts): enable strictest TypeScript config"
```

---

## Task 5: Wire up Tailwind CSS v4

**Files:**
- Modify: `package.json` (dep), `vite.config.ts`, `src/index.css`, `pnpm-workspace.yaml` (allowlist), `pnpm-lock.yaml`

- [ ] **Step 1: Install Tailwind v4 and its Vite plugin**

Run:
```bash
pnpm add tailwindcss @tailwindcss/vite
```
Expected: if the install fails on a blocked build script for `@tailwindcss/oxide`, review it (it's Tailwind v4's native engine — legitimate), add it to `onlyBuiltDependencies` in `pnpm-workspace.yaml`:
```yaml
onlyBuiltDependencies:
  - esbuild
  - "@tailwindcss/oxide"
```
then re-run `pnpm add tailwindcss @tailwindcss/vite`.

- [ ] **Step 2: Add the Tailwind plugin to `vite.config.ts`**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Evergreen-only target → drop Vite's inline module-preload polyfill <script>,
  // which a strict `script-src 'self'` CSP (Task 10) would block (→ blank page).
  build: { modulePreload: { polyfill: false } },
})
```

- [ ] **Step 3: Import Tailwind in the main stylesheet**

Replace the contents of `src/index.css` with:
```css
@import "tailwindcss";
```

- [ ] **Step 4: Use a Tailwind class in `src/App.tsx` to prove it works**

Replace `src/App.tsx` with a minimal placeholder shell:
```tsx
export default function App() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-2 bg-slate-50 text-slate-900">
      <h1 className="text-3xl font-bold">HardFit</h1>
      <p className="text-slate-600">Distribution fitting, in your browser.</p>
    </main>
  )
}
```
Ensure `src/main.tsx` imports `./index.css` (the scaffold already does).

- [ ] **Step 5: Verify the build still works**

Run:
```bash
pnpm build
```
Expected: PASS; generated CSS includes Tailwind utilities.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(ui): wire up Tailwind CSS v4 + placeholder app shell"
```

---

## Task 6: Add Biome for linting and formatting

**Files:**
- Create: `biome.json`
- Modify: `package.json` (dep + scripts), `pnpm-workspace.yaml` (if a script is blocked)

- [ ] **Step 1: Install Biome**

Run:
```bash
pnpm add -D -E @biomejs/biome
```
(If a build-script block fires for `@biomejs/biome`, review and allowlist it as in Task 3.)

- [ ] **Step 2: Create `biome.json`**

```json
{
  "$schema": "https://biomejs.dev/schemas/2.0.0/schema.json",
  "vcs": { "enabled": true, "clientKind": "git", "useIgnoreFile": true },
  "files": { "ignoreUnknown": true },
  "formatter": { "enabled": true, "indentStyle": "space", "indentWidth": 2, "lineWidth": 100 },
  "linter": {
    "enabled": true,
    "rules": { "recommended": true, "a11y": { "recommended": true } }
  },
  "javascript": { "formatter": { "quoteStyle": "single", "semicolons": "asNeeded" } }
}
```

- [ ] **Step 3: Add lint/format scripts to `package.json`**

```json
"format": "biome format --write .",
"lint": "biome lint .",
"check": "biome check ."
```

- [ ] **Step 4: Format the codebase and verify the check passes**

Run:
```bash
pnpm format
pnpm check
```
Expected: `pnpm check` exits 0 (after `format` normalized the scaffolded files).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(lint): add Biome lint + format"
```

---

## Task 7: Set up Vitest with a passing unit test

**Files:**
- Modify: `package.json` (deps + scripts), `vite.config.ts`
- Create: `vitest.setup.ts`, `src/lib/version.ts`, `src/lib/version.test.ts`

- [ ] **Step 1: Install Vitest and React Testing Library**

Run:
```bash
pnpm add -D -E vitest @vitest/coverage-v8 jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event fast-check
```

- [ ] **Step 2: Create the test setup file**

`vitest.setup.ts`:
```typescript
import '@testing-library/jest-dom/vitest'
```

- [ ] **Step 3: Add the `test` config to `vite.config.ts`**

```typescript
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { modulePreload: { polyfill: false } },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    coverage: { provider: 'v8', reporter: ['text', 'lcov'] },
  },
})
```

- [ ] **Step 4: Write the failing test**

`src/lib/version.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { APP_NAME } from './version'

describe('APP_NAME', () => {
  it('is HardFit', () => {
    expect(APP_NAME).toBe('HardFit')
  })
})
```

- [ ] **Step 5: Run it to confirm it fails**

Add to `package.json` scripts first:
```json
"test": "vitest run",
"test:watch": "vitest"
```
Run:
```bash
pnpm test
```
Expected: FAIL — `Failed to resolve import "./version"` (the module doesn't exist yet).

- [ ] **Step 6: Write the minimal implementation**

`src/lib/version.ts`:
```typescript
export const APP_NAME = 'HardFit'
```

- [ ] **Step 7: Run it to confirm it passes**

Run:
```bash
pnpm test
```
Expected: PASS (1 test).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "test: set up Vitest with a passing sample unit test"
```

---

## Task 8: Add a Playwright smoke + accessibility e2e test

**Files:**
- Modify: `package.json` (deps + scripts), `.gitignore`
- Create: `playwright.config.ts`, `e2e/smoke.spec.ts`

- [ ] **Step 1: Install Playwright, the axe integration, and wrangler; then install the Chromium browser**

Run:
```bash
pnpm add -D -E @playwright/test @axe-core/playwright wrangler
pnpm exec playwright install chromium
```
Expected: Chromium downloads. (Browser binaries live outside the repo; nothing to commit from the download.) **`wrangler` is needed because Vite's `pnpm preview` does NOT apply `public/_headers`, so it can't exercise the real CSP — `wrangler pages dev` does.** If the install fails on a blocked build script for `workerd` (wrangler's runtime), review and add it to `onlyBuiltDependencies` (Task 3), then re-run.

- [ ] **Step 2: Create `playwright.config.ts` that serves the built app *under the real Cloudflare CSP***

The webServer runs `wrangler pages dev dist`, which applies `public/_headers` (CSP + security headers) and `public/_redirects` exactly as Cloudflare Pages will in production — closing the local-vs-prod gap. It serves the **prebuilt** `dist/` (no rebuild), so run `pnpm build` before `pnpm e2e` locally; in CI the Build step runs first.
```typescript
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: 'html',
  use: { baseURL: 'http://127.0.0.1:4173', trace: 'on-first-retry' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'pnpm exec wrangler pages dev dist --port 4173 --ip 127.0.0.1',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
```

- [ ] **Step 3: Write the smoke + axe + CSP-regression tests**

`e2e/smoke.spec.ts` — the third test fails the build if the CSP blocks any script (the exact production blank-page failure mode):
```typescript
import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test('renders the app shell', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'HardFit' })).toBeVisible()
})

test('has no critical accessibility violations', async ({ page }) => {
  await page.goto('/')
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze()
  expect(results.violations).toEqual([])
})

test('produces no console errors under the real CSP', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
  page.on('pageerror', (err) => errors.push(err.message))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'HardFit' })).toBeVisible()
  expect(errors).toEqual([]) // CSP violations surface here as "Refused to ..." errors
})
```

- [ ] **Step 4: Ignore Playwright artifacts**

Append to `.gitignore`:
```
/test-results/
/playwright-report/
/playwright/.cache/
```

- [ ] **Step 5: Add the e2e script and run the tests (build first)**

Add to `package.json` scripts:
```json
"e2e": "playwright test"
```
Run (build first so `dist/` exists for `wrangler pages dev`):
```bash
pnpm build && pnpm e2e
```
Expected: PASS (3 tests), served under the real CSP. Common minimal fixes if a test fails: the axe test flagging missing `lang` → set `<html lang="en">` in `index.html`; the console-error test reporting `Refused to ... 'script-src'` → confirm `build.modulePreload.polyfill` is `false` (Task 5/7) and that `dist/index.html` has no inline executable `<script>` (see Task 14).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test(e2e): add Playwright smoke + axe accessibility check"
```

---

## Task 9: Add a license-compliance gate

**Files:**
- Modify: `package.json` (dep + script)

- [ ] **Step 1: Install a license checker**

Run:
```bash
pnpm add -D -E license-checker-rseidelsohn
```

- [ ] **Step 2: Add a `licenses` script that fails on non-permissive licenses**

Add to `package.json` scripts:
```json
"licenses": "license-checker-rseidelsohn --production --onlyAllow 'MIT;ISC;Apache-2.0;BSD-2-Clause;BSD-3-Clause;0BSD;BlueOak-1.0.0;CC0-1.0;Unlicense;Python-2.0;BSL-1.0' --excludePrivatePackages"
```

- [ ] **Step 3: Run it**

Run:
```bash
pnpm licenses
```
Expected: exits 0. If it reports a disallowed/unknown license, investigate that dependency; add the license to the allow-list **only** if it is genuinely permissive (do not blanket-allow).

- [ ] **Step 4: Commit**

```bash
git add package.json pnpm-lock.yaml
git commit -m "chore(legal): add production license-compliance gate"
```

---

## Task 10: Add Cloudflare security headers + SPA fallback

**Files:**
- Create: `public/_headers`
- Create: `public/_redirects`

- [ ] **Step 1: Create the strict CSP + security headers**

`public/_headers`:
```
/*
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'
  Strict-Transport-Security: max-age=31536000; includeSubDomains
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  Permissions-Policy: camera=(), microphone=(), geolocation=(), interest-cohort=()
```
> Note: `style-src 'unsafe-inline'` is required because Tailwind/React inject some inline styles; everything else is locked to `'self'`. `worker-src 'self' blob:` allows the Vite-emitted Web Worker (added in M1). Revisit and tighten in M6.

- [ ] **Step 2: Create the explicit SPA fallback**

`public/_redirects`:
```
/* /index.html 200
```

- [ ] **Step 3: Verify they ship into the build**

Run:
```bash
pnpm build
ls dist/_headers dist/_redirects
```
Expected: both files are present in `dist/` (Vite copies `public/` verbatim).

- [ ] **Step 4: Commit**

```bash
git add public/_headers public/_redirects
git commit -m "feat(security): add CSP + security headers and SPA fallback for Cloudflare"
```

---

## Task 11: Add the GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the CI workflow (actions referenced by tag for now; pinned to SHA in Step 3)**

`.github/workflows/ci.yml`:
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
permissions:
  contents: read
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - name: Enable Corepack
        run: corepack enable
      - name: Install (frozen lockfile)
        run: pnpm install --frozen-lockfile
      - name: Audit
        run: pnpm audit --audit-level=high
      - name: Typecheck
        run: pnpm typecheck
      - name: Lint + format check
        run: pnpm check
      - name: Unit tests
        run: pnpm test
      - name: Licenses
        run: pnpm licenses
      - name: Build
        run: pnpm build
      - name: Install Playwright browser
        run: pnpm exec playwright install --with-deps chromium
      - name: E2E tests
        run: pnpm e2e   # Playwright serves the prebuilt dist/ via `wrangler pages dev` → runs under the real CSP; no rebuild.
```
> The `Build` step produces `dist/`; the `E2E tests` step's Playwright `webServer` (`wrangler pages dev dist`) reuses it rather than rebuilding — so the build runs exactly once, and e2e exercises the production CSP/headers.

- [ ] **Step 2: Validate the workflow locally by running each gate**

Run:
```bash
pnpm install --frozen-lockfile && pnpm typecheck && pnpm check && pnpm test && pnpm licenses && pnpm build && pnpm e2e
```
Expected: every step exits 0. (`pnpm audit` may report nothing or advisories; if it fails on a transitive advisory with no fix, document it and add `|| true` only as a last resort with a comment — prefer upgrading.)

- [ ] **Step 3: Pin all third-party Actions to commit SHAs**

Install `pinact` (macOS) and pin:
```bash
brew install pinact
pinact run
```
Expected: `pinact` rewrites `uses: actions/checkout@v4` → `uses: actions/checkout@<40-char-sha> # v4` (and likewise for `setup-node`), eliminating mutable-tag supply-chain risk. If `brew` is unavailable, pin manually by resolving each tag with `gh api repos/<owner>/<repo>/commits/<tag> --jq .sha` and replacing the tag with the SHA (keep the version in a trailing comment).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add quality-gate workflow (audit/typecheck/lint/test/license/build/e2e), Actions pinned by SHA"
```

---

## Task 12: Add Dependabot

**Files:**
- Create: `.github/dependabot.yml`

- [ ] **Step 1: Create the Dependabot config**

`.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: npm
    directory: "/"
    schedule: { interval: weekly }
    open-pull-requests-limit: 10
    groups:
      all-dependencies:
        patterns: ["*"]
  - package-ecosystem: github-actions
    directory: "/"
    schedule: { interval: weekly }
```

- [ ] **Step 2: Commit**

```bash
git add .github/dependabot.yml
git commit -m "chore(security): enable Dependabot for npm + GitHub Actions"
```
> After pushing to GitHub: in the repo's **Settings → Code security**, enable **Dependabot alerts** and the **Dependency review** action, and install the free **Socket** GitHub App for PR-time supply-chain scanning. (These are dashboard toggles, not code.)

---

## Task 13: Add README and CLAUDE.md

**Files:**
- Create: `README.md`
- Create: `CLAUDE.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# HardFit

Free, bilingual (EN/ES), client-side web app for fitting data to probability distributions — a modern, zero-install alternative that beats EasyFit. See `docs/hardfit-technical-strategy.md` for the full strategy.

## Requirements
- Node ≥ 22.12 (`.nvmrc` → `nvm use`)
- pnpm via Corepack (`corepack enable`)

## Commands
```bash
pnpm install            # install (exact pins, scripts blocked unless allowlisted)
pnpm dev                # start the dev server
pnpm build              # production build → dist/
pnpm typecheck          # tsc --noEmit
pnpm check              # Biome lint + format check
pnpm test               # Vitest unit tests
pnpm build && pnpm e2e  # Playwright smoke + a11y + CSP tests (serves dist under the real CSP via wrangler)
pnpm licenses           # license-compliance gate
```

## Deploy
Auto-deploys to Cloudflare Pages on push to `main` (build: `pnpm build`, output: `dist`).
````

- [ ] **Step 2: Write `CLAUDE.md`**

```markdown
# HardFit — Project Memory

Client-side, stateless (no server/DB/accounts), bilingual (EN/ES) distribution-fitting SPA that must beat EasyFit. Full strategy: `docs/hardfit-technical-strategy.md`. Plans: `docs/superpowers/plans/`.

## Toolchain
- pnpm via Corepack ONLY (never npm/yarn). Node ≥ 22.12.
- Commands: `pnpm dev|build|typecheck|check|test|e2e|licenses`.

## Supply-chain rules (non-negotiable)
- Dependency install scripts are blocked by default (`pnpm-workspace.yaml` → `strictDepBuilds: true` + `onlyBuiltDependencies`). Never blanket-allow; review each.
- `minimumReleaseAge: 1440` cooldown; exact version pins (`save-exact`); commit the lockfile; CI uses `--frozen-lockfile`.
- Vendor critical libs; no runtime CDNs (would break the CSP). Pin GitHub Actions by SHA.

## Architecture invariants (apply from M1 on)
- `src/engine/` is PURE TypeScript: zero DOM/React/Comlink/Worker imports; runs in Node; TDD'd against scipy/R fixtures.
- All heavy compute runs in one Web Worker (Comlink). UI talks only to the typed worker proxy.
- Cancellation is cooperative (AbortSignal is NOT transferable).

## Stats rules (apply from M2 on)
- Rank by AICc (fallback AIC); the raw KS p-value is INVALID for fitted params → Lilliefors/bootstrap; label every p-value's provenance; bootstrap only the top 3–5; standard errors from a fresh Hessian.

## i18n rules (apply from M4 on)
- All user-facing strings via i18next; standalone numbers via the Intl formatter module; EN/ES files kept in sync via CI; `load: 'languageOnly'`.

## Working agreement
- TDD with frequent commits. Use **opus** for all subagents. Show real command output before claiming done.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add README and CLAUDE.md project memory"
```

---

## Task 14: Deploy to Cloudflare Pages and verify it's live

**Files:** none (dashboard + git remote configuration)

- [ ] **Step 1: Create a GitHub remote and push**

Create a repo on GitHub (e.g. `hardfit`) and push:
```bash
git remote add origin git@github.com:<your-username>/hardfit.git
git push -u origin main
```
Expected: code on GitHub; the CI workflow runs and goes green.

- [ ] **Step 2: Connect the repo to Cloudflare Pages**

In the Cloudflare dashboard → **Workers & Pages → Create → Pages → Connect to Git** → select the repo. Set:
- Framework preset: **Vite** (or None)
- Build command: `pnpm build`
- Build output directory: `dist`
- Environment variable: `NODE_VERSION = 22`

(If a CLI login is needed, run it yourself in this session by typing `! wrangler login` so the output lands here.)

- [ ] **Step 3: Verify the deployment is live and headers apply**

First confirm the production HTML has no inline executable `<script>` that the CSP would block (the blank-page failure mode):
```bash
pnpm build && grep -nE '<script(?![^>]*\bsrc=)[^>]*>[^<]' dist/index.html || echo "OK: no inline executable scripts"
```
Expected: `OK: no inline executable scripts`. (The e2e "no console errors under the real CSP" test from Task 8 already gates this in CI; this is a fast manual double-check.)

Then, after the first deploy, run (replace with your real `*.pages.dev` URL):
```bash
curl -sI https://hardfit.pages.dev | grep -i -E 'content-security-policy|strict-transport|x-content-type'
```
Expected: the CSP + security headers from `public/_headers` are present. Finally, open the URL in a browser — the "HardFit" shell must render (not a blank page) with a clean console.

- [ ] **Step 4: Tag the milestone**

```bash
git tag m0-scaffold
git push origin m0-scaffold
```

---

## Self-Review

**1. Spec coverage (M0 slice of the strategy):**
- Vite + React 19 + TS strictest → Tasks 2, 4. ✓
- pnpm via Corepack + supply-chain hardening (script-block, cooldown, exact pins, lockfile) → Tasks 1, 2, 3. ✓
- Tailwind v4 → Task 5. ✓
- Biome → Task 6. ✓
- Vitest + Playwright + axe + **CSP-regression test under the real Cloudflare CSP (via `wrangler pages dev`)** → Tasks 7, 8. ✓ (closes the `pnpm preview` doesn't-apply-`_headers` gap)
- License gate → Task 9. ✓
- CSP + security headers + SPA fallback → Task 10. ✓
- CI (audit/typecheck/lint/test/license/build/e2e, Actions pinned by SHA) → Task 11. ✓
- Dependabot + Dependency-review + Socket → Task 12. ✓
- Empty deploy to Cloudflare Pages → Task 14. ✓
- CLAUDE.md + README → Task 13. ✓
- *Deferred to later milestones (correctly out of M0):* the `engine/`, Web Worker/Comlink, charts, i18n, persistence, PWA — these begin in M1.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to" placeholders; every code step shows real content. The only intentional lookups are (a) the `onlyBuiltDependencies` list, which is reviewed against the install's own output, and (b) Action SHAs, which `pinact` resolves automatically — both are correct practice, not omissions.

**3. Type/name consistency:** `APP_NAME` (Task 7) is the only shared symbol and is referenced consistently. Script names (`typecheck`, `check`, `test`, `e2e`, `licenses`, `build`, `dev`) are used identically in `package.json`, the README, CLAUDE.md, and CI.

---

## Execution Handoff

Next milestones (each gets its own plan, written just-in-time before execution):
- **M1** — vertical slice: pure `engine/` for ~5 distributions (MLE) + KS + AICc, Comlink worker, CSV import, fit-all → ranked table, one chart, deployed end-to-end.
- **M2** — full engine breadth (~53 distributions, all GoF, AIC/AICc/BIC, bootstrap) TDD'd against scipy/R fixtures.
- **M3** — full visualization + import/export. **M4** — i18n (EN/ES). **M5** — polish/a11y/perf/onboarding. **M6** — deploy hardening + share links + PWA.
