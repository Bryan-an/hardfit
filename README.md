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
pnpm typecheck          # tsc -b (project references)
pnpm check              # Biome lint + format check
pnpm test               # Vitest unit tests
pnpm build && pnpm e2e  # Playwright smoke + a11y + CSP tests (serves dist under the real CSP via wrangler)
pnpm run licenses       # license-compliance gate (bare `pnpm licenses` is a built-in subcommand)
```

## Deploy
Auto-deploys to Cloudflare Pages on push to `main` (build: `pnpm build`, output: `dist`).
