# HardFit — Project Memory

Client-side, stateless (no server/DB/accounts), bilingual (EN/ES) distribution-fitting SPA that must beat EasyFit. Full strategy: `docs/hardfit-technical-strategy.md`. Plans: `docs/superpowers/plans/`.

## Toolchain
- pnpm via Corepack ONLY (never npm/yarn). Node ≥ 22.12.
- Commands: `pnpm dev|build|typecheck|check|test|e2e`, and `pnpm run licenses` (bare `pnpm licenses` is a built-in subcommand).

## Supply-chain rules (non-negotiable)
- Dependency install scripts are blocked by default (`pnpm-workspace.yaml` → `strictDepBuilds: true` + `onlyBuiltDependencies`). Never blanket-allow; review each.
- `minimumReleaseAge: 1440` cooldown; exact version pins (`save-exact`); commit the lockfile; CI uses `--frozen-lockfile`.
- Build-time-only tooling (Tailwind, Vite plugins) stays in devDependencies. Vendor critical libs; no runtime CDNs (would break the CSP). Pin GitHub Actions by SHA.

## Architecture invariants (apply from M1 on)
- `src/engine/` is PURE TypeScript: zero DOM/React/Comlink/Worker imports; runs in Node; TDD'd against scipy/R fixtures.
- All heavy compute runs in one Web Worker (Comlink). UI talks only to the typed worker proxy.
- Cancellation is cooperative (AbortSignal is NOT transferable).

## Stats rules (apply from M2 on)
- Rank by AICc (fallback AIC); the raw KS p-value is INVALID for fitted params → Lilliefors/bootstrap; label every p-value's provenance; bootstrap only the top 3–5; standard errors from a fresh Hessian.

## i18n rules (apply from M4 on)
- All user-facing strings via i18next; standalone numbers via the Intl formatter module; EN/ES files kept in sync via CI; `load: 'languageOnly'`.

## Working agreement
- TDD with frequent commits. Use Opus for all subagents. Show real command output before claiming done.
