import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

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
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text())
  })
  page.on('pageerror', (err) => errors.push(err.message))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'HardFit' })).toBeVisible()
  expect(errors).toEqual([])
})

test('fits the sample dataset end-to-end and renders ranked table + chart', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })
  page.on('pageerror', (e) => errors.push(e.message))

  await page.goto('/')
  await page.getByRole('button', { name: 'Load sample' }).click()

  // worker computes; ranked table appears. Assert the 5 M1 families are present (the full
  // catalog is larger — M2.3 Batch A added 8 more — but these five must always rank).
  await expect(page.getByRole('heading', { name: 'Ranked fits (by AICc)' })).toBeVisible({
    timeout: 15_000,
  })
  for (const label of ['Normal', 'Lognormal', 'Exponential', 'Gamma', 'Weibull']) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
  }
  // new GoF columns surfaced (Anderson-Darling, Chi-Squared)
  await expect(page.getByRole('columnheader', { name: 'AD' })).toBeVisible()
  await expect(page.getByRole('columnheader', { name: 'χ²' })).toBeVisible()
  // Plotly chart rendered
  await expect(page.locator('.plotly').first()).toBeVisible()

  // Background bootstrap completes: the "Parameters (CI)" column only renders once
  // runBootstrap resolves (top-3, B=999 takes a few seconds — hence the generous timeout),
  // and a per-parameter `[lo, hi]` confidence-interval bracket appears in the same render.
  await expect(page.getByRole('columnheader', { name: /Parameters/ })).toBeVisible({
    timeout: 30_000,
  })
  await expect(page.getByText(/\[\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\]/).first()).toBeVisible()

  expect(errors).toEqual([]) // no CSP violations / runtime errors (worker + Plotly + bootstrap under script-src 'self')
})
