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
