import { describe, expect, it } from 'vitest'
import { APP_NAME } from './version'

describe('APP_NAME', () => {
  it('is HardFit', () => {
    expect(APP_NAME).toBe('HardFit')
  })
})
