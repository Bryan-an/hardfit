import { describe, expect, it } from 'vitest'
import { parseNumbers } from './parseNumbers'

describe('parseNumbers', () => {
  it('parses newline/comma/space separated numbers', () => {
    expect(parseNumbers('1\n2.5, 3\t4 5')).toEqual([1, 2.5, 3, 4, 5])
  })
  it('ignores blank lines, a header row, and non-numeric tokens', () => {
    expect(parseNumbers('value\n1\n\n2\nNA\n3')).toEqual([1, 2, 3])
  })
  it('handles a single CSV column with a header', () => {
    expect(parseNumbers('x\r\n10\r\n20\r\n30')).toEqual([10, 20, 30])
  })
  it('returns [] for empty input', () => expect(parseNumbers('   ')).toEqual([]))
})
