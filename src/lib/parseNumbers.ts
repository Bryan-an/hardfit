/** Token separators for pasted/CSV input: any run of whitespace, commas, or semicolons. */
const SEPARATORS = /[\s,;]+/

/** Parse a flat list of numbers from pasted text or a single CSV column.
 *  Splits on any whitespace/comma/semicolon; drops tokens that aren't finite numbers
 *  (so a header row like "value" or an "NA" cell is ignored). */
export function parseNumbers(text: string): number[] {
  const out: number[] = []
  for (const tok of text.split(SEPARATORS)) {
    if (tok === '') continue
    const v = Number(tok)
    if (Number.isFinite(v)) out.push(v)
  }
  return out
}
