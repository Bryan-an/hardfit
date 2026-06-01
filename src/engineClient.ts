import * as Comlink from 'comlink'
import type { FitAllResult } from './engine/index'
import type { EngineApi } from './engine.worker' // TYPE-ONLY: keeps worker code out of the main bundle

// Vite statically detects this exact form (string literal + import.meta.url). Do NOT parameterize.
const worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' })
const engine = Comlink.wrap<EngineApi>(worker)

export async function runFitAll(
  data: readonly number[],
  onProgress?: (completed: number, total: number) => void,
): Promise<FitAllResult> {
  const samples = Float64Array.from(data)
  // transfer the buffer (zero-copy); `samples` is neutered on this side afterward.
  return engine.fitAll(
    Comlink.transfer(samples, [samples.buffer]),
    onProgress ? Comlink.proxy(onProgress) : undefined,
  )
}
