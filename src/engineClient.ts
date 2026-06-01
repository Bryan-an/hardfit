import * as Comlink from 'comlink'
import type { BootstrapOptions, BootstrapResult, FitAllResult, RankedFit } from './engine/index'
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

/**
 * Run the top-k parametric bootstrap in the worker. Transfers the sample buffer (zero-copy)
 * and proxies the progress callback (a worker-side call into a main-thread function). Cancel
 * mid-run with {@link cancelBootstrap}; the returned promise then rejects with the engine's
 * cancellation error (an ordinary `Error` across the Comlink boundary).
 */
export async function runBootstrap(
  data: readonly number[],
  ranked: readonly RankedFit[],
  opts: BootstrapOptions,
  onProgress?: (fraction: number) => void,
): Promise<BootstrapResult> {
  const samples = Float64Array.from(data)
  return engine.bootstrapTopFits(
    Comlink.transfer(samples, [samples.buffer]),
    ranked,
    opts,
    onProgress ? Comlink.proxy(onProgress) : undefined,
  )
}

/** Cooperatively cancel the in-flight bootstrap (the worker polls the flag each chunk). */
export async function cancelBootstrap(): Promise<void> {
  await engine.cancelBootstrap()
}
