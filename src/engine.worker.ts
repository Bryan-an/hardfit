import * as Comlink from 'comlink'
import {
  type BootstrapOptions,
  type BootstrapResult,
  bootstrapTopFits,
  DISTRIBUTIONS,
  type FitAllResult,
  fitAll,
  type RankedFit,
} from './engine/index'

/**
 * Cooperative cancellation flag for the in-flight bootstrap. An AbortSignal is NOT
 * transferable through Comlink (same constraint as M1's worker), so the client instead
 * calls `cancelBootstrap()`, which flips this flag; the engine's chunked loop polls it via
 * `isCancelled` at each chunk boundary (right after it yields a 0 ms timer, which is when
 * the worker services the queued cancel message) and throws to abort.
 */
let cancelled = false

const api = {
  /** Fit all distributions. Streams progress via a Comlink.proxy'd callback. */
  fitAll(
    samples: Float64Array,
    onProgress?: (completed: number, total: number) => void,
  ): FitAllResult {
    // samples.buffer was transferred in -> use directly. Engine wants number[].
    const data = Array.from(samples)
    // `onProgress` is optional under exactOptionalPropertyTypes: only set the
    // property when defined, so we never pass `{ onProgress: undefined }`.
    return fitAll(data, onProgress ? { onProgress } : {})
  },

  /**
   * Run the top-k parametric bootstrap (chunked, cooperatively cancellable). Resets the
   * cancel flag SYNCHRONOUSLY before any await so a stale `cancelBootstrap()` from a prior
   * run cannot abort this one, then delegates to the pure engine orchestrator. A cancel
   * mid-run rejects with the engine's cancellation error (propagated across Comlink).
   */
  async bootstrapTopFits(
    samples: Float64Array,
    ranked: readonly RankedFit[],
    opts: BootstrapOptions,
    onProgress?: (fraction: number) => void,
  ): Promise<BootstrapResult> {
    cancelled = false
    const data = Array.from(samples)
    return bootstrapTopFits(
      DISTRIBUTIONS,
      data,
      ranked,
      opts,
      onProgress ? (fraction) => onProgress(fraction) : undefined,
      () => cancelled,
    )
  },

  /** Request cancellation of the in-flight bootstrap (cooperative; polled each chunk). */
  cancelBootstrap(): void {
    cancelled = true
  },
}

export type EngineApi = typeof api // client imports this TYPE only
Comlink.expose(api)
