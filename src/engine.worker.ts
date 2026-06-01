import * as Comlink from 'comlink'
import { type FitAllResult, fitAll } from './engine/index'

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
}

export type EngineApi = typeof api // client imports this TYPE only
Comlink.expose(api)
