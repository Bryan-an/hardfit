import { MIN_FIT_SAMPLE_SIZE } from './constants'
import { DISTRIBUTIONS } from './distributions/index'
import { ksStatistic } from './gof'
import { aic, aicc, bic, logLik, rankByAICc } from './selection'
import type { Fit, FitAllResult, FitFailure, RankedFit } from './types'

export interface FitAllOptions {
  onProgress?: (completed: number, total: number) => void
}

/** Fit every distribution to `data`, collecting successes and failures (never throwing
 *  per-distribution), then rank the successes by AICc. Throws only when the sample is too
 *  small to score. Reports progress after each distribution via `opts.onProgress`. */
export function fitAll(data: readonly number[], opts: FitAllOptions = {}): FitAllResult {
  const n = data.length
  if (n < MIN_FIT_SAMPLE_SIZE) {
    throw new Error(`fitAll: need at least ${MIN_FIT_SAMPLE_SIZE} data points`)
  }
  const fits: Fit[] = []
  const failures: FitFailure[] = []
  const total = DISTRIBUTIONS.length
  DISTRIBUTIONS.forEach((dist, i) => {
    try {
      const params = dist.fit(data)
      const ll = logLik(data, (x) => dist.logpdf(x, params))
      if (!Number.isFinite(ll)) throw new Error('non-finite log-likelihood')
      fits.push({
        name: dist.name,
        label: dist.label,
        k: dist.k,
        params,
        logLik: ll,
        aic: aic(ll, dist.k),
        aicc: aicc(ll, dist.k, n),
        bic: bic(ll, dist.k, n),
        ks: ksStatistic(data, (x) => dist.cdf(x, params)),
      })
    } catch (e) {
      failures.push({
        name: dist.name,
        label: dist.label,
        error: e instanceof Error ? e.message : String(e),
      })
    }
    opts.onProgress?.(i + 1, total)
  })
  // `Fit` already satisfies `Rankable` (name/logLik/k); rankByAICc returns `(Fit & Ranked)[]`,
  // whose field set is exactly `RankedFit`, so this assigns with no cast.
  const ranked: RankedFit[] = rankByAICc(fits, n)
  return { ranked, failures, n }
}
