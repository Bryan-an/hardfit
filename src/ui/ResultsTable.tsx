import type { BootstrapResult, ParamCI } from '../engine/index'
import { DistributionName, type GofResult, type RankedFit } from '../engine/types'

/** Decimal places shown for the numeric diagnostic columns. */
const DECIMAL_PLACES = 3
/** Placeholder rendered for a non-finite value (e.g. +Infinity AICc). */
const EMPTY_VALUE = '—'
/** Marker shown in place of a p-value when only a diagnostic statistic is available. */
const DIAGNOSTIC_LABEL = 'diag.'
/** Prefix labelling a rendered p-value (keeps the secondary text scannable). */
const PVALUE_PREFIX = 'p='
/** Method label for a parametric-bootstrap p-value (surfaced via the cell's `title`). */
const BOOTSTRAP_METHOD: GofResult['method'] = 'bootstrap'
/**
 * Gamma's `fit` stores `shape`, `scale`, AND `rate` (scale = 1/rate), so its bootstrap
 * `paramCIs` carries a redundant `scale` entry. The locked convention is shape + rate, so
 * the params cell omits this key for gamma rather than showing the duplicate.
 */
const GAMMA_REDUNDANT_PARAM = 'scale'

const fmt = (v: number): string => (Number.isFinite(v) ? v.toFixed(DECIMAL_PLACES) : EMPTY_VALUE)

/** Format a confidence interval as a `[lo, hi]` bracket at the diagnostic precision. */
const formatCI = ([lo, hi]: readonly [number, number]): string => `[${fmt(lo)}, ${fmt(hi)}]`

/**
 * Renders one goodness-of-fit cell: the statistic as the primary value, with the
 * p-value (or a diagnostic marker when unavailable) as secondary text. The p-value
 * derivation method is surfaced via a `title` tooltip so closed-form vs table vs
 * diagnostic vs bootstrap stays visible without cluttering the table.
 */
function GofCell({ result }: { result: GofResult }) {
  const secondary =
    result.pValue !== null ? `${PVALUE_PREFIX}${fmt(result.pValue)}` : DIAGNOSTIC_LABEL
  return (
    <td className="py-1 pr-3" title={result.method}>
      {fmt(result.statistic)} <sup className="text-xs text-slate-500">{secondary}</sup>
    </td>
  )
}

/**
 * Which parameter keys to show for a row's CIs, in display order. Gamma drops the redundant
 * `scale` (it keeps shape + rate, the locked convention); every other distribution shows all
 * of its fitted parameters.
 */
function ciParamEntries(name: string, paramCIs: Record<string, ParamCI>): [string, ParamCI][] {
  return Object.entries(paramCIs).filter(
    ([key]) => !(name === DistributionName.Gamma && key === GAMMA_REDUNDANT_PARAM),
  )
}

/**
 * Per-parameter point estimate + bootstrap confidence interval, one line each
 * (e.g. `shape 2.34 [1.81, 3.02]`). Uses the BCa interval; non-finite values fall back to
 * the shared empty-value placeholder via `fmt`/`formatCI`.
 */
function ParamsCell({ name, paramCIs }: { name: string; paramCIs: Record<string, ParamCI> }) {
  return (
    <td className="py-1 pr-3 align-top">
      <ul className="list-none">
        {ciParamEntries(name, paramCIs).map(([key, ci]) => (
          <li key={key} className="whitespace-nowrap">
            <span className="font-medium">{key}</span> {fmt(ci.point)}{' '}
            <span className="text-slate-500">{formatCI(ci.bca)}</span>
          </li>
        ))}
      </ul>
    </td>
  )
}

/** A `GofResult` carrying the bootstrap p-value for a statistic (KS/AD/CvM share `GofCell`). */
const bootstrapGof = (statistic: number, pValue: number): GofResult => ({
  statistic,
  pValue,
  method: BOOTSTRAP_METHOD,
})

export function ResultsTable({
  ranked,
  bootstrap,
}: {
  ranked: RankedFit[]
  bootstrap?: BootstrapResult
}) {
  const hasBootstrap = bootstrap !== undefined && Object.keys(bootstrap).length > 0
  return (
    <table className="w-full text-sm border-collapse">
      <caption className="sr-only">Distributions ranked by AICc (best first)</caption>
      <thead>
        <tr className="text-left border-b border-slate-300">
          <th scope="col" className="py-1 pr-3">
            #
          </th>
          <th scope="col" className="py-1 pr-3">
            Distribution
          </th>
          {hasBootstrap && (
            <th scope="col" className="py-1 pr-3">
              Parameters (CI)
            </th>
          )}
          <th scope="col" className="py-1 pr-3">
            AICc
          </th>
          <th scope="col" className="py-1 pr-3">
            ΔAICc
          </th>
          <th scope="col" className="py-1 pr-3">
            Weight
          </th>
          <th scope="col" className="py-1 pr-3">
            KS (D)
          </th>
          <th scope="col" className="py-1 pr-3">
            log-lik
          </th>
          <th scope="col" className="py-1 pr-3">
            AD
          </th>
          <th scope="col" className="py-1 pr-3">
            CvM
          </th>
          <th scope="col" className="py-1 pr-3">
            χ²
          </th>
        </tr>
      </thead>
      <tbody>
        {ranked.map((r) => {
          const bs = bootstrap?.[r.name]
          // With bootstrap p-values, KS gains a secondary it lacks in the M2.1 display and
          // AD/CvM swap their diagnostic/closed-form marker for the rigorous bootstrap p.
          const ksResult: GofResult = bs
            ? bootstrapGof(r.ks, bs.gofPValues.ks)
            : { statistic: r.ks, pValue: null, method: 'diagnostic' }
          const adResult = bs ? bootstrapGof(r.ad.statistic, bs.gofPValues.ad) : r.ad
          const cvmResult = bs ? bootstrapGof(r.cvm.statistic, bs.gofPValues.cvm) : r.cvm
          return (
            <tr key={r.name} className="border-b border-slate-100">
              <td className="py-1 pr-3">{r.rank}</td>
              <th scope="row" className="py-1 pr-3 font-medium text-left">
                {r.label}
              </th>
              {hasBootstrap &&
                (bs ? (
                  <ParamsCell name={r.name} paramCIs={bs.paramCIs} />
                ) : (
                  <td className="py-1 pr-3 text-slate-400">{EMPTY_VALUE}</td>
                ))}
              <td className="py-1 pr-3">{fmt(r.aicc)}</td>
              <td className="py-1 pr-3">{fmt(r.deltaAICc)}</td>
              <td className="py-1 pr-3">{fmt(r.weight)}</td>
              {bs ? <GofCell result={ksResult} /> : <td className="py-1 pr-3">{fmt(r.ks)}</td>}
              <td className="py-1 pr-3">{fmt(r.logLik)}</td>
              <GofCell result={adResult} />
              <GofCell result={cvmResult} />
              <GofCell result={r.chiSquared} />
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
