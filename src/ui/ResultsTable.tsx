import type { GofResult, RankedFit } from '../engine/types'

/** Decimal places shown for the numeric diagnostic columns. */
const DECIMAL_PLACES = 3
/** Placeholder rendered for a non-finite value (e.g. +Infinity AICc). */
const EMPTY_VALUE = '—'
/** Marker shown in place of a p-value when only a diagnostic statistic is available. */
const DIAGNOSTIC_LABEL = 'diag.'
/** Prefix labelling a rendered p-value (keeps the secondary text scannable). */
const PVALUE_PREFIX = 'p='

const fmt = (v: number): string => (Number.isFinite(v) ? v.toFixed(DECIMAL_PLACES) : EMPTY_VALUE)

/**
 * Renders one goodness-of-fit cell: the statistic as the primary value, with the
 * p-value (or a diagnostic marker when unavailable) as secondary text. The p-value
 * derivation method is surfaced via a `title` tooltip so closed-form vs table vs
 * diagnostic stays visible without cluttering the table.
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

export function ResultsTable({ ranked }: { ranked: RankedFit[] }) {
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
        {ranked.map((r) => (
          <tr key={r.name} className="border-b border-slate-100">
            <td className="py-1 pr-3">{r.rank}</td>
            <th scope="row" className="py-1 pr-3 font-medium text-left">
              {r.label}
            </th>
            <td className="py-1 pr-3">{fmt(r.aicc)}</td>
            <td className="py-1 pr-3">{fmt(r.deltaAICc)}</td>
            <td className="py-1 pr-3">{fmt(r.weight)}</td>
            <td className="py-1 pr-3">{fmt(r.ks)}</td>
            <td className="py-1 pr-3">{fmt(r.logLik)}</td>
            <GofCell result={r.ad} />
            <GofCell result={r.cvm} />
            <GofCell result={r.chiSquared} />
          </tr>
        ))}
      </tbody>
    </table>
  )
}
