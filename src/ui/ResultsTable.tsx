import type { RankedFit } from '../engine/types'

/** Decimal places shown for the numeric diagnostic columns. */
const DECIMAL_PLACES = 3
/** Placeholder rendered for a non-finite value (e.g. +Infinity AICc). */
const EMPTY_VALUE = '—'

const fmt = (v: number): string => (Number.isFinite(v) ? v.toFixed(DECIMAL_PLACES) : EMPTY_VALUE)

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
          </tr>
        ))}
      </tbody>
    </table>
  )
}
