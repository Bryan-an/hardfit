import type { Data, Layout } from 'plotly.js'
import Plotly from 'plotly.js-cartesian-dist-min'
import { useEffect, useRef } from 'react'
import { CHART_HEIGHT_PX } from './chart-constants'

export function PlotlyChart({ data, layout }: { data: Data[]; layout?: Partial<Layout> }) {
  const divRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = divRef.current
    if (!el) return
    void Plotly.react(el, data, layout ?? {}, { responsive: true })
    return () => {
      // Capture `el`: divRef.current may be null at cleanup (React 19 StrictMode double-mount).
      Plotly.purge(el)
    }
  }, [data, layout])
  return <div ref={divRef} style={{ width: '100%', height: CHART_HEIGHT_PX }} />
}
