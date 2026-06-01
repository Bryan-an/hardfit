import { useState } from 'react'
import { parseNumbers } from '../lib/parseNumbers'

/** Built-in example column so a visitor can try the app with one click. */
const SAMPLE_DATASET = '2.1 3.4 1.8 5.2 2.9 4.1 3.0 2.5 6.0 3.7 1.2 4.8 2.2 3.9 5.5 2.7 3.1 4.4'
/** File extensions accepted by the upload input. */
const ACCEPTED_FILE_TYPES = '.csv,.txt'

export function DataInput({ onData }: { onData: (data: number[]) => void }) {
  const [text, setText] = useState('')
  const submit = (raw: string) => onData(parseNumbers(raw))

  return (
    <div className="flex flex-col gap-2">
      <textarea
        aria-label="Data values"
        className="border border-slate-300 rounded p-2 font-mono text-sm h-32"
        placeholder="Paste numbers (one per line or comma/space separated)"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex gap-2 items-center flex-wrap">
        <button
          type="button"
          className="px-3 py-1 rounded bg-slate-900 text-white"
          onClick={() => submit(text)}
        >
          Fit distributions
        </button>
        <button
          type="button"
          className="px-3 py-1 rounded border border-slate-300"
          onClick={() => {
            setText(SAMPLE_DATASET)
            submit(SAMPLE_DATASET)
          }}
        >
          Load sample
        </button>
        <input
          type="file"
          accept={ACCEPTED_FILE_TYPES}
          aria-label="Upload data file"
          onChange={async (e) => {
            const file = e.target.files?.[0]
            if (file) {
              const content = await file.text()
              setText(content)
              submit(content)
            }
          }}
        />
      </div>
    </div>
  )
}
