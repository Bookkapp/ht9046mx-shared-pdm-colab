import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { lineOption, scatterOption } from '../charts'
import { EChart } from '../components/EChart'
import { Empty, ErrorBanner, PageHeading, Shell } from '../components/Shell'
import { number } from '../format'

const DEFAULT = [
  { machine_id: 'MX057', module_id: 1, metric: 'hp2' },
  { machine_id: 'MX070', module_id: 1, metric: 'lp2' },
]

export default function ComparePage() {
  const [catalog, setCatalog] = useState(null)
  const [fleet, setFleet] = useState(null)
  const [rows, setRows] = useState(DEFAULT)
  const [selectedDate, setSelectedDate] = useState('')
  const [normalized, setNormalized] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => { Promise.all([api.catalog(), api.fleet()]).then(([nextCatalog, nextFleet]) => { setCatalog(nextCatalog); setFleet(nextFleet); const names = nextFleet.machines.map((item) => item.machine_id); if (!names.includes('MX057') && names[0]) setRows((current) => current.map((row, index) => ({ ...row, machine_id: names[Math.min(index, names.length - 1)] }))) }).catch(setError) }, [])
  const run = async () => {
    setLoading(true); setError(null)
    try { setResult(await api.comparison({ selected_date: selectedDate || null, series: rows })) } catch (nextError) { setError(nextError) } finally { setLoading(false) }
  }
  useEffect(() => { if (catalog && fleet) run() }, [catalog, fleet]) // initial source-backed view

  const definitions = useMemo(() => (result?.series || []).map((item) => ({ key: item.id, label: item.label, color: item.color })), [result])
  const aligned = useMemo(() => {
    const byTime = new Map()
    ;(result?.series || []).forEach((series) => series.points.forEach((point) => byTime.set(point.event_time, { ...(byTime.get(point.event_time) || { event_time: point.event_time }), [series.id]: point.value })))
    return [...byTime.values()].sort((a, b) => a.event_time.localeCompare(b.event_time))
  }, [result])
  const trend = useMemo(() => lineOption(aligned, definitions, { normalize: normalized }), [aligned, definitions, normalized])
  const scatter = useMemo(() => scatterOption(result?.relationship?.points || [], result?.series?.[0]?.label || 'Series A', result?.series?.[1]?.label || 'Series B'), [result])

  const change = (index, key, value) => setRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, [key]: key === 'module_id' ? Number(value) : value } : row))
  return <Shell active="compare" error={error}>
    <PageHeading eyebrow="CROSS-MACHINE EVIDENCE" title="Compare any signal or model evidence" description="เลือก machine, module และ metric ได้ทั้งค่าดิบ, engineered feature, Robust Z, residual, Isolation Forest, LSTM และ GMM โดย correlation ใช้เฉพาะ timestamp ที่ align กัน" actions={<label className="date-field"><span>Source date</span><input type="date" value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)} /></label>} />
    <ErrorBanner error={error} />
    <section className="panel compare-builder"><div className="panel-title"><div><p className="eyebrow">SERIES BUILDER</p><h2>Comparison definition</h2><span>2–6 series · data is loaded from the same source contract as Machine Monitor</span></div><button className="primary-button" disabled={loading} onClick={run} type="button">{loading ? 'Loading…' : 'Run comparison'}</button></div>
      <div className="compare-rows">{rows.map((row, index) => <div className="compare-row" key={index}><span>{index + 1}</span><select value={row.machine_id} onChange={(event) => change(index, 'machine_id', event.target.value)}>{(fleet?.machines || []).map((item) => <option key={item.machine_id}>{item.machine_id}</option>)}</select><select value={row.module_id} onChange={(event) => change(index, 'module_id', event.target.value)}>{[1,2,3,4,5,6,8].map((item) => <option value={item} key={item}>Module {item}</option>)}</select><select value={row.metric} onChange={(event) => change(index, 'metric', event.target.value)}>{(catalog?.metrics || []).map((metric) => <option value={metric.key} key={metric.key}>{metric.family} · {metric.short_label}</option>)}</select>{rows.length > 2 ? <button className="icon-button" onClick={() => setRows((current) => current.filter((_, rowIndex) => rowIndex !== index))} type="button">×</button> : <i />}</div>)}</div>
      <div className="builder-foot"><button className="secondary-button" disabled={rows.length >= 6} onClick={() => setRows((current) => [...current, { ...(current.at(-1) || DEFAULT[0]) }])} type="button">+ Add series</button><label className="toggle"><input type="checkbox" checked={normalized} onChange={(event) => setNormalized(event.target.checked)} /><span>Standardize each series for mixed-unit trend comparison</span></label></div>
    </section>
    {result ? <><section className="compare-summary"><article><span>Pearson r · first two series</span><strong>{number(result.relationship.pearson_r, 3)}</strong><small>{result.relationship.pair_count} aligned 5-minute windows</small></article><article><span>Interpretation</span><strong>{Math.abs(result.relationship.pearson_r || 0) >= .7 ? 'Strong co-movement' : Math.abs(result.relationship.pearson_r || 0) >= .4 ? 'Moderate co-movement' : 'Weak co-movement'}</strong><small>Correlation is not causation or fault proof</small></article>{result.series.map((item) => <article key={item.id}><span>{item.label}</span><strong>{item.point_count}</strong><small>{item.family} · {item.unit}</small></article>)}</section><section className="panel chart-panel hero-chart"><div className="panel-title"><div><p className="eyebrow">ALIGNED TIME SERIES</p><h2>{normalized ? 'Standardized relationship movement' : 'Native-unit movement'}</h2><span>Use standardized mode only to compare shape; tooltips retain plotted values</span></div></div><EChart option={trend} height={440} label="Selected comparison time series" /></section><section className="panel chart-panel hero-chart"><div className="panel-title"><div><p className="eyebrow">PAIR RELATIONSHIP</p><h2>First two series scatter</h2><span>{result.relationship.warning}</span></div></div>{result.relationship.points.length ? <EChart option={scatter} height={420} label="Comparison scatter plot" /> : <Empty>Need at least three aligned timestamps for correlation.</Empty>}</section></> : <Empty>Choose series and run comparison.</Empty>}
  </Shell>
}
