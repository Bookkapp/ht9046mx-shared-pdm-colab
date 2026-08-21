import { useCallback, useMemo, useState } from 'react'
import { api } from '../api'
import { dateTime, lifecycleLabel, tone } from '../format'
import { usePolling } from '../hooks'
import { Badge, Empty, ErrorBanner, PageHeading, Shell } from '../components/Shell'

const REFRESH = Number(import.meta.env.VITE_REFRESH_MS || 15000)

function Stat({ label, value, detail, accent = '' }) {
  return <article className={`stat-card ${accent}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>
}

export default function FleetPage() {
  const loader = useCallback(() => api.fleet(), [])
  const { data, error, loading, refreshing, updatedAt, refresh } = usePolling(loader, REFRESH)
  const [query, setQuery] = useState('')
  const [state, setState] = useState('ALL')
  const machines = useMemo(() => (data?.machines || []).filter((machine) => {
    const matchesText = machine.machine_id.includes(query.trim().toUpperCase()) || machine.ip.includes(query.trim())
    return matchesText && (state === 'ALL' || machine.lifecycle_state === state)
  }), [data, query, state])
  const summary = data?.summary || {}
  const artifact = data?.artifact || {}
  const states = [...new Set((data?.machines || []).map((item) => item.lifecycle_state))]

  return <Shell active="fleet" error={error} updatedAt={updatedAt} onRefresh={refresh} refreshing={refreshing}>
    <PageHeading eyebrow="CONTROLLED HYBRID FLEET" title="Model state before machine state" description="ติดตามว่าแต่ละเครื่องมีข้อมูลพอหรือยัง, profile อยู่ lifecycle ใด, COM2/LSTM เห็นหลักฐานอะไร และมีจุดใดที่ระบบ abstain แทนการเรียกว่า Normal" actions={<><a className="secondary-button" href="#/pipeline">View equations</a><a className="primary-button" href="#/handlers">Add machine</a></>} />
    <ErrorBanner error={error} />
    <section className="source-banner"><div><i className={artifact.available ? 'good' : 'bad'} /><span><strong>{artifact.model_version || 'Shared model unavailable'}</strong> · {artifact.group_count || 0} groups · {artifact.epochs_completed || 0} epochs · input {(artifact.input_shape || []).join(' × ')}</span></div><small>Weights immutable · per-group threshold · {artifact.role || 'shadow evidence'}</small></section>
    <section className="stats-grid">
      <Stat label="Configured handlers" value={summary.configured_handlers ?? '—'} detail={`${summary.data_sources_available || 0} synchronized sources available`} />
      <Stat label="Active frozen profiles" value={summary.active_frozen ?? '—'} detail="Human-approved production baselines" accent="green" />
      <Stat label="Approval required" value={summary.approval_required ?? '—'} detail={`${summary.shadow_validation || 0} machines still in shadow`} accent="amber" />
      <Stat label="Review records in loaded tail" value={(summary.p1_review_records || 0) + (summary.p2_review_records || 0)} detail={`P1 ${summary.p1_review_records || 0} · P2 ${summary.p2_review_records || 0}`} accent="red" />
      <Stat label="Shared LSTM validation" value={artifact.final_validation_loss != null ? Number(artifact.final_validation_loss).toFixed(4) : '—'} detail={`Mean held-out exceedance ${artifact.mean_test_exceedance_rate != null ? `${(artifact.mean_test_exceedance_rate * 100).toFixed(2)}%` : '—'}`} />
    </section>

    <section className="directory-panel">
      <div className="directory-heading"><div><p className="eyebrow">MACHINE DIRECTORY</p><h2>Onboarding and evidence state</h2><span>{machines.length} visible machines</span></div><div className="filters"><input aria-label="Search machine or IP" placeholder="Search MX057 or IP" value={query} onChange={(event) => setQuery(event.target.value)} /><select aria-label="Lifecycle filter" value={state} onChange={(event) => setState(event.target.value)}><option value="ALL">All lifecycle states</option>{states.map((item) => <option key={item} value={item}>{lifecycleLabel(item)}</option>)}</select></div></div>
      {loading && !data ? <Empty>Loading fleet state…</Empty> : null}
      <div className="machine-grid">{machines.map((machine) => {
        const latest = machine.latest_decision
        return <a className="machine-card" href={`#/machines/${machine.machine_id}`} key={machine.machine_id}>
          <div className="machine-card-head"><div><span className={`source-light ${machine.data_source_available ? 'online' : ''}`} /><strong>{machine.machine_id}</strong><small>{machine.ip}</small></div><Badge tone={tone(machine.lifecycle_state)}>{lifecycleLabel(machine.lifecycle_state)}</Badge></div>
          <dl><div><dt>Profile modules</dt><dd>{machine.profiled_modules} / 7</dd></div><div><dt>LSTM trained groups</dt><dd>{machine.shared_lstm_groups} / 7</dd></div><div><dt>Latest review</dt><dd><Badge tone={tone(latest?.review_level)}>{latest?.review_level || 'NO SCORE'}</Badge></dd></div></dl>
          <div className="machine-card-foot"><span>{machine.latest_source_file || 'Waiting for synchronized log'}</span><small>{dateTime(latest?.event_time || machine.latest_source_modified_at)}</small></div>
        </a>
      })}</div>
      {!loading && !machines.length ? <Empty>No machines match the current filter.</Empty> : null}
    </section>
  </Shell>
}
