import { useCallback, useMemo, useState } from 'react'
import { api } from '../api'
import { correlationHeatmapOption, lineOption } from '../charts'
import { EChart } from '../components/EChart'
import { Badge, Empty, ErrorBanner, PageHeading, Shell } from '../components/Shell'
import { dateTime, lifecycleLabel, number, tone } from '../format'
import { usePolling } from '../hooks'

const REFRESH = Number(import.meta.env.VITE_REFRESH_MS || 15000)
const MODULES = [1, 2, 3, 4, 5, 6, 8]

function PanelTitle({ eyebrow, title, detail, action }) {
  return <div className="panel-title"><div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2><span>{detail}</span></div>{action}</div>
}

function ChartPanel({ eyebrow, title, detail, option, height = 300, note, empty }) {
  return <article className="panel chart-panel"><PanelTitle eyebrow={eyebrow} title={title} detail={detail} />{empty ? <Empty>{empty}</Empty> : <EChart option={option} height={height} label={title} />}{note ? <p className="chart-note">{note}</p> : null}</article>
}

function DecisionCard({ latest }) {
  if (!latest) return <article className="panel decision-card"><p className="eyebrow">LATEST DECISION</p><h2>No model decision yet</h2><p>ระบบยังไม่มี Candidate/Active Profile หรือยังไม่มี scoring cycle ที่สมบูรณ์ จึงไม่สรุปว่า Normal</p></article>
  return <article className="panel decision-card"><div className="decision-head"><div><p className="eyebrow">LATEST DECISION</p><h2>{latest.review_level}</h2></div><Badge tone={tone(latest.review_level)}>{latest.window_status}</Badge></div><p>{latest.reason_codes?.length ? latest.reason_codes.join(' · ') : 'No anomaly reason code in this eligible window.'}</p><dl><div><dt>Event time</dt><dd>{dateTime(latest.event_time)}</dd></div><div><dt>Context</dt><dd>{latest.operating_mode || '—'} / {latest.regime || '—'}</dd></div><div><dt>Profile</dt><dd>{latest.profile_version || 'not active'}</dd></div><div><dt>Coverage</dt><dd>{number((latest.coverage || 0) * 100, 1, '%')}</dd></div></dl></article>
}

export default function MachinePage({ machine }) {
  const [moduleId, setModuleId] = useState(1)
  const [selectedDate, setSelectedDate] = useState('')
  const loader = useCallback(() => api.monitor(machine, moduleId, selectedDate), [machine, moduleId, selectedDate])
  const { data, error, loading, refreshing, updatedAt, refresh } = usePolling(loader, REFRESH)
  const [apiKey, setApiKey] = useState('')
  const [actor, setActor] = useState('')
  const [reason, setReason] = useState('')
  const [actionState, setActionState] = useState({ busy: false, message: '' })
  const signals = data?.signals || []
  const model = data?.model_points || []
  const latest = data?.latest_decision
  const lifecycle = data?.lifecycle || {}
  const profile = data?.profile || {}
  const context = data?.selected_context || profile.contexts?.[0]
  const group = data?.shared_lstm_group || {}
  const latestSignal = signals.at(-1)
  const qualitySummary = useMemo(() => signals.reduce((summary, item) => {
    summary[item.window_status] = (summary[item.window_status] || 0) + 1
    return summary
  }, {}), [signals])
  const qualitySignals = useMemo(() => signals.map((item) => ({ ...item, coverage_percent: item.coverage == null ? null : item.coverage * 100 })), [signals])

  const com1 = useMemo(() => lineOption(signals, [{ key: 'hp1', label: 'HP 1st', color: '#c8102e' }, { key: 'lp1', label: 'LP 1st', color: '#7f1d2d' }], { yName: 'source pressure' }), [signals])
  const com2 = useMemo(() => lineOption(signals, [{ key: 'hp2', label: 'HP 2nd', color: '#0067b1' }, { key: 'lp2', label: 'LP 2nd', color: '#003b5c' }], { yName: 'source pressure' }), [signals])
  const thermal = useMemo(() => lineOption(signals, [{ key: 'temphi', label: 'Temp high', color: '#c8102e' }, { key: 'templo', label: 'Temp low', color: '#0e7490' }, { key: 'temperature_span', label: 'Temp span', color: '#00856a' }], { yName: '°C' }), [signals])
  const control = useMemo(() => lineOption(signals, [{ key: 'valve', label: 'Valve', color: '#d97706' }, { key: 'pressure_ratio', label: 'Pressure ratio', color: '#475467' }], { normalize: true }), [signals])
  const quality = useMemo(() => lineOption(qualitySignals, [{ key: 'coverage_percent', label: 'Coverage', color: '#00856a' }], { yName: '% observed', thresholds: [{ label: 'minimum coverage', value: (data?.quality_gate?.minimum_coverage || .9) * 100, color: '#d97706' }] }), [qualitySignals, data?.quality_gate?.minimum_coverage])
  const robust = useMemo(() => lineOption(model, [{ key: 'z_hp2', label: 'Z HP2', color: '#c8102e' }, { key: 'z_lp2_residual', label: 'Z LP2 residual', color: '#0067b1' }, { key: 'z_pressure_gap', label: 'Z pressure gap', color: '#6941c6' }, { key: 'z_temperature_span', label: 'Z temp span', color: '#00856a' }], { yName: 'robust z', thresholds: [{ label: 'entry +3.5', value: 3.5 }, { label: 'exit +2.5', value: 2.5, color: '#98a2b3' }, { label: 'exit −2.5', value: -2.5, color: '#98a2b3' }, { label: 'entry −3.5', value: -3.5 }] }), [model])
  const mergedLp2 = useMemo(() => {
    const byTime = new Map(signals.map((item) => [item.event_time, { ...item }]))
    model.forEach((item) => byTime.set(item.event_time, { ...(byTime.get(item.event_time) || { event_time: item.event_time }), ...item }))
    return [...byTime.values()].sort((a, b) => a.event_time.localeCompare(b.event_time))
  }, [signals, model])
  const residual = useMemo(() => lineOption(mergedLp2, [{ key: 'lp2', label: 'Actual LP2', color: '#003b5c' }, { key: 'predicted_lp2', label: 'Ridge expected LP2', color: '#d97706', width: 1.5 }], { yName: 'source pressure' }), [mergedLp2])
  const isolation = useMemo(() => lineOption(model, [{ key: 'isolation_score', label: 'Isolation score', color: '#6941c6' }], { yName: 'score', thresholds: latest?.isolation_entry_threshold != null ? [{ label: 'entry Q99', value: latest.isolation_entry_threshold }, { label: 'exit Q95', value: latest.isolation_exit_threshold, color: '#98a2b3' }] : [] }), [model, latest])
  const lstm = useMemo(() => lineOption(model, [{ key: 'lstm_score', label: 'Reconstruction Q95', color: '#101820' }, { key: 'lstm_threshold', label: 'Group threshold', color: '#c8102e' }], { yName: 'reconstruction MAE' }), [model])
  const relationshipFeatures = useMemo(() => [
    { key: 'hp1', label: 'HP1' }, { key: 'lp1', label: 'LP1' }, { key: 'hp2', label: 'HP2' }, { key: 'lp2', label: 'LP2' },
    { key: 'valve', label: 'Valve' }, { key: 'temphi', label: 'TempHi' }, { key: 'templo', label: 'TempLo' },
    { key: 'pressure_gap', label: 'Gap' }, { key: 'pressure_ratio', label: 'Ratio' }, { key: 'temperature_span', label: 'Temp span' },
  ], [])
  const heatmap = useMemo(() => correlationHeatmapOption(signals, relationshipFeatures), [signals, relationshipFeatures])

  const lifecycleAction = async (action) => {
    if (!actor.trim()) { setActionState({ busy: false, message: 'กรุณาระบุชื่อผู้ดำเนินการ' }); return }
    setActionState({ busy: true, message: '' })
    try {
      await api.lifecycle(machine, action, actor.trim(), reason.trim(), apiKey)
      setActionState({ busy: false, message: `${action} completed` })
      refresh()
    } catch (actionError) { setActionState({ busy: false, message: actionError.message }) }
  }

  return <Shell active="fleet" machine={`${machine} · M${moduleId}`} error={error} updatedAt={updatedAt} onRefresh={refresh} refreshing={refreshing}>
    <PageHeading eyebrow="MACHINE MODEL MONITOR" title={machine} description="ตรวจสัญญาณจริงเทียบกับ context baseline, COM2 evidence, GMM regime และ Shared LSTM shadow โดยแยกสิ่งที่ model รู้ สิ่งที่ไม่รู้ และเหตุผลที่ยกระดับ review" actions={<><a className="secondary-button" href="#/">← Fleet</a><label className="date-field"><span>Source date</span><input type="date" value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)} /></label></>} />
    <ErrorBanner error={error} />
    <section className="module-tabs" aria-label="Module selector">{MODULES.map((value) => <button className={moduleId === value ? 'active' : ''} key={value} onClick={() => setModuleId(value)} type="button"><span>M{value}</span><small>{value <= 4 ? 'Index' : value <= 6 ? 'Shuttle' : 'Hot2'}</small></button>)}</section>
    <section className="model-status-grid">
      <DecisionCard latest={latest} />
      <article className="panel lifecycle-card"><div className="decision-head"><div><p className="eyebrow">PROFILE LIFECYCLE</p><h2>{lifecycleLabel(lifecycle.state)}</h2></div><Badge tone={tone(lifecycle.state)}>{profile.status || 'NO PROFILE'}</Badge></div><p>{profile.available ? `${profile.contexts?.length || 0} frozen contexts · ${profile.source_windows || 0} selected windows` : 'ยังไม่มี baseline ของ module นี้ ระบบจะไม่ใช้ค่า default ของเครื่องอื่นมาตัดสินว่า Normal'}</p><div className="lifecycle-track">{['COLLECTING_DATA', 'LEARNING', 'CANDIDATE_PROFILE_READY', 'SHADOW_VALIDATION', 'APPROVAL_REQUIRED', 'ACTIVE'].map((item) => <span className={item === lifecycle.state ? 'current' : ''} key={item}>{item.replaceAll('_', ' ')}</span>)}</div></article>
      <article className="panel lstm-card"><p className="eyebrow">SHARED LSTM GROUP</p><h2>{group.group_name || `${machine}__M${String(moduleId).padStart(2, '0')}`}</h2><div className="big-metric">{group.configured ? 'TRAINED GROUP' : 'LOCAL CALIBRATION REQUIRED'}</div><dl><div><dt>Validation P99 threshold</dt><dd>{number(group.threshold?.value, 4)}</dd></div><div><dt>Held-out P95</dt><dd>{number(group.held_out_metrics?.test_mae_p95, 4)}</dd></div><div><dt>Test exceedance</dt><dd>{group.held_out_metrics?.test_exceedance_rate != null ? number(group.held_out_metrics.test_exceedance_rate * 100, 2, '%') : '—'}</dd></div></dl></article>
    </section>

    <div className="section-row"><div><p className="eyebrow">RAW CONTEXT</p><h2>กราฟเครื่องและ compressor stages</h2><span>5-minute event-time median · quality/status ยังคงแยกเก็บ ไม่ถือว่า missing คือ Normal</span></div><Badge tone={data?.source?.available ? 'success' : 'warning'}>{data?.source?.file_name || 'NO SOURCE FILE'}</Badge></div>
    {loading && !data ? <Empty>Reading synchronized log and model profile…</Empty> : null}
    <section className="compare-summary quality-summary"><article><span>Eligible windows</span><strong>{qualitySummary.ELIGIBLE || 0}</strong><small>ผ่าน state, transition, range, gap และ coverage gates</small></article><article><span>Off / transition</span><strong>{qualitySummary.OFF_OR_TRANSITION || 0}</strong><small>ไม่นำไปเรียน baseline หรือสรุป Normal</small></article><article><span>Quality / incomplete</span><strong>{(qualitySummary.DATA_QUALITY_REVIEW || 0) + (qualitySummary.INCOMPLETE_WINDOW || 0)}</strong><small>ตรวจ sentinel, range, duplicate, points และ time gap</small></article><article><span>Latest context</span><strong>{latestSignal?.sv || '—'}</strong><small>{latestSignal?.global_status || '—'} / {latestSignal?.module_status || '—'} · Busy {number(latestSignal?.busy, 0)}</small></article></section>
    <section className="chart-grid two">
      <ChartPanel eyebrow="COM1 · FIRST STAGE" title="HP 1st / LP 1st" detail="Context signals retained for baseline and diagnosis" option={com1} empty={!signals.length ? data?.source?.message : null} />
      <ChartPanel eyebrow="COM2 · PRIMARY DETECTOR" title="HP 2nd / LP 2nd" detail="COM2 pressure relationship used by Ridge, residual, and robust evidence" option={com2} empty={!signals.length ? data?.source?.message : null} />
      <ChartPanel eyebrow="THERMAL RESPONSE" title="Temp high / Temp low / span" detail="Temperature-span robust evidence uses the selected context" option={thermal} empty={!signals.length ? data?.source?.message : null} />
      <ChartPanel eyebrow="CONTROL CONTEXT" title="Valve and pressure ratio" detail="Indexed chart makes mixed units comparable; values are not replaced in the model" option={control} empty={!signals.length ? data?.source?.message : null} note="Operating mode is deterministic from SV + Valve bucket. GMM then resolves a regime inside that mode." />
    </section>
    <section className="panel chart-panel hero-chart"><PanelTitle eyebrow="DATA QUALITY GATE" title="Window coverage before any model scoring" detail={`Required ≥ ${number((data?.quality_gate?.minimum_coverage || .9) * 100, 0, '%')} · minimum ${data?.quality_gate?.minimum_window_points || 30} points · maximum gap ${data?.quality_gate?.max_gap_seconds || 15}s`} />{signals.length ? <EChart option={quality} height={280} label="Five-minute data coverage" /> : <Empty>{data?.source?.message || 'No quality windows available.'}</Empty>}<p className="chart-note">Status, Busy และ SV เก็บเป็น context แบบ categorical; graph นี้แสดง coverage เชิงตัวเลข ส่วนเหตุผล rejection อยู่ในแต่ละ window โดยไม่ถูกส่งต่อไปให้ COM2 หรือ LSTM ตัดสิน</p></section>
    <section className="panel chart-panel hero-chart"><PanelTitle eyebrow="FULL FEATURE RELATIONSHIP" title="Correlation matrix of the selected machine/module" detail="Pearson r over aligned five-minute windows for raw and engineered context features" />{signals.length ? <EChart option={heatmap} height={500} label="Full feature correlation matrix" /> : <Empty>{data?.source?.message || 'No signal windows available.'}</Empty>}<p className="chart-note">Correlation helpsตรวจความสมเหตุผลของ pressure/control/temperature relationships แต่ไม่พิสูจน์สาเหตุของ fault และไม่ได้ใช้แทน GMM posterior หรือ residual evidence</p></section>

    <div className="section-row"><div><p className="eyebrow">EXPLAINABLE COM2</p><h2>หลักฐานแต่ละตัวก่อน fusion</h2><span>คะแนนไม่ถูกบวกเป็น probability เดียว เพื่อรักษาความหมายของแต่ละ detector</span></div><Badge tone={latest?.com2_active ? 'warning' : 'neutral'}>{latest?.com2_active ? 'COM2 EVIDENCE ACTIVE' : 'NO ACTIVE COM2 EVIDENCE'}</Badge></div>
    <section className="chart-grid two">
      <ChartPanel eyebrow="ROBUST Z / MAD" title="Deviation from frozen context" detail="z = (x − median) / (1.4826 × MAD)" option={robust} empty={!model.length ? 'ยังไม่มี model decisions สำหรับช่วงนี้' : null} note="Entry ±3.5; exit ±2.5. LP2 residual เป็น directional trigger เมื่อ z ≤ −3.5" />
      <ChartPanel eyebrow="RIDGE CONDITIONAL EXPECTATION" title="Actual LP2 vs expected LP2" detail="Expected from HP2, Valve, TempHi, TempLo with Ridge α=1" option={residual} empty={!model.length ? 'Ridge prediction จะปรากฏหลังมี Candidate/Active Profile' : null} note={`Latest residual ${number(latest?.lp2_residual, 3)} · robust z ${number(latest?.z_lp2_residual, 2)}`} />
      <ChartPanel eyebrow="ISOLATION FOREST" title="Multivariate pressure-pattern score" detail="200 trees on four robust evidence dimensions" option={isolation} empty={!model.length ? 'ยังไม่มี Isolation Forest score' : null} note="Threshold Q99/Q95 ถูก calibrate ต่อ machine/module/mode/regime และไม่ใช่ failure probability" />
      <ChartPanel eyebrow="IMMUTABLE SHARED LSTM" title="Reconstruction score vs group threshold" detail="Bucket Q95 error; weights remain frozen" option={lstm} empty={!model.length ? 'ยังไม่มี Shared LSTM shadow score' : null} note={`Latest top error feature: ${latest?.lstm_top_error_feature || '—'} · calibration ${latest?.lstm_calibration_source || '—'}`} />
    </section>

    <section className="profile-layout">
      <article className="panel profile-detail"><PanelTitle eyebrow="FROZEN PROFILE" title={context ? `${context.operating_mode} / ${context.regime}` : 'No context selected'} detail={context ? `${context.training_windows} healthy-selected windows · ${profile.version}` : 'Profile values will appear after automatic bootstrap'} />
        {context ? <><div className="baseline-grid">{['hp2', 'lp2', 'pressure_gap', 'temperature_span'].map((key) => <div key={key}><span>{key}</span><strong>{number(context.feature_center?.[key], 3)}</strong><small>MAD scale {number(context.feature_scale?.[key], 3)} · entry band ±{number((context.feature_scale?.[key] || 0) * 3.5, 3)}</small></div>)}</div><div className="equation-box"><code>LP2_hat = {number(context.ridge?.intercept, 4)} + Σ βⱼxⱼ</code><p>{context.ridge?.features?.map((feature, index) => `${feature} × ${number(context.ridge.coefficients[index], 4)}`).join(' + ')}</p></div><div className="context-table"><div><span>Residual center</span><strong>{number(context.ridge?.residual_center, 4)}</strong></div><div><span>Residual MAD scale</span><strong>{number(context.ridge?.residual_scale, 4)}</strong></div><div><span>IF entry Q99</span><strong>{number(context.isolation_forest?.entry_threshold, 4)}</strong></div><div><span>IF exit Q95</span><strong>{number(context.isolation_forest?.exit_threshold, 4)}</strong></div></div></> : <Empty>Automatic bootstrap requires at least 7 eligible days and 200 windows; 14 days is recommended.</Empty>}
      </article>
      <article className="panel approval-panel"><PanelTitle eyebrow="CONTROLLED ACTIVATION" title="Human decision gate" detail="Candidate never overwrites Active Frozen Profile by itself" />
        <label className="field"><span>Engineer / approver</span><input value={actor} onChange={(event) => setActor(event.target.value)} placeholder="Name or employee ID" /></label><label className="field"><span>Reason / review note</span><textarea value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why approve, reject, or continue learning" /></label><details><summary>API key</summary><label className="field"><span>X-API-Key</span><input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} /></label></details><div className="approval-actions"><button className="primary-button" disabled={actionState.busy || lifecycle.state !== 'APPROVAL_REQUIRED'} onClick={() => lifecycleAction('approve')} type="button">Approve frozen profile</button><button className="danger-button" disabled={actionState.busy || lifecycle.state !== 'APPROVAL_REQUIRED'} onClick={() => lifecycleAction('reject')} type="button">Reject</button><button className="secondary-button" disabled={actionState.busy} onClick={() => lifecycleAction('continue-learning')} type="button">Continue learning</button></div>{actionState.message ? <p className="action-message">{actionState.message}</p> : null}<p className="guardrail">Approve/Reject เปลี่ยน lifecycle และเขียน audit history เท่านั้น ไม่สั่งหยุดเครื่อง ไม่แก้ LSTM weights และไม่ลบ profile รุ่นก่อน</p></article>
    </section>
  </Shell>
}
