import { useCallback } from 'react'
import { api } from '../api'
import { Badge, Empty, ErrorBanner, PageHeading, Shell } from '../components/Shell'
import { dateTime, number } from '../format'
import { usePolling } from '../hooks'

const REFRESH = Number(import.meta.env.VITE_REFRESH_MS || 15000)

export default function PipelinePage() {
  const loader = useCallback(async () => ({ pipeline: await api.pipeline(), artifact: await api.artifact() }), [])
  const { data, error, loading, refreshing, updatedAt, refresh } = usePolling(loader, REFRESH)
  const pipeline = data?.pipeline
  const artifact = data?.artifact
  return <Shell active="pipeline" error={error} updatedAt={updatedAt} onRefresh={refresh} refreshing={refreshing}>
    <PageHeading eyebrow="MODEL GOVERNANCE & EQUATIONS" title="What the pipeline is actually calculating" description="หน้าเดียวสำหรับตรวจ version, train evidence, สมการ, threshold, lifecycle และ source contract เพื่อไม่ให้ dashboard แปลคะแนนเกินกว่าสิ่งที่ model รู้จริง" />
    <ErrorBanner error={error} />
    {loading && !data ? <Empty>Loading artifact and policy…</Empty> : null}
    {artifact ? <section className="artifact-hero"><div><p className="eyebrow">IMMUTABLE SHARED ARTIFACT</p><h2>{artifact.model_version}</h2><span>{artifact.model_type}</span></div><div className="artifact-metrics"><div><span>Epochs</span><strong>{artifact.epochs_completed}</strong></div><div><span>Groups</span><strong>{artifact.group_count}</strong></div><div><span>Input</span><strong>{artifact.input_shape.join(' × ')}</strong></div><div><span>Train loss</span><strong>{number(artifact.final_loss, 4)}</strong></div><div><span>Validation loss</span><strong>{number(artifact.final_validation_loss, 4)}</strong></div><div><span>Mean test exceedance</span><strong>{number(artifact.mean_test_exceedance_rate * 100, 2, '%')}</strong></div></div><p>Created {dateTime(artifact.created_at_utc)} · {artifact.threshold_method} · weights mutable: <b>{String(artifact.weights_mutable)}</b></p></section> : null}
    <section className="pipeline-layout"><div className="pipeline-stages">{(pipeline?.stages || []).map((stage, index) => <article className="pipeline-stage" key={stage.key}><div className="stage-index">{String(index + 1).padStart(2, '0')}</div><div><div className="stage-title"><h2>{stage.name}</h2><Badge tone={stage.status === 'ACTIVE_SHADOW' ? 'info' : 'success'}>{stage.status}</Badge></div><code>{stage.formula}</code><p>{stage.detail}</p></div></article>)}</div><aside className="pipeline-aside"><article className="panel"><p className="eyebrow">POLICY THRESHOLDS</p><h2>{pipeline?.policy?.policy_version || '—'}</h2><dl className="policy-list"><div><dt>Window</dt><dd>{pipeline?.policy?.window_seconds}s</dd></div><div><dt>Coverage</dt><dd>{number((pipeline?.policy?.minimum_coverage || 0) * 100, 0, '%')}</dd></div><div><dt>Robust Z entry / exit</dt><dd>{pipeline?.policy?.robust_entry_z} / {pipeline?.policy?.robust_exit_z}</dd></div><div><dt>GMM posterior</dt><dd>≥ {pipeline?.policy?.regime_min_posterior}</dd></div><div><dt>P1 dual evidence</dt><dd>{pipeline?.policy?.p1_dual_seconds / 60} min</dd></div><div><dt>P2 single evidence</dt><dd>{pipeline?.policy?.p2_single_seconds / 60} min</dd></div><div><dt>Bootstrap</dt><dd>{pipeline?.policy?.bootstrap_min_days}–{pipeline?.policy?.bootstrap_recommended_days} days</dd></div></dl></article><article className="panel"><p className="eyebrow">PROFILE LIFECYCLE</p><h2>Automatic creation, controlled activation</h2><ol className="lifecycle-list">{(pipeline?.lifecycle || []).map((item) => <li key={item}>{item.replaceAll('_', ' ')}</li>)}</ol><p className="guardrail">{pipeline?.activation_policy}</p></article><article className="panel"><p className="eyebrow">SOURCE CONTRACT</p><h2>Where every panel comes from</h2><dl className="source-list">{Object.entries(pipeline?.sources || {}).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl></article></aside></section>
    <section className="panel feature-ledger"><div className="panel-title"><div><p className="eyebrow">LSTM INPUT FEATURES</p><h2>60 × {artifact?.feature_columns?.length || 0} sequence tensor</h2><span>Raw, engineered, delta, rolling mean, and rolling standard deviation</span></div></div><div>{(artifact?.feature_columns || []).map((feature) => <code key={feature}>{feature}</code>)}</div></section>
  </Shell>
}
