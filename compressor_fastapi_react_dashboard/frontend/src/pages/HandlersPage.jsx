import { useCallback, useState } from 'react'
import { api } from '../api'
import { Badge, Empty, ErrorBanner, PageHeading, Shell } from '../components/Shell'
import { dateTime } from '../format'
import { usePolling } from '../hooks'

const REFRESH = Number(import.meta.env.VITE_REFRESH_MS || 15000)

function previewCode(value) {
  const digits = value.trim().toUpperCase().replace(/^MX/, '')
  return digits ? `MX${digits.padStart(3, '0')}` : ''
}

export default function HandlersPage() {
  const loader = useCallback(async () => {
    const [handlers, sync, config] = await Promise.all([api.handlers(), api.syncStatus(), api.config()])
    return { handlers, sync, config }
  }, [])
  const { data, error, refreshing, updatedAt, refresh } = usePolling(loader, REFRESH)
  const [machine, setMachine] = useState('')
  const [ip, setIp] = useState('')
  const [key, setKey] = useState('')
  const [draft, setDraft] = useState({})
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState('')
  const code = previewCode(machine)
  const destinationRoot = data?.config?.handler_destination_root || 'C:\\HT9046MX\\data\\incoming'
  const sync = data?.sync
  const latestSync = sync?.latest
  const syncReady = latestSync?.status === 'ready'

  const mutate = async (name, operation) => {
    setBusy(name); setNotice('')
    try { await operation(); setNotice(`${name}: saved to the persistent handler registry. The next SMB sync and model cycle will discover it automatically.`); await refresh() } catch (mutationError) { setNotice(mutationError.message) } finally { setBusy('') }
  }
  const submit = async (event) => {
    event.preventDefault()
    await mutate(machine || 'New machine', () => api.createHandler({ machine_code: machine, ip }, key))
    setMachine(''); setIp('')
  }
  return <Shell active="handlers" error={error} updatedAt={updatedAt} onRefresh={refresh} refreshing={refreshing}>
    <PageHeading eyebrow="HANDLER & MODEL ONBOARDING" title="Add a machine with code and IP" description="Dashboard เขียน machine registry ถาวรเพียงจุดเดียว จากนั้น Scheduled SMB Sync จะคัดลอก log เข้า local storage และ Controlled Hybrid runner จะพบเครื่องใหม่เอง โดยไม่ต้องตั้ง threshold, MAD, GMM หรือ LSTM ด้วยมือ" />
    <ErrorBanner error={error} />
    {notice ? <div className="notice-banner">{notice}</div> : null}
    <section className="source-banner"><div><i className={latestSync && syncReady ? 'good' : 'bad'} /><strong>SMB sync {latestSync ? (syncReady ? 'ready' : 'needs review') : 'has not run yet'}</strong></div><small>{latestSync ? `${latestSync.handlers?.length || 0} configured source(s) · last completed ${dateTime(latestSync.completed_at_utc)}` : 'Install HT9046MX-SMB-Sync, then run one cycle to validate the shares.'}</small><Badge tone={syncReady ? 'success' : 'warning'}>{latestSync?.status || 'NOT STARTED'}</Badge></section>
    <section className="handler-layout"><form className="panel handler-form" onSubmit={submit}><p className="eyebrow">NEW HANDLER</p><h2>Machine identity</h2><label className="field"><span>Machine code</span><input required pattern="(?:MX)?[0-9]{1,20}" placeholder="MX012" value={machine} onChange={(event) => setMachine(event.target.value.toUpperCase())} /></label><label className="field"><span>IPv4 address</span><input required inputMode="decimal" placeholder="10.196.132.182" value={ip} onChange={(event) => setIp(event.target.value)} /></label><div className="derived-path"><span>Generated share</span><code>{ip && code ? `\\\\${ip}\\Comp_log_data_${code}` : '\\\\IP\\Comp_log_data_MX012'}</code><span>Destination</span><code>{code ? `${destinationRoot}\\Comp_log_data_${code}` : `${destinationRoot}\\Comp_log_data_MX012`}</code></div><details><summary>API key</summary><label className="field"><span>X-API-Key</span><input type="password" value={key} onChange={(event) => setKey(event.target.value)} /></label></details><button className="primary-button full" disabled={Boolean(busy)} type="submit">{busy ? 'Saving…' : 'Add handler'}</button></form>
      <section className="panel handler-registry"><div className="panel-title"><div><p className="eyebrow">REGISTERED CONFIGURATION</p><h2>{data?.handlers?.length || 0} handlers</h2><span>Atomic handlers.json writes + backup · one registry serves Dashboard, SMB sync, and model runner</span></div></div><div className="handler-list">{(data?.handlers || []).map((item) => { const nextIp = draft[item.name] ?? item.ip; return <article className="handler-row" key={item.name}><div className="handler-name"><i className={item.enabled ? 'enabled' : ''} /><div><strong>{item.name}</strong><small>{item.enabled ? 'ENABLED' : 'DISABLED'}</small></div></div><label><span>IPv4</span><input value={nextIp} onChange={(event) => setDraft((current) => ({ ...current, [item.name]: event.target.value }))} /></label><div className="handler-path"><code>{item.share_path}</code><small>{item.destination}</small></div><div className="row-actions">{nextIp !== item.ip ? <button className="secondary-button" disabled={Boolean(busy)} onClick={() => mutate(item.name, () => api.updateHandler(item.name, { ip: nextIp }, key))} type="button">Save IP</button> : null}<button className="secondary-button" disabled={Boolean(busy)} onClick={() => mutate(item.name, () => api.updateHandler(item.name, { enabled: !item.enabled }, key))} type="button">{item.enabled ? 'Disable' : 'Enable'}</button><button className="danger-button" disabled={Boolean(busy)} onClick={() => { if (window.confirm(`Remove ${item.name} from future sync/model cycles? Data and profiles remain.`)) mutate(item.name, () => api.deleteHandler(item.name, key)) }} type="button">Remove</button></div></article>})}{!data?.handlers?.length ? <Empty>No handler configuration.</Empty> : null}</div></section></section>
  </Shell>
}
