import { dateTime } from '../format'

const NAV = [
  { key: 'fleet', label: 'Fleet overview', href: '#/', icon: '▦' },
  { key: 'compare', label: 'Compare signals', href: '#/compare', icon: '⇄' },
  { key: 'pipeline', label: 'Pipeline & model', href: '#/pipeline', icon: '⌁' },
  { key: 'handlers', label: 'Handler setup', href: '#/handlers', icon: '⚙' },
]

export function AdiMark() {
  return (
    <span className="adi-lockup" aria-label="Analog Devices internal dashboard">
      <svg viewBox="0 0 46 38" aria-hidden="true"><path d="M3 3h40v32H3z" fill="currentColor"/><path d="M12 29 23 9l11 20h-7l-4-8-4 8z" fill="white"/></svg>
      <span><strong>ANALOG DEVICES</strong><small>HT9046MX · CONDITION INTELLIGENCE</small></span>
    </span>
  )
}

export function Shell({ active, children, error, updatedAt, onRefresh, refreshing, machine }) {
  const current = NAV.find((item) => item.key === active) || NAV[0]
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand-home" href="#/"><AdiMark /></a>
        <p className="nav-label">MODEL OPERATIONS</p>
        <nav aria-label="Main navigation">{NAV.map((item) => <a className={active === item.key ? 'active' : ''} href={item.href} key={item.key}><span>{item.icon}</span>{item.label}</a>)}</nav>
        <div className="sidebar-bottom">
          <div className={`pipeline-live ${error ? 'bad' : ''}`}><i />{error ? 'Model API unavailable' : 'Data pipeline online'}</div>
          <small>{updatedAt ? `Last refresh · ${dateTime(updatedAt)}` : 'Waiting for first refresh'}</small>
          <a href="#/pipeline">Controlled Hybrid v1 <span>→</span></a>
        </div>
      </aside>
      <main className="main-stage">
        <header className="utility-bar">
          <div className="utility-title"><p>FACTORY MONITOR / HT9046MX</p><strong>{machine || current.label}</strong></div>
          <div className="top-actions">
            <span className={`connection ${error ? 'bad' : 'good'}`}><i />{error ? 'API unavailable' : 'Live model data'}</span>
            {updatedAt ? <small>Updated {dateTime(updatedAt)}</small> : null}
            {onRefresh ? <button className="icon-refresh" aria-label="Refresh model data" onClick={onRefresh} disabled={refreshing} type="button">{refreshing ? '…' : '↻'}</button> : null}
          </div>
        </header>
        <section className="workspace">{children}</section>
        <footer>Model evidence supports human review · review levels are not failure probabilities</footer>
      </main>
    </div>
  )
}

export function PageHeading({ eyebrow, title, description, actions }) {
  return <section className="page-heading"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{description}</p></div>{actions ? <div className="page-actions">{actions}</div> : null}</section>
}

export function Badge({ children, tone = 'neutral' }) {
  return <span className={`badge ${tone}`}>{children}</span>
}

export function Empty({ children }) {
  return <div className="empty-state">{children}</div>
}

export function ErrorBanner({ error }) {
  return error ? <div className="error-banner" role="alert"><strong>Data source unavailable</strong><span>{error.message}</span></div> : null
}
