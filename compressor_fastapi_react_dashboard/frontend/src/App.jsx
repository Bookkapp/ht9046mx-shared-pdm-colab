import { lazy, Suspense, useEffect, useState } from 'react'

const FleetPage = lazy(() => import('./pages/FleetPage'))
const MachinePage = lazy(() => import('./pages/MachinePage'))
const ComparePage = lazy(() => import('./pages/ComparePage'))
const PipelinePage = lazy(() => import('./pages/PipelinePage'))
const HandlersPage = lazy(() => import('./pages/HandlersPage'))

function route() {
  const hash = window.location.hash || '#/'
  const machine = hash.match(/^#\/machines\/(MX[0-9]{1,20})$/i)
  if (machine) return { page: 'machine', machine: machine[1].toUpperCase() }
  if (hash === '#/compare') return { page: 'compare' }
  if (hash === '#/pipeline') return { page: 'pipeline' }
  if (hash === '#/handlers') return { page: 'handlers' }
  return { page: 'fleet' }
}

export default function App() {
  const [current, setCurrent] = useState(route)
  useEffect(() => {
    const update = () => setCurrent(route())
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  return <Suspense fallback={<div className="route-loading">Loading model evidence…</div>}>
    {current.page === 'machine' ? <MachinePage machine={current.machine} /> : null}
    {current.page === 'compare' ? <ComparePage /> : null}
    {current.page === 'pipeline' ? <PipelinePage /> : null}
    {current.page === 'handlers' ? <HandlersPage /> : null}
    {current.page === 'fleet' ? <FleetPage /> : null}
  </Suspense>
}
