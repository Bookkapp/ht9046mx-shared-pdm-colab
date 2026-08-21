import { useCallback, useEffect, useRef, useState } from 'react'

export function usePolling(loader, interval = 15000) {
  const mounted = useRef(true)
  const [state, setState] = useState({ data: null, error: null, loading: true, refreshing: false, updatedAt: null })
  const refresh = useCallback(async (background = false) => {
    setState((current) => ({ ...current, loading: !background && !current.data, refreshing: background, error: null }))
    try {
      const data = await loader()
      if (mounted.current) setState({ data, error: null, loading: false, refreshing: false, updatedAt: new Date() })
      return data
    } catch (error) {
      if (mounted.current) setState((current) => ({ ...current, error, loading: false, refreshing: false }))
      return null
    }
  }, [loader])

  useEffect(() => {
    mounted.current = true
    refresh(false)
    const timer = setInterval(() => refresh(true), interval)
    return () => { mounted.current = false; clearInterval(timer) }
  }, [interval, refresh])
  return { ...state, refresh: () => refresh(true) }
}
