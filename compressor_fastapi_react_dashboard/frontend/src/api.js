const PREFIX = import.meta.env.VITE_API_PREFIX || '/api/v1'

async function request(path, options = {}) {
  const headers = { Accept: 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...(options.headers || {}) }
  const response = await fetch(`${PREFIX}${path}`, { cache: 'no-store', ...options, headers })
  const payload = await response.json().catch(() => null)
  if (!response.ok) throw new Error(payload?.detail || `${response.status} ${response.statusText}`)
  return payload
}

function adminHeaders(apiKey) {
  return apiKey ? { 'X-API-Key': apiKey } : {}
}

export const api = {
  config: () => request('/config'),
  catalog: () => request('/catalog'),
  fleet: () => request('/model/fleet'),
  artifact: () => request('/model/artifact'),
  pipeline: () => request('/model/pipeline'),
  monitor: (machine, moduleId, selectedDate = '') => request(`/model/machines/${machine}/monitor?module_id=${moduleId}${selectedDate ? `&selected_date=${selectedDate}` : ''}`),
  profiles: (machine) => request(`/model/machines/${machine}/profiles`),
  comparison: (body) => request('/model/comparison', { method: 'POST', body: JSON.stringify(body) }),
  sourceStatus: () => request('/source/status'),
  lifecycle: (machine, action, actor, reason, key) => request(`/model/machines/${machine}/${action}`, { method: 'POST', body: JSON.stringify({ actor, reason: reason || null }), headers: adminHeaders(key) }),
}
