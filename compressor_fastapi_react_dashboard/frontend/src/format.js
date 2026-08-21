export function number(value, digits = 2, suffix = '') {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? `${numeric.toFixed(digits)}${suffix}` : '—'
}

export function dateTime(value) {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.valueOf())) return String(value)
  return new Intl.DateTimeFormat('th-TH', { dateStyle: 'medium', timeStyle: 'short' }).format(parsed)
}

export function lifecycleLabel(value) {
  return String(value || 'COLLECTING_DATA').replaceAll('_', ' ')
}

export function tone(value) {
  if (value === 'P1_REVIEW' || value === 'REJECTED') return 'danger'
  if (value === 'P2_REVIEW' || value === 'APPROVAL_REQUIRED') return 'warning'
  if (value === 'SHADOW' || value === 'SHADOW_VALIDATION') return 'info'
  if (value === 'NORMAL' || value === 'ACTIVE') return 'success'
  return 'neutral'
}
