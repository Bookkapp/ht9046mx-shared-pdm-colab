const axis = { axisLine: { lineStyle: { color: '#d0d5dd' } }, axisLabel: { color: '#667085', fontSize: 10 }, splitLine: { lineStyle: { color: '#edf0f3' } } }

export function lineOption(points, definitions, { yName = '', thresholds = [], normalize = false } = {}) {
  const source = definitions.map((definition) => {
    let values = points.map((point) => [point.event_time, point[definition.key]]).filter((item) => item[1] !== null && item[1] !== undefined)
    if (normalize && values.length) {
      const finite = values.map((item) => Number(item[1])).filter(Number.isFinite)
      const center = finite.reduce((sum, value) => sum + value, 0) / Math.max(1, finite.length)
      const spread = Math.sqrt(finite.reduce((sum, value) => sum + (value - center) ** 2, 0) / Math.max(1, finite.length)) || 1
      values = values.map(([time, value]) => [time, (value - center) / spread])
    }
    return { name: definition.label, type: 'line', data: values, showSymbol: false, smooth: false, connectNulls: false, sampling: 'lttb', lineStyle: { width: definition.width || 2, color: definition.color }, itemStyle: { color: definition.color }, emphasis: { focus: 'series' }, ...(definition.area ? { areaStyle: { color: definition.color, opacity: .08 } } : {}) }
  })
  if (source.length && thresholds.length) {
    source[0].markLine = { silent: true, symbol: 'none', label: { color: '#667085', fontSize: 9, formatter: ({ name }) => name }, lineStyle: { type: 'dashed', width: 1 }, data: thresholds.map((item) => ({ name: item.label, yAxis: item.value, lineStyle: { color: item.color || '#d97706' } })) }
  }
  return {
    animation: false,
    color: definitions.map((item) => item.color),
    tooltip: { trigger: 'axis', valueFormatter: (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—' },
    legend: { top: 0, right: 0, textStyle: { color: '#475467', fontSize: 10 } },
    grid: { left: 48, right: 18, top: 38, bottom: 42 },
    xAxis: { type: 'time', ...axis, splitLine: { show: false } },
    yAxis: { type: 'value', name: normalize ? 'standardized' : yName, nameTextStyle: { color: '#98a2b3', fontSize: 9 }, ...axis },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 15, bottom: 8, borderColor: 'transparent', backgroundColor: '#f2f4f7', fillerColor: 'rgba(200,16,46,.08)' }],
    series: source,
  }
}

export function scatterOption(points, leftLabel, rightLabel, color = '#c8102e') {
  return {
    animation: false,
    tooltip: { trigger: 'item', formatter: ({ value }) => `${leftLabel}: ${Number(value[0]).toFixed(3)}<br/>${rightLabel}: ${Number(value[1]).toFixed(3)}` },
    grid: { left: 54, right: 18, top: 20, bottom: 45 },
    xAxis: { type: 'value', name: leftLabel, nameLocation: 'middle', nameGap: 28, ...axis },
    yAxis: { type: 'value', name: rightLabel, nameLocation: 'middle', nameGap: 38, ...axis },
    series: [{ type: 'scatter', data: points.map((item) => [item.x, item.y]), symbolSize: 7, itemStyle: { color, opacity: .62 } }],
  }
}

export function correlationHeatmapOption(points, features) {
  const correlation = (left, right) => {
    const pairs = points.map((point) => [Number(point[left.key]), Number(point[right.key])]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    if (pairs.length < 3) return null
    const xs = pairs.map((item) => item[0]); const ys = pairs.map((item) => item[1])
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length; const my = ys.reduce((a, b) => a + b, 0) / ys.length
    const numerator = pairs.reduce((sum, [x, y]) => sum + (x - mx) * (y - my), 0)
    const denominator = Math.sqrt(xs.reduce((sum, x) => sum + (x - mx) ** 2, 0) * ys.reduce((sum, y) => sum + (y - my) ** 2, 0))
    return denominator ? numerator / denominator : null
  }
  const data = []
  features.forEach((left, x) => features.forEach((right, y) => {
    const value = correlation(left, right)
    if (value !== null) data.push([x, y, Number(value.toFixed(3))])
  }))
  const labels = features.map((item) => item.label)
  return {
    animation: false,
    tooltip: { formatter: ({ value }) => `${labels[value[0]]} ↔ ${labels[value[1]]}<br/><b>r = ${Number(value[2]).toFixed(3)}</b>` },
    grid: { left: 92, right: 54, top: 18, bottom: 84 },
    xAxis: { type: 'category', data: labels, axisLabel: { rotate: 42, color: '#667085', fontSize: 9 }, splitArea: { show: true } },
    yAxis: { type: 'category', data: labels, axisLabel: { color: '#667085', fontSize: 9 }, splitArea: { show: true } },
    visualMap: { min: -1, max: 1, calculable: false, orient: 'vertical', right: 0, top: 'center', inRange: { color: ['#0067b1', '#f7f8fa', '#c8102e'] }, textStyle: { color: '#667085', fontSize: 9 } },
    series: [{ type: 'heatmap', data, label: { show: true, fontSize: 8, color: '#344054', formatter: ({ value }) => Number(value[2]).toFixed(2) }, emphasis: { itemStyle: { borderColor: '#101820', borderWidth: 1 } } }],
  }
}
