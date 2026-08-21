import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { HeatmapChart, LineChart, ScatterChart } from 'echarts/charts'
import { DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([HeatmapChart, LineChart, ScatterChart, DataZoomComponent, GridComponent, LegendComponent, MarkLineComponent, TooltipComponent, VisualMapComponent, CanvasRenderer])

export function EChart({ option, height = 320, label = 'data chart' }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current) return undefined
    const chart = echarts.init(ref.current, null, { renderer: 'canvas' })
    chart.setOption(option, { notMerge: true })
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(ref.current)
    return () => { observer.disconnect(); chart.dispose() }
  }, [option])
  return <div className="chart-canvas" ref={ref} style={{ height }} role="img" aria-label={label} />
}
