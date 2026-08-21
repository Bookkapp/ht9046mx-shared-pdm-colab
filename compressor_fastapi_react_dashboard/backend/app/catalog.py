from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    short_label: str
    family: str
    unit: str
    source: str
    color: str
    description: str


METRICS: tuple[Metric, ...] = (
    Metric("hp1", "COM1 high pressure", "HP 1st", "COM1", "source pressure", "signal", "#c8102e", "First-stage high-side pressure."),
    Metric("lp1", "COM1 low pressure", "LP 1st", "COM1", "source pressure", "signal", "#7f1d2d", "First-stage low-side pressure."),
    Metric("hp2", "COM2 high pressure", "HP 2nd", "COM2", "source pressure", "signal", "#0067b1", "Second-stage high-side pressure used by the primary detector."),
    Metric("lp2", "COM2 low pressure", "LP 2nd", "COM2", "source pressure", "signal", "#003b5c", "Second-stage low-side pressure modelled conditionally by Ridge."),
    Metric("valve", "Expansion valve", "Valve", "CONTROL", "source position", "signal", "#d97706", "Valve position used in operating mode, GMM, and Ridge."),
    Metric("temphi", "High temperature", "Temp high", "THERMAL", "°C", "signal", "#b42318", "Upper temperature channel."),
    Metric("templo", "Low temperature", "Temp low", "THERMAL", "°C", "signal", "#0e7490", "Lower temperature channel."),
    Metric("pressure_gap", "COM2 pressure gap", "HP2 − LP2", "ENGINEERED", "source pressure", "signal", "#6941c6", "HP2 minus LP2."),
    Metric("pressure_ratio", "COM2 pressure ratio", "HP2 / |LP2|", "ENGINEERED", "ratio", "signal", "#475467", "HP2 divided by max(abs(LP2), 0.1)."),
    Metric("temperature_span", "Temperature span", "Temp span", "ENGINEERED", "°C", "signal", "#00856a", "TempHi minus TempLo."),
    Metric("busy", "Module busy flag", "Busy", "STATE / QUALITY", "0 or 1", "signal", "#d97706", "Median Busy flag inside the event-time window."),
    Metric("coverage", "Window coverage", "Coverage", "STATE / QUALITY", "ratio", "signal", "#00856a", "Observed points divided by expected points in the five-minute window."),
    Metric("point_count", "Window point count", "Points", "STATE / QUALITY", "rows", "signal", "#475467", "Raw logger rows observed in the five-minute window."),
    Metric("maximum_gap_seconds", "Maximum sampling gap", "Max gap", "STATE / QUALITY", "seconds", "signal", "#b42318", "Largest positive event-time gap inside the window."),
    Metric("z_hp2", "Robust HP2 Z", "Z HP2", "COM2 EVIDENCE", "robust z", "model", "#c8102e", "HP2 deviation from context median scaled by 1.4826 MAD."),
    Metric("z_lp2_residual", "LP2 residual Robust Z", "Z LP2 residual", "COM2 EVIDENCE", "robust z", "model", "#0067b1", "Conditional LP2 residual deviation from its context baseline."),
    Metric("z_pressure_gap", "Pressure-gap Robust Z", "Z gap", "COM2 EVIDENCE", "robust z", "model", "#6941c6", "Pressure-gap deviation in the selected frozen context."),
    Metric("z_temperature_span", "Temperature-span Robust Z", "Z temp span", "COM2 EVIDENCE", "robust z", "model", "#00856a", "Temperature-span deviation in the selected frozen context."),
    Metric("lp2_residual", "LP2 conditional residual", "LP2 residual", "COM2 EVIDENCE", "source pressure", "model", "#d97706", "Actual LP2 minus Ridge expected LP2."),
    Metric("isolation_score", "Isolation Forest score", "IF score", "COM2 EVIDENCE", "score", "model", "#7f56d9", "Multivariate outlier score; it is not a failure probability."),
    Metric("lstm_score", "LSTM reconstruction score", "LSTM score", "LSTM EVIDENCE", "MAE", "model", "#101820", "P95 sequence reconstruction error in the five-minute bucket."),
    Metric("lstm_ratio", "LSTM score / threshold", "LSTM ratio", "LSTM EVIDENCE", "ratio", "model", "#344054", "Group-calibrated reconstruction score divided by its threshold."),
    Metric("regime_posterior", "GMM regime posterior", "Regime posterior", "REGIME", "probability", "model", "#0e7490", "Posterior for the selected component; not failure probability."),
    Metric("regime_log_likelihood", "GMM log likelihood", "Regime log-L", "REGIME", "log density", "model", "#667085", "Likelihood of the context under the frozen regime model."),
)

METRIC_BY_KEY = {metric.key: metric for metric in METRICS}

MODULES: tuple[dict[str, Any], ...] = (
    {"module_no": 1, "name": "Index L Arm1", "group": "Index"},
    {"module_no": 2, "name": "Index R Arm1", "group": "Index"},
    {"module_no": 3, "name": "Index L Arm2", "group": "Index"},
    {"module_no": 4, "name": "Index R Arm2", "group": "Index"},
    {"module_no": 5, "name": "SH1", "group": "Shuttle"},
    {"module_no": 6, "name": "SH2", "group": "Shuttle"},
    {"module_no": 7, "name": "Hot1 (excluded from model)", "group": "Hot"},
    {"module_no": 8, "name": "Hot2", "group": "Hot"},
)


def catalog_payload() -> dict[str, Any]:
    return {
        "source_grain": "raw event-time log rows",
        "monitor_grain": "five-minute event-time window by machine and module",
        "modules": list(MODULES),
        "metrics": [asdict(metric) for metric in METRICS],
        "max_compare_series": 6,
        "warning": "Anomaly evidence and GMM posterior are not failure probabilities.",
    }
