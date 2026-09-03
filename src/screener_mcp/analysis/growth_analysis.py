"""
Growth Analysis Engine.
Calculates 1Y YoY, 3Y CAGR, 5Y CAGR, 10Y CAGR, margin expansion/contraction.
Handles missing data cleanly using NaN (no invented numbers).
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def compute_cagr(start_val: float, end_val: float, periods: int) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR).
    Returns NaN if start/end values are <= 0, missing, or periods <= 0.
    """
    if periods <= 0:
        return np.nan
    if pd.isna(start_val) or pd.isna(end_val):
        return np.nan
    if start_val <= 0 or end_val <= 0:
        # CAGR is mathematically undefined for non-positive baselines
        return np.nan
    try:
        return (end_val / start_val) ** (1.0 / periods) - 1.0
    except (ZeroDivisionError, OverflowError, ValueError):
        return np.nan


def get_series_by_label(df: pd.DataFrame, target_labels: list[str]) -> Optional[pd.Series]:
    """Find the first matching row in df by case-insensitive substring match."""
    if df is None or df.empty:
        return None
    for target in target_labels:
        target_lower = target.lower()
        for idx in df.index:
            if target_lower in str(idx).lower():
                return df.loc[idx]
    return None


@dataclass
class MetricGrowthSummary:
    metric_name: str
    latest_value: float
    yoy_1y: float
    cagr_3y: float
    cagr_5y: float
    cagr_10y: float
    trend: str  # "Accelerating", "Stable", "Decelerating", "Declining", "Volatile"
    historical_yoy: list[float] = field(default_factory=list)


@dataclass
class GrowthAnalysisResult:
    revenue_growth: MetricGrowthSummary
    ebitda_growth: MetricGrowthSummary
    pat_growth: MetricGrowthSummary
    eps_growth: MetricGrowthSummary
    margin_expansion_1y_bps: float
    margin_expansion_3y_bps: float
    margin_expansion_5y_bps: float
    opm_trend: str
    npm_trend: str
    df_growth_summary: pd.DataFrame
    raw_cagr_tables: dict[str, dict[str, str]] = field(default_factory=dict)


def _determine_trend(yoy_list: list[float]) -> str:
    valid = [x for x in yoy_list if not pd.isna(x)]
    if len(valid) < 2:
        return "Insufficient Data"
    recent = valid[-3:] if len(valid) >= 3 else valid
    if all(x > 0 for x in recent):
        if len(recent) >= 2 and recent[-1] > recent[-2] > (recent[-3] if len(recent) >= 3 else 0):
            return "Accelerating"
        return "Consistent Growth"
    elif all(x < 0 for x in recent):
        return "Declining"
    elif any(x < 0 for x in recent):
        return "Volatile"
    return "Stable"


def analyze_metric_growth(series: pd.Series, metric_name: str) -> MetricGrowthSummary:
    """Analyze historical growth of a single metric series."""
    # Exclude 'TTM' if present to focus on fiscal year ends, or keep numeric columns
    cols = [c for c in series.index if str(c).strip().upper() != "TTM"]
    clean_series = series[cols].dropna()

    if clean_series.empty:
        return MetricGrowthSummary(
            metric_name=metric_name,
            latest_value=np.nan,
            yoy_1y=np.nan,
            cagr_3y=np.nan,
            cagr_5y=np.nan,
            cagr_10y=np.nan,
            trend="No Data",
        )

    vals = clean_series.values
    n = len(vals)
    latest_val = float(vals[-1]) if n > 0 else np.nan

    # 1Y YoY
    yoy_1y = (vals[-1] / vals[-2] - 1.0) if n >= 2 and vals[-2] != 0 else np.nan

    # CAGRs
    cagr_3y = compute_cagr(vals[-4], vals[-1], 3) if n >= 4 else np.nan
    cagr_5y = compute_cagr(vals[-6], vals[-1], 5) if n >= 6 else np.nan
    cagr_10y = compute_cagr(vals[-11], vals[-1], 10) if n >= 11 else np.nan

    # Historical YoY
    hist_yoy = []
    for i in range(1, n):
        prev = vals[i - 1]
        curr = vals[i]
        if prev != 0 and not pd.isna(prev) and not pd.isna(curr):
            hist_yoy.append(curr / prev - 1.0)
        else:
            hist_yoy.append(np.nan)

    trend = _determine_trend(hist_yoy)

    return MetricGrowthSummary(
        metric_name=metric_name,
        latest_value=latest_val,
        yoy_1y=yoy_1y,
        cagr_3y=cagr_3y,
        cagr_5y=cagr_5y,
        cagr_10y=cagr_10y,
        trend=trend,
        historical_yoy=hist_yoy,
    )


def analyze_growth(df_pl: pd.DataFrame, compounding_dict: Optional[dict] = None) -> GrowthAnalysisResult:
    """
    Perform comprehensive growth analysis across P&L statement.
    """
    rev_series = get_series_by_label(df_pl, ["Sales", "Revenue", "Total Revenue"])
    ebitda_series = get_series_by_label(df_pl, ["Operating Profit", "EBITDA"])
    pat_series = get_series_by_label(df_pl, ["Net Profit", "PAT"])
    eps_series = get_series_by_label(df_pl, ["EPS in Rs", "EPS"])
    opm_series = get_series_by_label(df_pl, ["OPM %", "OPM"])

    rev_growth = analyze_metric_growth(rev_series, "Revenue") if rev_series is not None else MetricGrowthSummary("Revenue", np.nan, np.nan, np.nan, np.nan, np.nan, "No Data")
    ebitda_growth = analyze_metric_growth(ebitda_series, "Operating Profit") if ebitda_series is not None else MetricGrowthSummary("Operating Profit", np.nan, np.nan, np.nan, np.nan, np.nan, "No Data")
    pat_growth = analyze_metric_growth(pat_series, "Net Profit") if pat_series is not None else MetricGrowthSummary("Net Profit", np.nan, np.nan, np.nan, np.nan, np.nan, "No Data")
    eps_growth = analyze_metric_growth(eps_series, "EPS") if eps_series is not None else MetricGrowthSummary("EPS", np.nan, np.nan, np.nan, np.nan, np.nan, "No Data")

    # Margin expansion in bps (1% = 100 bps)
    margin_1y_bps = np.nan
    margin_3y_bps = np.nan
    margin_5y_bps = np.nan
    opm_trend = "Stable"

    if opm_series is not None:
        opm_cols = [c for c in opm_series.index if str(c).strip().upper() != "TTM"]
        clean_opm = opm_series[opm_cols].dropna().values
        if len(clean_opm) >= 2:
            margin_1y_bps = (clean_opm[-1] - clean_opm[-2]) * 100
        if len(clean_opm) >= 4:
            margin_3y_bps = (clean_opm[-1] - clean_opm[-4]) * 100
        if len(clean_opm) >= 6:
            margin_5y_bps = (clean_opm[-1] - clean_opm[-6]) * 100

        if not np.isnan(margin_3y_bps):
            if margin_3y_bps > 150:
                opm_trend = "Expanding"
            elif margin_3y_bps < -150:
                opm_trend = "Contracting"

    # Summary table
    rows = [
        {
            "Metric": "Revenue (Sales)",
            "Latest FY": rev_growth.latest_value,
            "1Y YoY": rev_growth.yoy_1y,
            "3Y CAGR": rev_growth.cagr_3y,
            "5Y CAGR": rev_growth.cagr_5y,
            "10Y CAGR": rev_growth.cagr_10y,
            "Trend": rev_growth.trend,
        },
        {
            "Metric": "Operating Profit (EBITDA)",
            "Latest FY": ebitda_growth.latest_value,
            "1Y YoY": ebitda_growth.yoy_1y,
            "3Y CAGR": ebitda_growth.cagr_3y,
            "5Y CAGR": ebitda_growth.cagr_5y,
            "10Y CAGR": ebitda_growth.cagr_10y,
            "Trend": ebitda_growth.trend,
        },
        {
            "Metric": "Net Profit (PAT)",
            "Latest FY": pat_growth.latest_value,
            "1Y YoY": pat_growth.yoy_1y,
            "3Y CAGR": pat_growth.cagr_3y,
            "5Y CAGR": pat_growth.cagr_5y,
            "10Y CAGR": pat_growth.cagr_10y,
            "Trend": pat_growth.trend,
        },
        {
            "Metric": "EPS",
            "Latest FY": eps_growth.latest_value,
            "1Y YoY": eps_growth.yoy_1y,
            "3Y CAGR": eps_growth.cagr_3y,
            "5Y CAGR": eps_growth.cagr_5y,
            "10Y CAGR": eps_growth.cagr_10y,
            "Trend": eps_growth.trend,
        },
    ]

    df_summary = pd.DataFrame(rows).set_index("Metric")

    return GrowthAnalysisResult(
        revenue_growth=rev_growth,
        ebitda_growth=ebitda_growth,
        pat_growth=pat_growth,
        eps_growth=eps_growth,
        margin_expansion_1y_bps=margin_1y_bps,
        margin_expansion_3y_bps=margin_3y_bps,
        margin_expansion_5y_bps=margin_5y_bps,
        opm_trend=opm_trend,
        npm_trend="Expanding" if not np.isnan(pat_growth.cagr_3y) and not np.isnan(rev_growth.cagr_3y) and pat_growth.cagr_3y > rev_growth.cagr_3y else "Contracting",
        df_growth_summary=df_summary,
        raw_cagr_tables=compounding_dict or {},
    )
