"""
Red Flag Detection Engine.
Identifies governance, operational, debt, cash flow, and valuation warning signs.
Outputs Metric, Current Value, Historical Value, Severity, and Reason for every flag.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .growth_analysis import GrowthAnalysisResult
from .ratio_analysis import RatioAnalysisResult
from .valuation import ValuationAnalysisResult


@dataclass
class RedFlag:
    metric: str
    current_value: str
    historical_value: str
    reason: str
    severity: str  # "HIGH", "MEDIUM", "LOW"


@dataclass
class RedFlagsResult:
    flags: List[RedFlag]
    flag_count: int
    high_count: int
    medium_count: int
    low_count: int
    df_red_flags: pd.DataFrame


def detect_red_flags(
    growth: GrowthAnalysisResult,
    ratios: RatioAnalysisResult,
    valuation: ValuationAnalysisResult,
    df_bs: pd.DataFrame,
    df_cf: pd.DataFrame,
    df_shareholding: pd.DataFrame,
) -> RedFlagsResult:
    """Run comprehensive forensic checks across financial statements."""
    flags: List[RedFlag] = []

    # 1. Debt Increasing Rapidly
    if df_bs is not None and not df_bs.empty:
        borrowings_row = None
        for idx in df_bs.index:
            if "borrowing" in str(idx).lower() or "debt" in str(idx).lower():
                borrowings_row = df_bs.loc[idx].dropna()
                break

        if borrowings_row is not None and len(borrowings_row) >= 2:
            cols = [c for c in borrowings_row.index if str(c).strip().upper() != "TTM"]
            clean_b = borrowings_row[cols]
            if len(clean_b) >= 2:
                latest_debt = float(clean_b.iloc[-1])
                prev_debt = float(clean_b.iloc[-2])
                old_debt = float(clean_b.iloc[-4]) if len(clean_b) >= 4 else float(clean_b.iloc[0])

                if latest_debt > 100:  # Ignore negligible debt amounts
                    yoy_debt_change = (latest_debt - prev_debt) / prev_debt if prev_debt > 0 else 0
                    if yoy_debt_change > 0.35:
                        flags.append(RedFlag(
                            metric="Debt Growth (1Y)",
                            current_value=f"Rs {latest_debt:,.1f} Cr",
                            historical_value=f"Rs {prev_debt:,.1f} Cr",
                            reason=f"Borrowings surged by {yoy_debt_change*100:.1f}% in the latest fiscal year.",
                            severity="HIGH" if ratios.debt_to_equity_latest > 0.8 else "MEDIUM",
                        ))
                    elif len(clean_b) >= 4 and old_debt > 0 and (latest_debt / old_debt) ** (1/3) - 1 > 0.20:
                        cagr_d = ((latest_debt / old_debt) ** (1/3) - 1) * 100
                        flags.append(RedFlag(
                            metric="Debt Growth (3Y CAGR)",
                            current_value=f"Rs {latest_debt:,.1f} Cr",
                            historical_value=f"Rs {old_debt:,.1f} Cr",
                            reason=f"Debt has been compounding rapidly at {cagr_d:.1f}% CAGR over 3 years.",
                            severity="HIGH" if ratios.debt_to_equity_latest > 1.0 else "MEDIUM",
                        ))

    # 2. ROE Declining
    if not np.isnan(ratios.roe_latest) and not np.isnan(ratios.roe_3y_avg):
        roe_drop = ratios.roe_3y_avg - ratios.roe_latest
        if roe_drop > 4.0 and ratios.roe_latest < 15.0:
            flags.append(RedFlag(
                metric="Return on Equity (ROE)",
                current_value=f"{ratios.roe_latest:.1f}%",
                historical_value=f"{ratios.roe_3y_avg:.1f}% (3Y Avg)",
                reason=f"ROE has contracted by {roe_drop:.1f} percentage points relative to its 3-year baseline.",
                severity="HIGH" if ratios.roe_latest < 10 else "MEDIUM",
            ))

    # 3. ROCE Declining
    if not np.isnan(ratios.roce_latest) and not np.isnan(ratios.roce_3y_avg):
        roce_drop = ratios.roce_3y_avg - ratios.roce_latest
        if roce_drop > 4.0 and ratios.roce_latest < 15.0:
            flags.append(RedFlag(
                metric="Return on Capital Employed (ROCE)",
                current_value=f"{ratios.roce_latest:.1f}%",
                historical_value=f"{ratios.roce_3y_avg:.1f}% (3Y Avg)",
                reason=f"ROCE declined by {roce_drop:.1f} percentage points below 3-year average.",
                severity="HIGH" if ratios.roce_latest < 12 else "MEDIUM",
            ))

    # 4. Profit Growth Slowing / Negative
    if not np.isnan(growth.pat_growth.yoy_1y) and growth.pat_growth.yoy_1y < -0.10:
        flags.append(RedFlag(
            metric="Net Profit (PAT) Decline",
            current_value=f"{growth.pat_growth.yoy_1y*100:+.1f}% YoY",
            historical_value=f"{growth.pat_growth.cagr_3y*100:.1f}% 3Y CAGR" if not np.isnan(growth.pat_growth.cagr_3y) else "N/A",
            reason=f"Net profit fell by {abs(growth.pat_growth.yoy_1y)*100:.1f}% in the latest period.",
            severity="HIGH" if growth.pat_growth.yoy_1y < -0.25 else "MEDIUM",
        ))

    # 5. Revenue Growth Slowing
    if not np.isnan(growth.revenue_growth.yoy_1y) and growth.revenue_growth.yoy_1y < -0.05:
        flags.append(RedFlag(
            metric="Revenue De-growth",
            current_value=f"{growth.revenue_growth.yoy_1y*100:+.1f}% YoY",
            historical_value=f"{growth.revenue_growth.cagr_3y*100:.1f}% 3Y CAGR" if not np.isnan(growth.revenue_growth.cagr_3y) else "N/A",
            reason="Top-line sales contracted over the past year.",
            severity="HIGH" if growth.revenue_growth.yoy_1y < -0.15 else "MEDIUM",
        ))

    # 6. CFO Consistently Below Net Profit (Cash Conversion Quality)
    if not np.isnan(ratios.cfo_to_pat_3y_avg):
        if ratios.cfo_to_pat_3y_avg < 0.70 and not np.isnan(growth.pat_growth.latest_value) and growth.pat_growth.latest_value > 0:
            flags.append(RedFlag(
                metric="Cash Flow from Operations / PAT",
                current_value=f"{ratios.cfo_to_pat_latest:.2f}x" if not np.isnan(ratios.cfo_to_pat_latest) else "N/A",
                historical_value=f"{ratios.cfo_to_pat_3y_avg:.2f}x (3Y Avg)",
                reason=f"Cash conversion is weak; CFO is only {ratios.cfo_to_pat_3y_avg*100:.0f}% of reported accounting profits over 3 years.",
                severity="HIGH" if ratios.cfo_to_pat_3y_avg < 0.50 else "MEDIUM",
            ))

    # 7. Negative Free Cash Flow
    if not np.isnan(ratios.fcf_latest) and ratios.fcf_latest < 0:
        flags.append(RedFlag(
            metric="Free Cash Flow",
            current_value=f"Rs {ratios.fcf_latest:,.1f} Cr",
            historical_value="Negative",
            reason="Operating cash flow is insufficient to cover capital expenditures, resulting in negative Free Cash Flow.",
            severity="HIGH" if ratios.debt_to_equity_latest > 0.5 else "MEDIUM",
        ))

    # 8. Receivables / Debtor Days Deterioration
    if not np.isnan(ratios.debtor_days_latest) and ratios.df_efficiency is not None and not ratios.df_efficiency.empty:
        if "Debtor Days" in ratios.df_efficiency.index:
            s_dd = ratios.df_efficiency.loc["Debtor Days"].dropna()
            if len(s_dd) >= 3:
                prev_dd = float(s_dd.iloc[-3])
                if (ratios.debtor_days_latest - prev_dd) > 15:
                    flags.append(RedFlag(
                        metric="Debtor Days Lengthening",
                        current_value=f"{ratios.debtor_days_latest:.0f} days",
                        historical_value=f"{prev_dd:.0f} days (3Y ago)",
                        reason=f"Debtor collection period lengthened by {ratios.debtor_days_latest - prev_dd:.0f} days, tying up working capital.",
                        severity="MEDIUM",
                    ))

    # 9. Margin Deterioration
    if not np.isnan(growth.margin_expansion_3y_bps) and growth.margin_expansion_3y_bps < -250:
        flags.append(RedFlag(
            metric="Operating Margin (OPM)",
            current_value=f"{ratios.opm_latest:.1f}%",
            historical_value=f"{growth.margin_expansion_3y_bps/100:+.1f}% 3Y change",
            reason=f"Operating margin contracted by {abs(growth.margin_expansion_3y_bps)/100:.1f} percentage points over 3 years.",
            severity="MEDIUM",
        ))

    # 10. Promoter Holding Decline
    if df_shareholding is not None and not df_shareholding.empty:
        promoter_row = None
        for idx in df_shareholding.index:
            if "promoter" in str(idx).lower():
                promoter_row = df_shareholding.loc[idx].dropna()
                break

        if promoter_row is not None and len(promoter_row) >= 2:
            latest_sh = float(promoter_row.iloc[-1])
            oldest_sh = float(promoter_row.iloc[0])
            drop = oldest_sh - latest_sh
            if drop > 2.0:
                flags.append(RedFlag(
                    metric="Promoter Holding Decline",
                    current_value=f"{latest_sh:.1f}%",
                    historical_value=f"{oldest_sh:.1f}% ({len(promoter_row)} quarters ago)",
                    reason=f"Promoter stake dropped by {drop:.1f}% over the tracked period.",
                    severity="HIGH" if drop > 5.0 else "MEDIUM",
                ))

    # 11. High / Excessive Valuation
    if not np.isnan(valuation.pe_ratio):
        if valuation.pe_ratio > 65 and (np.isnan(growth.pat_growth.cagr_3y) or growth.pat_growth.cagr_3y < 0.15):
            flags.append(RedFlag(
                metric="Valuation (P/E Multiple)",
                current_value=f"{valuation.pe_ratio:.1f}x",
                historical_value=f"Peer Median: {valuation.peer_median_pe:.1f}x" if not np.isnan(valuation.peer_median_pe) else "Elevated",
                reason=f"P/E ratio of {valuation.pe_ratio:.1f}x appears stretched relative to underlying earnings growth.",
                severity="HIGH" if valuation.pe_ratio > 85 else "MEDIUM",
            ))

    # 12. Heavy Leverage (Debt / Equity > 1.5)
    if not np.isnan(ratios.debt_to_equity_latest) and ratios.debt_to_equity_latest > 1.5:
        flags.append(RedFlag(
            metric="Debt to Equity Ratio",
            current_value=f"{ratios.debt_to_equity_latest:.2f}x",
            historical_value="Target: < 1.0x",
            reason=f"Company carries elevated balance sheet leverage with Debt/Equity of {ratios.debt_to_equity_latest:.2f}x.",
            severity="HIGH" if ratios.debt_to_equity_latest > 2.0 else "MEDIUM",
        ))

    high_c = sum(1 for f in flags if f.severity == "HIGH")
    med_c = sum(1 for f in flags if f.severity == "MEDIUM")
    low_c = sum(1 for f in flags if f.severity == "LOW")

    # DataFrame representation
    if flags:
        df_rf = pd.DataFrame([
            {
                "Severity": f.severity,
                "Metric": f.metric,
                "Current Value": f.current_value,
                "Historical Value": f.historical_value,
                "Reason": f.reason,
            }
            for f in flags
        ])
    else:
        df_rf = pd.DataFrame(columns=["Severity", "Metric", "Current Value", "Historical Value", "Reason"])

    return RedFlagsResult(
        flags=flags,
        flag_count=len(flags),
        high_count=high_c,
        medium_count=med_c,
        low_count=low_c,
        df_red_flags=df_rf,
    )
