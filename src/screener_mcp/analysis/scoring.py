"""
Financial Health Scoring Engine.
Transparent, rules-based 0-100 scoring model across 6 fundamental pillars.
Documents exact weights, sub-metrics, points awarded, and rationale.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .growth_analysis import GrowthAnalysisResult
from .ratio_analysis import RatioAnalysisResult
from .valuation import ValuationAnalysisResult
from .red_flags import RedFlagsResult


@dataclass
class ScoreItem:
    metric: str
    actual_value: str
    points: float
    max_points: float
    rationale: str


@dataclass
class CategoryScore:
    category: str
    score: float  # 0 to 100
    grade: str    # "Excellent", "Good", "Moderate", "Weak", "Poor"
    items: List[ScoreItem] = field(default_factory=list)


@dataclass
class FinancialHealthScorecard:
    overall_score: float  # 0 to 100
    overall_grade: str    # "A+ (Outstanding)", "A (Strong)", "B (Average)", "C (Weak)", "D (Distressed)"
    growth_score: CategoryScore
    profitability_score: CategoryScore
    balance_sheet_score: CategoryScore
    cash_flow_score: CategoryScore
    valuation_score: CategoryScore
    shareholding_score: CategoryScore
    major_strengths: List[str]
    major_weaknesses: List[str]
    conclusion: str
    df_scorecard: pd.DataFrame


def _grade_score(score: float) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Moderate"
    elif score >= 35:
        return "Weak"
    return "Poor"


def _overall_grade(score: float) -> str:
    if score >= 85:
        return "A+ (Outstanding)"
    elif score >= 70:
        return "A (Strong)"
    elif score >= 55:
        return "B (Healthy / Average)"
    elif score >= 40:
        return "C (Weak / Cautious)"
    return "D (High Risk / Distressed)"


def calculate_financial_health_score(
    growth: GrowthAnalysisResult,
    ratios: RatioAnalysisResult,
    valuation: ValuationAnalysisResult,
    red_flags: RedFlagsResult,
    df_shareholding: pd.DataFrame,
) -> FinancialHealthScorecard:
    """Evaluate financial health across all pillars with fully documented rules."""

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Growth Score (Max 100)
    # ─────────────────────────────────────────────────────────────────────────
    g_items: List[ScoreItem] = []

    # Revenue 3Y CAGR (Max 25 pts)
    r_cagr = growth.revenue_growth.cagr_3y
    if not np.isnan(r_cagr):
        if r_cagr >= 0.20:
            pts = 25.0
            r_text = "Strong revenue expansion (>20% CAGR)"
        elif r_cagr >= 0.12:
            pts = 20.0
            r_text = "Solid revenue growth (12-20% CAGR)"
        elif r_cagr >= 0.06:
            pts = 12.0
            r_text = "Moderate revenue growth (6-12% CAGR)"
        elif r_cagr > 0:
            pts = 5.0
            r_text = "Sluggish positive growth (<6% CAGR)"
        else:
            pts = 0.0
            r_text = "Revenue contracting"
        g_items.append(ScoreItem("Revenue 3Y CAGR", f"{r_cagr*100:.1f}%", pts, 25.0, r_text))
    else:
        g_items.append(ScoreItem("Revenue 3Y CAGR", "N/A", 10.0, 25.0, "Historical baseline unavailable"))

    # Net Profit 3Y CAGR (Max 25 pts)
    p_cagr = growth.pat_growth.cagr_3y
    if not np.isnan(p_cagr):
        if p_cagr >= 0.20:
            pts = 25.0
            r_text = "Exceptional earnings growth (>20% CAGR)"
        elif p_cagr >= 0.12:
            pts = 20.0
            r_text = "Healthy earnings growth (12-20% CAGR)"
        elif p_cagr >= 0.06:
            pts = 12.0
            r_text = "Moderate earnings growth (6-12% CAGR)"
        elif p_cagr > 0:
            pts = 5.0
            r_text = "Low profit growth (<6% CAGR)"
        else:
            pts = 0.0
            r_text = "Negative earnings growth"
        g_items.append(ScoreItem("Net Profit 3Y CAGR", f"{p_cagr*100:.1f}%", pts, 25.0, r_text))
    else:
        g_items.append(ScoreItem("Net Profit 3Y CAGR", "N/A", 10.0, 25.0, "Historical baseline unavailable"))

    # EPS 1Y Growth (Max 25 pts)
    eps_yoy = growth.eps_growth.yoy_1y
    if not np.isnan(eps_yoy):
        if eps_yoy >= 0.15:
            pts = 25.0
            r_text = "Strong recent EPS acceleration (>=15%)"
        elif eps_yoy >= 0.08:
            pts = 20.0
            r_text = "Decent recent EPS growth (8-15%)"
        elif eps_yoy > 0:
            pts = 10.0
            r_text = "Mild positive EPS growth"
        else:
            pts = 0.0
            r_text = "EPS contraction"
        g_items.append(ScoreItem("EPS 1Y YoY Growth", f"{eps_yoy*100:.1f}%", pts, 25.0, r_text))
    else:
        g_items.append(ScoreItem("EPS 1Y YoY Growth", "N/A", 10.0, 25.0, "Data unavailable"))

    # Operating Margin Trend (Max 25 pts)
    m_exp = growth.margin_expansion_3y_bps
    if not np.isnan(m_exp):
        if m_exp >= 150:
            pts = 25.0
            r_text = "Significant margin expansion (+150 bps)"
        elif m_exp >= -50:
            pts = 20.0
            r_text = "Stable operating margins"
        elif m_exp >= -200:
            pts = 10.0
            r_text = "Mild margin compression"
        else:
            pts = 0.0
            r_text = "Heavy margin contraction (>200 bps drop)"
        g_items.append(ScoreItem("OPM 3Y Change", f"{m_exp/100:+.1f}%", pts, 25.0, r_text))
    else:
        g_items.append(ScoreItem("OPM 3Y Change", "N/A", 15.0, 25.0, "Data unavailable"))

    growth_tot = sum(i.points for i in g_items)
    score_growth = CategoryScore("Growth", growth_tot, _grade_score(growth_tot), g_items)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Profitability Score (Max 100)
    # ─────────────────────────────────────────────────────────────────────────
    p_items: List[ScoreItem] = []

    # ROCE Latest (Max 30 pts)
    roce = ratios.roce_latest
    if not np.isnan(roce):
        if roce >= 25.0:
            pts = 30.0
            r_text = "Outstanding return on capital (>=25%)"
        elif roce >= 18.0:
            pts = 24.0
            r_text = "Strong return on capital (18-25%)"
        elif roce >= 12.0:
            pts = 16.0
            r_text = "Acceptable return on capital (12-18%)"
        elif roce > 0:
            pts = 6.0
            r_text = "Low return on capital (<12%)"
        else:
            pts = 0.0
            r_text = "Negative ROCE"
        p_items.append(ScoreItem("ROCE", f"{roce:.1f}%", pts, 30.0, r_text))
    else:
        p_items.append(ScoreItem("ROCE", "N/A", 12.0, 30.0, "Data unavailable"))

    # ROE Latest (Max 30 pts)
    roe = ratios.roe_latest
    if not np.isnan(roe):
        if roe >= 22.0:
            pts = 30.0
            r_text = "High return on equity (>=22%)"
        elif roe >= 15.0:
            pts = 24.0
            r_text = "Good return on equity (15-22%)"
        elif roe >= 10.0:
            pts = 15.0
            r_text = "Moderate return on equity (10-15%)"
        elif roe > 0:
            pts = 5.0
            r_text = "Subpar return on equity (<10%)"
        else:
            pts = 0.0
            r_text = "Negative ROE"
        p_items.append(ScoreItem("ROE", f"{roe:.1f}%", pts, 30.0, r_text))
    else:
        p_items.append(ScoreItem("ROE", "N/A", 12.0, 30.0, "Data unavailable"))

    # Operating Profit Margin (Max 20 pts)
    opm = ratios.opm_latest
    if not np.isnan(opm):
        if opm >= 22.0:
            pts = 20.0
            r_text = "High operating profitability (>=22%)"
        elif opm >= 14.0:
            pts = 16.0
            r_text = "Healthy operating margins (14-22%)"
        elif opm >= 8.0:
            pts = 10.0
            r_text = "Average margins (8-14%)"
        elif opm > 0:
            pts = 4.0
            r_text = "Thin operating margins (<8%)"
        else:
            pts = 0.0
            r_text = "Operating loss"
        p_items.append(ScoreItem("Operating Margin (OPM)", f"{opm:.1f}%", pts, 20.0, r_text))
    else:
        p_items.append(ScoreItem("Operating Margin (OPM)", "N/A", 10.0, 20.0, "Data unavailable"))

    # Net Profit Margin (Max 20 pts)
    npm = ratios.npm_latest
    if not np.isnan(npm):
        if npm >= 15.0:
            pts = 20.0
            r_text = "Robust net profitability (>=15%)"
        elif npm >= 8.0:
            pts = 16.0
            r_text = "Decent net margin (8-15%)"
        elif npm > 0:
            pts = 8.0
            r_text = "Low net margin (<8%)"
        else:
            pts = 0.0
            r_text = "Net loss"
        p_items.append(ScoreItem("Net Margin (NPM)", f"{npm:.1f}%", pts, 20.0, r_text))
    else:
        p_items.append(ScoreItem("Net Margin (NPM)", "N/A", 10.0, 20.0, "Data unavailable"))

    prof_tot = sum(i.points for i in p_items)
    score_profitability = CategoryScore("Profitability", prof_tot, _grade_score(prof_tot), p_items)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Balance Sheet Score (Max 100)
    # ─────────────────────────────────────────────────────────────────────────
    b_items: List[ScoreItem] = []

    # Debt to Equity (Max 40 pts)
    de = ratios.debt_to_equity_latest
    if not np.isnan(de):
        if de <= 0.05:
            pts = 40.0
            r_text = "Virtually debt-free (D/E < 0.05)"
        elif de <= 0.35:
            pts = 35.0
            r_text = "Conservative leverage (D/E < 0.35)"
        elif de <= 0.75:
            pts = 25.0
            r_text = "Manageable debt (D/E 0.35-0.75)"
        elif de <= 1.25:
            pts = 12.0
            r_text = "Moderate to elevated leverage"
        else:
            pts = 0.0
            r_text = "High leverage (D/E > 1.25)"
        b_items.append(ScoreItem("Debt to Equity", f"{de:.2f}x", pts, 40.0, r_text))
    else:
        b_items.append(ScoreItem("Debt to Equity", "N/A", 25.0, 40.0, "Data unavailable"))

    # Interest Coverage (Max 35 pts)
    ic = ratios.interest_coverage_latest
    if not np.isnan(ic):
        if ic >= 15.0:
            pts = 35.0
            r_text = "Exceptional debt service capacity (IC >= 15x)"
        elif ic >= 6.0:
            pts = 30.0
            r_text = "Comfortable debt coverage (6-15x)"
        elif ic >= 2.5:
            pts = 18.0
            r_text = "Adequate debt coverage (2.5-6x)"
        elif ic >= 1.0:
            pts = 5.0
            r_text = "Strained interest coverage (1-2.5x)"
        else:
            pts = 0.0
            r_text = "Inability to service interest from operations"
        b_items.append(ScoreItem("Interest Coverage", f"{ic:.1f}x", pts, 35.0, r_text))
    else:
        b_items.append(ScoreItem("Interest Coverage", "N/A", 25.0, 35.0, "Virtually debt-free / negligible interest"))

    # Asset Quality / Working Capital (Max 25 pts)
    wc_days = ratios.working_capital_days_latest
    if not np.isnan(wc_days):
        if wc_days <= 45:
            pts = 25.0
            r_text = "Lean working capital cycle (<=45 days)"
        elif wc_days <= 90:
            pts = 20.0
            r_text = "Healthy working capital management (45-90 days)"
        elif wc_days <= 150:
            pts = 10.0
            r_text = "Moderate working capital stretch"
        else:
            pts = 2.0
            r_text = "High working capital requirement (>150 days)"
        b_items.append(ScoreItem("Working Capital Days", f"{wc_days:.0f} days", pts, 25.0, r_text))
    else:
        b_items.append(ScoreItem("Working Capital Days", "N/A", 15.0, 25.0, "Data unavailable"))

    bs_tot = sum(i.points for i in b_items)
    score_balance_sheet = CategoryScore("Balance Sheet", bs_tot, _grade_score(bs_tot), b_items)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Cash Flow Score (Max 100)
    # ─────────────────────────────────────────────────────────────────────────
    cf_items: List[ScoreItem] = []

    # CFO / Net Profit (Max 40 pts)
    cfo_pat = ratios.cfo_to_pat_3y_avg
    if not np.isnan(cfo_pat):
        if cfo_pat >= 1.0:
            pts = 40.0
            r_text = "Superb cash conversion (CFO >= 100% of PAT)"
        elif cfo_pat >= 0.80:
            pts = 32.0
            r_text = "Solid cash conversion (80-100% of PAT)"
        elif cfo_pat >= 0.60:
            pts = 20.0
            r_text = "Acceptable cash conversion (60-80% of PAT)"
        elif cfo_pat > 0:
            pts = 8.0
            r_text = "Poor earnings-to-cash conversion (<60%)"
        else:
            pts = 0.0
            r_text = "Negative operating cash flow"
        cf_items.append(ScoreItem("CFO / Net Profit (3Y Avg)", f"{cfo_pat:.2f}x", pts, 40.0, r_text))
    else:
        cf_items.append(ScoreItem("CFO / Net Profit (3Y Avg)", "N/A", 20.0, 40.0, "Data unavailable"))

    # Free Cash Flow Generation (Max 40 pts)
    fcf = ratios.fcf_latest
    pat_latest = growth.pat_growth.latest_value
    if not np.isnan(fcf):
        if fcf > 0 and not np.isnan(pat_latest) and pat_latest > 0 and (fcf / pat_latest) >= 0.6:
            pts = 40.0
            r_text = "High free cash flow generation (>60% of PAT)"
        elif fcf > 0:
            pts = 30.0
            r_text = "Positive Free Cash Flow"
        else:
            pts = 0.0
            r_text = "Negative Free Cash Flow"
        cf_items.append(ScoreItem("Free Cash Flow", f"Rs {fcf:,.1f} Cr", pts, 40.0, r_text))
    else:
        cf_items.append(ScoreItem("Free Cash Flow", "N/A", 20.0, 40.0, "Data unavailable"))

    # Cash Conversion Cycle (Max 20 pts)
    ccc = ratios.cash_conversion_cycle_latest
    if not np.isnan(ccc):
        if ccc <= 30:
            pts = 20.0
            r_text = "Excellent cash cycle (<=30 days)"
        elif ccc <= 75:
            pts = 15.0
            r_text = "Healthy cash conversion cycle"
        elif ccc <= 130:
            pts = 8.0
            r_text = "Extended cash conversion cycle"
        else:
            pts = 0.0
            r_text = "Working capital heavy (>130 days)"
        cf_items.append(ScoreItem("Cash Conversion Cycle", f"{ccc:.0f} days", pts, 20.0, r_text))
    else:
        cf_items.append(ScoreItem("Cash Conversion Cycle", "N/A", 12.0, 20.0, "Data unavailable"))

    cf_tot = sum(i.points for i in cf_items)
    score_cash_flow = CategoryScore("Cash Flow", cf_tot, _grade_score(cf_tot), cf_items)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Valuation Score (Max 100)
    # ─────────────────────────────────────────────────────────────────────────
    v_items: List[ScoreItem] = []

    # P/E Multiple (Max 40 pts)
    pe = valuation.pe_ratio
    if not np.isnan(pe):
        if pe <= 15.0:
            pts = 40.0
            r_text = "Attractive low valuation (P/E <= 15x)"
        elif pe <= 25.0:
            pts = 32.0
            r_text = "Fair reasonable multiple (P/E 15-25x)"
        elif pe <= 40.0:
            pts = 22.0
            r_text = "Moderate premium multiple (P/E 25-40x)"
        elif pe <= 65.0:
            pts = 10.0
            r_text = "High growth multiple (P/E 40-65x)"
        else:
            pts = 2.0
            r_text = "Very expensive multiple (P/E > 65x)"
        v_items.append(ScoreItem("P/E Ratio", f"{pe:.1f}x", pts, 40.0, r_text))
    else:
        v_items.append(ScoreItem("P/E Ratio", "N/A", 15.0, 40.0, "Loss-making or unavailable"))

    # P/E vs Peer Median (Max 30 pts)
    pe_rel = valuation.pe_vs_peer_median
    if not np.isnan(pe_rel):
        if pe_rel <= -20:
            pts = 30.0
            r_text = "Trading at significant discount to sector peers (>20% discount)"
        elif pe_rel <= 10:
            pts = 25.0
            r_text = "Valuation in line with sector peers"
        elif pe_rel <= 35:
            pts = 15.0
            r_text = "Modest peer valuation premium"
        else:
            pts = 5.0
            r_text = "Significant valuation premium over peers (>35%)"
        v_items.append(ScoreItem("P/E vs Peers", f"{pe_rel:+.1f}%", pts, 30.0, r_text))
    else:
        v_items.append(ScoreItem("P/E vs Peers", "N/A", 18.0, 30.0, "Peer benchmark unavailable"))

    # Dividend Yield (Max 30 pts)
    div_y = valuation.dividend_yield
    if not np.isnan(div_y):
        if div_y >= 3.0:
            pts = 30.0
            r_text = "High dividend yield (>=3.0%)"
        elif div_y >= 1.5:
            pts = 24.0
            r_text = "Decent dividend yield (1.5-3.0%)"
        elif div_y >= 0.5:
            pts = 15.0
            r_text = "Modest dividend yield"
        else:
            pts = 5.0
            r_text = "Low or zero dividend yield"
        v_items.append(ScoreItem("Dividend Yield", f"{div_y:.2f}%", pts, 30.0, r_text))
    else:
        v_items.append(ScoreItem("Dividend Yield", "N/A", 10.0, 30.0, "Data unavailable"))

    val_tot = sum(i.points for i in v_items)
    score_valuation = CategoryScore("Valuation", val_tot, _grade_score(val_tot), v_items)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Shareholding Score (Max 100)
    # ─────────────────────────────────────────────────────────────────────────
    sh_items: List[ScoreItem] = []

    # Promoter Holding (Max 50 pts)
    promoter_holding = np.nan
    fii_dii_holding = np.nan
    if df_shareholding is not None and not df_shareholding.empty:
        for idx in df_shareholding.index:
            idx_s = str(idx).lower()
            if "promoter" in idx_s:
                s_prom = df_shareholding.loc[idx].dropna()
                if not s_prom.empty:
                    promoter_holding = float(s_prom.iloc[-1])
            elif "fii" in idx_s or "dii" in idx_s or "institution" in idx_s:
                s_inst = df_shareholding.loc[idx].dropna()
                if not s_inst.empty:
                    fii_dii_holding = float(s_inst.iloc[-1]) if np.isnan(fii_dii_holding) else fii_dii_holding + float(s_inst.iloc[-1])

    if not np.isnan(promoter_holding):
        if promoter_holding >= 60.0:
            pts = 50.0
            r_text = "High promoter commitment (>=60%)"
        elif promoter_holding >= 45.0:
            pts = 40.0
            r_text = "Healthy promoter stake (45-60%)"
        elif promoter_holding >= 30.0:
            pts = 25.0
            r_text = "Moderate promoter stake (30-45%)"
        else:
            pts = 15.0
            r_text = "Low promoter holding (<30%)"
        sh_items.append(ScoreItem("Promoter Holding", f"{promoter_holding:.1f}%", pts, 50.0, r_text))
    else:
        sh_items.append(ScoreItem("Promoter Holding", "N/A", 30.0, 50.0, "Institutional-held / board-managed"))

    # Institutional Ownership (FII + DII) (Max 50 pts)
    if not np.isnan(fii_dii_holding):
        if fii_dii_holding >= 30.0:
            pts = 50.0
            r_text = "Strong institutional backing (FII+DII >= 30%)"
        elif fii_dii_holding >= 15.0:
            pts = 40.0
            r_text = "Good institutional presence (15-30%)"
        else:
            pts = 20.0
            r_text = "Limited institutional interest"
        sh_items.append(ScoreItem("Institutional Stake", f"{fii_dii_holding:.1f}%", pts, 50.0, r_text))
    else:
        sh_items.append(ScoreItem("Institutional Stake", "N/A", 30.0, 50.0, "Data unavailable"))

    sh_tot = sum(i.points for i in sh_items)
    score_shareholding = CategoryScore("Shareholding", sh_tot, _grade_score(sh_tot), sh_items)

    # ─────────────────────────────────────────────────────────────────────────
    # Overall Score (Weighted)
    # Weights: Profitability (25%), Growth (20%), Balance Sheet (20%), Cash Flow (15%), Valuation (10%), Shareholding (10%)
    # ─────────────────────────────────────────────────────────────────────────
    overall = (
        0.25 * prof_tot +
        0.20 * growth_tot +
        0.20 * bs_tot +
        0.15 * cf_tot +
        0.10 * val_tot +
        0.10 * sh_tot
    )

    # Deduct penalties for HIGH severity red flags (-3 pts per High red flag)
    penalty = min(red_flags.high_count * 4.0 + red_flags.medium_count * 1.5, 20.0)
    overall_adj = max(0.0, min(100.0, overall - penalty))

    # Identify Strengths & Weaknesses
    strengths = []
    weaknesses = []

    if roce >= 20.0:
        strengths.append(f"Outstanding capital efficiency with ROCE of {roce:.1f}%")
    if de <= 0.2:
        strengths.append(f"Robust balance sheet with negligible debt (D/E: {de:.2f}x)")
    if cfo_pat >= 0.9:
        strengths.append(f"High cash conversion quality (CFO/PAT at {cfo_pat:.2f}x)")
    if r_cagr >= 0.12:
        strengths.append(f"Healthy 3-year revenue compounding at {r_cagr*100:.1f}% CAGR")
    if p_cagr >= 0.15:
        strengths.append(f"Strong earnings momentum (3Y PAT CAGR: {p_cagr*100:.1f}%)")
    if div_y >= 2.0:
        strengths.append(f"Attractive shareholder cash return (Dividend Yield: {div_y:.2f}%)")

    if de > 1.0:
        weaknesses.append(f"Elevated balance sheet debt (Debt to Equity: {de:.2f}x)")
    if roce < 12.0 and not np.isnan(roce):
        weaknesses.append(f"Subdued capital productivity (ROCE: {roce:.1f}%)")
    if cfo_pat < 0.7 and not np.isnan(cfo_pat):
        weaknesses.append(f"Weak cash conversion (CFO is only {cfo_pat*100:.0f}% of Net Profit)")
    if pe > 50.0 and not np.isnan(pe):
        weaknesses.append(f"Rich valuation multiple (P/E: {pe:.1f}x)")
    if r_cagr < 0.05 and not np.isnan(r_cagr):
        weaknesses.append("Sluggish medium-term revenue expansion")
    for rf in red_flags.flags:
        if rf.severity == "HIGH" and rf.metric not in weaknesses:
            weaknesses.append(f"{rf.metric}: {rf.reason}")

    if not strengths:
        strengths.append("Operating in a stable market segment")
    if not weaknesses:
        weaknesses.append("No critical fundamental red flags detected")

    # Conclusion
    conclusion = (
        f"The company scores {overall_adj:.1f}/100, earning an overall rating of {_overall_grade(overall_adj)}. "
        f"Key performance drivers include {score_profitability.grade.lower()} profitability ({prof_tot:.0f}/100) and "
        f"{score_balance_sheet.grade.lower()} balance sheet resilience ({bs_tot:.0f}/100). "
    )
    if red_flags.high_count > 0:
        conclusion += f"However, {red_flags.high_count} high-severity warning signs require investor monitoring. "
    else:
        conclusion += "Financial health remains sound with manageable balance sheet risks. "

    df_scorecard = pd.DataFrame([
        {"Pillar": "Profitability", "Weight": "25%", "Score (0-100)": round(prof_tot, 1), "Rating": score_profitability.grade},
        {"Pillar": "Growth", "Weight": "20%", "Score (0-100)": round(growth_tot, 1), "Rating": score_growth.grade},
        {"Pillar": "Balance Sheet", "Weight": "20%", "Score (0-100)": round(bs_tot, 1), "Rating": score_balance_sheet.grade},
        {"Pillar": "Cash Flow", "Weight": "15%", "Score (0-100)": round(cf_tot, 1), "Rating": score_cash_flow.grade},
        {"Pillar": "Valuation", "Weight": "10%", "Score (0-100)": round(val_tot, 1), "Rating": score_valuation.grade},
        {"Pillar": "Shareholding", "Weight": "10%", "Score (0-100)": round(sh_tot, 1), "Rating": score_shareholding.grade},
        {"Pillar": "OVERALL HEALTH", "Weight": "100%", "Score (0-100)": round(overall_adj, 1), "Rating": _overall_grade(overall_adj)},
    ]).set_index("Pillar")

    return FinancialHealthScorecard(
        overall_score=round(overall_adj, 1),
        overall_grade=_overall_grade(overall_adj),
        growth_score=score_growth,
        profitability_score=score_profitability,
        balance_sheet_score=score_balance_sheet,
        cash_flow_score=score_cash_flow,
        valuation_score=score_valuation,
        shareholding_score=score_shareholding,
        major_strengths=strengths[:5],
        major_weaknesses=weaknesses[:5],
        conclusion=conclusion,
        df_scorecard=df_scorecard,
    )
