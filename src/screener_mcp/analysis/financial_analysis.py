"""
Master Financial Analysis Orchestrator.
Combines Data Collection, Growth Analysis, Ratio Analysis, Valuation,
Red Flags Engine, and Financial Health Scoring into a unified report.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .data_collector import CollectedFinancialData, collect_fundamental_data
from .growth_analysis import GrowthAnalysisResult, analyze_growth
from .ratio_analysis import RatioAnalysisResult, analyze_ratios
from .valuation import ValuationAnalysisResult, analyze_valuation
from .red_flags import RedFlagsResult, detect_red_flags
from .scoring import FinancialHealthScorecard, calculate_financial_health_score


@dataclass
class ComprehensiveAnalysis:
    symbol: str
    data: CollectedFinancialData
    growth: GrowthAnalysisResult
    ratios: RatioAnalysisResult
    valuation: ValuationAnalysisResult
    red_flags: RedFlagsResult
    scorecard: FinancialHealthScorecard

    def to_markdown_summary(self) -> str:
        """Produce a clean, executive markdown summary for LLM / chat rendering."""
        ov = self.data.overview
        name = ov.get("name", self.symbol)
        price = ov.get("current_price", "N/A")
        mcap = ov.get("key_ratios", {}).get("Market Cap", "N/A")
        sectors = ", ".join(ov.get("sectors", [])) or "N/A"

        lines = [
            f"# {name} ({self.symbol}) - Comprehensive Fundamental Analysis",
            f"**Sector / Industry**: {sectors}  |  **Current Price**: Rs {price}  |  **Market Cap**: Rs {mcap} Cr",
            f"**Retrieval Timestamp**: {self.data.retrieval_date}  |  **Statement Mode**: {self.data.financial_type.capitalize()}",
            "",
            "---",
            "",
            f"## [Score] Financial Health Score: {self.scorecard.overall_score}/100 - {self.scorecard.overall_grade}",
            "",
            "| Fundamental Pillar | Weight | Score (0-100) | Rating |",
            "|---|---|---|---|",
            f"| Profitability | 25% | **{self.scorecard.profitability_score.score:.0f}** | {self.scorecard.profitability_score.grade} |",
            f"| Growth | 20% | **{self.scorecard.growth_score.score:.0f}** | {self.scorecard.growth_score.grade} |",
            f"| Balance Sheet | 20% | **{self.scorecard.balance_sheet_score.score:.0f}** | {self.scorecard.balance_sheet_score.grade} |",
            f"| Cash Flow | 15% | **{self.scorecard.cash_flow_score.score:.0f}** | {self.scorecard.cash_flow_score.grade} |",
            f"| Valuation | 10% | **{self.scorecard.valuation_score.score:.0f}** | {self.scorecard.valuation_score.grade} |",
            f"| Shareholding | 10% | **{self.scorecard.shareholding_score.score:.0f}** | {self.scorecard.shareholding_score.grade} |",
            "",
            "### Executive Conclusion",
            f"{self.scorecard.conclusion}",
            "",
            "---",
            "",
            "## Key Growth & Quality Metrics",
            f"- **Revenue 3Y CAGR**: {self.growth.revenue_growth.cagr_3y*100:.1f}% (Trend: {self.growth.revenue_growth.trend})" if self.growth.revenue_growth.cagr_3y == self.growth.revenue_growth.cagr_3y else "- **Revenue 3Y CAGR**: N/A",
            f"- **Net Profit 3Y CAGR**: {self.growth.pat_growth.cagr_3y*100:.1f}%" if self.growth.pat_growth.cagr_3y == self.growth.pat_growth.cagr_3y else "- **Net Profit 3Y CAGR**: N/A",
            f"- **ROCE (Latest / 3Y Avg)**: {self.ratios.roce_latest:.1f}% / {self.ratios.roce_3y_avg:.1f}%" if self.ratios.roce_latest == self.ratios.roce_latest else "- **ROCE**: N/A",
            f"- **ROE (Latest / 3Y Avg)**: {self.ratios.roe_latest:.1f}% / {self.ratios.roe_3y_avg:.1f}%" if self.ratios.roe_latest == self.ratios.roe_latest else "- **ROE**: N/A",
            f"- **Debt-to-Equity**: {self.ratios.debt_to_equity_latest:.2f}x" if self.ratios.debt_to_equity_latest == self.ratios.debt_to_equity_latest else "- **Debt-to-Equity**: N/A",
            f"- **Cash Flow Conversion (CFO/PAT 3Y Avg)**: {self.ratios.cfo_to_pat_3y_avg:.2f}x" if self.ratios.cfo_to_pat_3y_avg == self.ratios.cfo_to_pat_3y_avg else "- **CFO/PAT**: N/A",
            f"- **P/E Multiple**: {self.valuation.pe_ratio:.1f}x (Peer Median: {self.valuation.peer_median_pe:.1f}x)" if self.valuation.pe_ratio == self.valuation.pe_ratio else "- **P/E Multiple**: N/A",
            "",
            "### Major Strengths",
        ]

        for s in self.scorecard.major_strengths:
            lines.append(f"- [STRENGTH] {s}")

        lines.append("")
        lines.append("### Major Weaknesses / Concerns")
        for w in self.scorecard.major_weaknesses:
            lines.append(f"- [WEAKNESS] {w}")

        lines.append("")
        lines.append(f"## Red Flag Audit ({self.red_flags.flag_count} Detected)")
        if self.red_flags.flags:
            lines.append("| Severity | Metric | Current Value | Historical / Target | Reason |")
            lines.append("|---|---|---|---|---|")
            for rf in self.red_flags.flags:
                sev_icon = "HIGH" if rf.severity == "HIGH" else "MEDIUM"
                lines.append(f"| {sev_icon} | {rf.metric} | {rf.current_value} | {rf.historical_value} | {rf.reason} |")
        else:
            lines.append("**Clean forensic profile**: No significant financial or governance red flags triggered.")

        return "\n".join(lines)


async def run_comprehensive_analysis(
    symbol: str,
    financial_type: str = "consolidated"
) -> ComprehensiveAnalysis:
    """Run end-to-end fundamental analysis pipeline for a company."""
    data = await collect_fundamental_data(symbol, financial_type)
    growth = analyze_growth(data.df_pl, data.compounding)
    ratios = analyze_ratios(data.df_pl, data.df_bs, data.df_cf, data.df_ratios)
    valuation = analyze_valuation(data.overview, data.df_peers, growth.eps_growth.latest_value)
    red_flags = detect_red_flags(growth, ratios, valuation, data.df_bs, data.df_cf, data.df_shareholding)
    scorecard = calculate_financial_health_score(growth, ratios, valuation, red_flags, data.df_shareholding)

    return ComprehensiveAnalysis(
        symbol=symbol.upper().strip(),
        data=data,
        growth=growth,
        ratios=ratios,
        valuation=valuation,
        red_flags=red_flags,
        scorecard=scorecard,
    )
