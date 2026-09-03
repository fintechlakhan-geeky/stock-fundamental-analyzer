"""
CLI Runner for Indian Stock Fundamental Analysis & Excel Report Generation.
Usage:
    python -m screener_mcp TCS
    python -m screener_mcp RELIANCE --output ./reports
    python -m screener_mcp INFY --mode standalone
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .analysis.financial_analysis import run_comprehensive_analysis
from .export.excel_report import export_fundamental_excel_report


async def run_cli_async(symbol: str, output_path: str = "", financial_type: str = "consolidated"):
    print(f"\n========================================================")
    print(f"  Fundamental Analysis Engine: {symbol.upper()}")
    print(f"  Mode: {financial_type.capitalize()} | Screener.in + NSE")
    print(f"========================================================\n")

    print(f"[*] Step 1/3: Collecting and normalizing financial statements...")
    analysis = await run_comprehensive_analysis(symbol, financial_type=financial_type)

    print(f"[*] Step 2/3: Computing ratios, CAGR, scoring, and red flags...")
    sc = analysis.scorecard
    print(f"\n>>> Overall Financial Health Score: {sc.overall_score}/100 ({sc.overall_grade})")
    print(f"    - Profitability:  {sc.profitability_score.score:.0f}/100 ({sc.profitability_score.grade})")
    print(f"    - Growth:         {sc.growth_score.score:.0f}/100 ({sc.growth_score.grade})")
    print(f"    - Balance Sheet:  {sc.balance_sheet_score.score:.0f}/100 ({sc.balance_sheet_score.grade})")
    print(f"    - Cash Flow:      {sc.cash_flow_score.score:.0f}/100 ({sc.cash_flow_score.grade})")
    print(f"    - Valuation:      {sc.valuation_score.score:.0f}/100 ({sc.valuation_score.grade})")
    print(f"    - Shareholding:   {sc.shareholding_score.score:.0f}/100 ({sc.shareholding_score.grade})")

    if analysis.red_flags.flag_count > 0:
        print(f"\n[!] Red Flags Detected ({analysis.red_flags.flag_count}):")
        for rf in analysis.red_flags.flags:
            sev = "HIGH" if rf.severity == "HIGH" else "MED"
            print(f"    [{sev}] {rf.metric}: {rf.reason}")
    else:
        print(f"\n[+] Red Flags: 0 detected (Clean forensic profile)")

    print(f"\n[*] Step 3/3: Rendering charts and building 14-sheet Excel report...")
    if not output_path:
        output_path = f"{symbol.upper()}_Fundamental_Analysis.xlsx"

    file_path = export_fundamental_excel_report(analysis, output_path)
    file_size_kb = os.path.getsize(file_path) / 1024
    print(f"\n[SUCCESS] Excel report successfully generated:")
    print(f"  -> File: {file_path} ({file_size_kb:.1f} KB)")
    print(f"  -> Sheets: Summary, Company Overview, P&L, Balance Sheet, Cash Flow,")
    print(f"             Quarterly Results, Ratios, Growth Analysis, Valuation,")
    print(f"             Shareholding, Peer Comparison, Red Flags, Charts, Raw Data.\n")

    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Indian Stock Fundamental Analysis & Excel Report Generator"
    )
    parser.add_argument(
        "symbol",
        help="Stock symbol or NSE/BSE ticker (e.g. TCS, RELIANCE, INFY, HDFCBANK)"
    )
    parser.add_argument(
        "-o", "--output",
        default="",
        help="Output Excel file path (default: <SYMBOL>_Fundamental_Analysis.xlsx)"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["consolidated", "standalone"],
        default="consolidated",
        help="Financial statement mode (default: consolidated)"
    )

    args = parser.parse_args()
    asyncio.run(run_cli_async(args.symbol, output_path=args.output, financial_type=args.mode))


if __name__ == "__main__":
    main()
