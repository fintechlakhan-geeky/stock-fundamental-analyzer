"""
Valuation Analysis Engine.
Evaluates current valuation multiples, historical context, and peer comparisons.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .data_collector import parse_numeric


@dataclass
class ValuationAnalysisResult:
    current_price: float
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    dividend_yield: float
    book_value: float
    face_value: float
    earnings_yield: float
    graham_number: float
    pe_vs_peer_median: float  # percentage premium (+) or discount (-)
    peer_median_pe: float
    peer_median_roce: float
    valuation_status: str  # "Undervalued", "Fairly Valued", "Overvalued", "Extremely Overvalued"
    df_valuation_summary: pd.DataFrame
    df_peer_comparison: pd.DataFrame


def analyze_valuation(
    overview: dict[str, Any],
    df_peers: Optional[pd.DataFrame] = None,
    latest_eps: float = np.nan,
) -> ValuationAnalysisResult:
    """Analyze current valuation multiples and peer positioning."""
    ratios = overview.get("key_ratios", {})
    price = parse_numeric(overview.get("current_price"))
    mcap = parse_numeric(ratios.get("Market Cap"))
    pe = parse_numeric(ratios.get("Stock P/E"))
    bv = parse_numeric(ratios.get("Book Value"))
    div_yield = parse_numeric(ratios.get("Dividend Yield"))
    face_val = parse_numeric(ratios.get("Face Value"))

    pb = (price / bv) if price and bv and not np.isnan(price) and not np.isnan(bv) and bv > 0 else np.nan
    earnings_yield = (100.0 / pe) if pe and not np.isnan(pe) and pe > 0 else np.nan

    # Graham Number = sqrt(22.5 * EPS * BV)
    graham_num = np.nan
    if not np.isnan(latest_eps) and not np.isnan(bv) and latest_eps > 0 and bv > 0:
        graham_num = math.sqrt(22.5 * latest_eps * bv)

    # Peer comparison valuation
    peer_pe_list = []
    peer_roce_list = []
    df_peer_clean = pd.DataFrame()

    if df_peers is not None and not df_peers.empty:
        df_peer_clean = df_peers.copy()
        if "P/E" in df_peer_clean.columns:
            peer_pes = df_peer_clean["P/E"].apply(parse_numeric).dropna()
            peer_pe_list = [p for p in peer_pes if p > 0]
        if "ROCE %" in df_peer_clean.columns:
            peer_roces = df_peer_clean["ROCE %"].apply(parse_numeric).dropna()
            peer_roce_list = list(peer_roces)

    peer_median_pe = float(np.median(peer_pe_list)) if peer_pe_list else np.nan
    peer_median_roce = float(np.median(peer_roce_list)) if peer_roce_list else np.nan

    pe_vs_peer = np.nan
    if not np.isnan(pe) and not np.isnan(peer_median_pe) and peer_median_pe > 0:
        pe_vs_peer = (pe - peer_median_pe) / peer_median_pe * 100

    # Assessment
    if np.isnan(pe):
        status = "Unknown / Loss Making"
    elif pe < 15:
        status = "Undervalued / Attractive"
    elif pe <= 30:
        status = "Fairly Valued"
    elif pe <= 55:
        status = "Premium / Growth Multiple"
    else:
        status = "High Multiple / Expensive"

    if not np.isnan(pe_vs_peer):
        if pe_vs_peer > 40:
            status += " (High Peer Premium)"
        elif pe_vs_peer < -30:
            status += " (Significant Peer Discount)"

    summary_data = {
        "Metric": [
            "Current Share Price (Rs)",
            "Market Capitalization (Cr)",
            "Price to Earnings (P/E)",
            "Price to Book (P/B)",
            "Dividend Yield (%)",
            "Book Value per Share (Rs)",
            "Earnings Yield (%)",
            "Graham Number (Rs)",
            "Peer Median P/E",
            "Peer Median ROCE (%)",
            "P/E Premium/Discount vs Peers (%)",
            "Valuation Assessment",
        ],
        "Value": [
            f"{price:,.2f}" if not np.isnan(price) else "N/A",
            f"{mcap:,.1f}" if not np.isnan(mcap) else "N/A",
            f"{pe:.2f}x" if not np.isnan(pe) else "N/A",
            f"{pb:.2f}x" if not np.isnan(pb) else "N/A",
            f"{div_yield:.2f}%" if not np.isnan(div_yield) else "N/A",
            f"{bv:,.2f}" if not np.isnan(bv) else "N/A",
            f"{earnings_yield:.2f}%" if not np.isnan(earnings_yield) else "N/A",
            f"{graham_num:,.2f}" if not np.isnan(graham_num) else "N/A",
            f"{peer_median_pe:.2f}x" if not np.isnan(peer_median_pe) else "N/A",
            f"{peer_median_roce:.2f}%" if not np.isnan(peer_median_roce) else "N/A",
            f"{pe_vs_peer:+.1f}%" if not np.isnan(pe_vs_peer) else "N/A",
            status,
        ],
    }

    df_summary = pd.DataFrame(summary_data).set_index("Metric")

    return ValuationAnalysisResult(
        current_price=price,
        market_cap=mcap,
        pe_ratio=pe,
        pb_ratio=pb,
        dividend_yield=div_yield,
        book_value=bv,
        face_value=face_val,
        earnings_yield=earnings_yield,
        graham_number=graham_num,
        pe_vs_peer_median=pe_vs_peer,
        peer_median_pe=peer_median_pe,
        peer_median_roce=peer_median_roce,
        valuation_status=status,
        df_valuation_summary=df_summary,
        df_peer_comparison=df_peer_clean,
    )
