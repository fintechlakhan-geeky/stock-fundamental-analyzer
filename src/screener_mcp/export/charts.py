"""
Matplotlib Chart Generation Engine.
Generates 11 clean, publication-grade financial charts for Indian stock analysis.
Uses non-interactive Agg backend for server/headless execution.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.financial_analysis import ComprehensiveAnalysis


# Professional color palette
COLOR_NAVY = "#1F4E79"
COLOR_TEAL = "#008080"
COLOR_GREEN = "#2E7D32"
COLOR_RED = "#C62828"
COLOR_AMBER = "#F57C00"
COLOR_PURPLE = "#6A1B9A"
COLOR_GREY = "#616161"
COLOR_GRID = "#E0E0E0"


def _setup_figure(title: str, xlabel: str = "", ylabel: str = "", figsize: tuple = (8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID, zorder=1)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#212121")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=6, color="#424242")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=6, color="#424242")
    ax.tick_params(axis="both", which="major", labelsize=8, colors="#424242")
    for spine in ax.spines.values():
        spine.set_color("#BDBDBD")
    return fig, ax


def _filter_numeric_cols(df: Optional[pd.DataFrame], max_cols: int = 10) -> list[str]:
    if df is None or df.empty:
        return []
    cols = [c for c in df.columns if str(c).strip().upper() != "TTM"]
    return cols[-max_cols:] if len(cols) > max_cols else cols


def generate_revenue_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df = analysis.data.df_pl
    cols = _filter_numeric_cols(df)
    if "Sales" in df.index and cols:
        vals = df.loc["Sales", cols].values
        fig, ax = _setup_figure(f"{analysis.symbol} — Revenue (Sales) Trend", ylabel="Rs. Crore")
        bars = ax.bar(cols, vals, color=COLOR_NAVY, width=0.55, zorder=2)
        # Value labels
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h != 0:
                ax.text(bar.get_x() + bar.get_width()/2., h * 1.01, f"{h:,.0f}",
                        ha="center", va="bottom", fontsize=6.5, color="#212121", rotation=0)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_net_profit_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df = analysis.data.df_pl
    cols = _filter_numeric_cols(df)
    if "Net Profit" in df.index and cols:
        vals = df.loc["Net Profit", cols].values
        fig, ax = _setup_figure(f"{analysis.symbol} — Net Profit (PAT) Trend", ylabel="Rs. Crore")
        colors = [COLOR_GREEN if v >= 0 else COLOR_RED for v in vals]
        bars = ax.bar(cols, vals, color=colors, width=0.55, zorder=2)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h != 0:
                ax.text(bar.get_x() + bar.get_width()/2., h * 1.01 if h >= 0 else h * 0.95,
                        f"{h:,.0f}", ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=6.5, color="#212121")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_eps_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df = analysis.data.df_pl
    cols = _filter_numeric_cols(df)
    eps_label = next((k for k in df.index if "eps" in str(k).lower()), None)
    if eps_label and cols:
        vals = df.loc[eps_label, cols].values
        fig, ax = _setup_figure(f"{analysis.symbol} — Diluted EPS Trend", ylabel="Rs per share")
        ax.plot(cols, vals, marker="o", linewidth=2.2, color=COLOR_PURPLE, zorder=3)
        ax.fill_between(cols, vals, color=COLOR_PURPLE, alpha=0.1, zorder=2)
        for i, (col, val) in enumerate(zip(cols, vals)):
            if not np.isnan(val):
                ax.annotate(f"{val:.1f}", (col, val), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, fontweight="bold", color="#212121")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_opm_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df = analysis.data.df_pl
    cols = _filter_numeric_cols(df)
    opm_label = next((k for k in df.index if "opm" in str(k).lower()), None)
    if opm_label and cols:
        vals = df.loc[opm_label, cols].values
        fig, ax = _setup_figure(f"{analysis.symbol} — Operating Profit Margin (OPM %)", ylabel="Margin %")
        ax.plot(cols, vals, marker="s", linewidth=2.0, color=COLOR_TEAL, zorder=3)
        for col, val in zip(cols, vals):
            if not np.isnan(val):
                ax.annotate(f"{val:.1f}%", (col, val), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color="#212121")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_roe_roce_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df_r = analysis.ratios.df_all_ratios
    cols = _filter_numeric_cols(df_r)
    roe = df_r.loc["ROE (%)", cols].values if "ROE (%)" in df_r.index else []
    roce = df_r.loc["ROCE %", cols].values if "ROCE %" in df_r.index else []

    if len(cols) >= 2 and (len(roe) or len(roce)):
        fig, ax = _setup_figure(f"{analysis.symbol} — Return Ratios: ROE vs ROCE", ylabel="Percentage (%)")
        if len(roce):
            ax.plot(cols, roce, marker="o", label="ROCE (%)", color=COLOR_NAVY, linewidth=2.2, zorder=3)
        if len(roe):
            ax.plot(cols, roe, marker="^", label="ROE (%)", color=COLOR_AMBER, linewidth=2.0, linestyle="--", zorder=3)
        ax.legend(loc="best", fontsize=8)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_debt_trend_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df_bs = analysis.data.df_bs
    cols = _filter_numeric_cols(df_bs)
    borrowings = df_bs.loc["Borrowings", cols].values if "Borrowings" in df_bs.index else []
    eq = df_bs.loc["Equity Capital", cols].values if "Equity Capital" in df_bs.index else 0
    res = df_bs.loc["Reserves", cols].values if "Reserves" in df_bs.index else 0
    net_worth = eq + res if hasattr(eq, "__len__") and hasattr(res, "__len__") else []

    if len(cols) and len(borrowings):
        fig, ax = _setup_figure(f"{analysis.symbol} — Debt vs Net Worth Trend", ylabel="Rs. Crore")
        x = np.arange(len(cols))
        w = 0.35
        ax.bar(x - w/2, borrowings, width=w, label="Borrowings (Debt)", color=COLOR_RED, alpha=0.85, zorder=2)
        if len(net_worth):
            ax.bar(x + w/2, net_worth, width=w, label="Net Worth (Equity + Reserves)", color=COLOR_NAVY, alpha=0.85, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=35, ha="right")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_cash_flow_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df_cf = analysis.data.df_cf
    df_pl = analysis.data.df_pl
    cols = _filter_numeric_cols(df_cf)
    cfo = df_cf.loc["Cash from Operating Activity", cols].values if "Cash from Operating Activity" in df_cf.index else []
    pat = df_pl.loc["Net Profit", cols].values if "Net Profit" in df_pl.index and set(cols).issubset(df_pl.columns) else []

    if len(cols) and len(cfo):
        fig, ax = _setup_figure(f"{analysis.symbol} — Cash Flow Quality: CFO vs Net Profit", ylabel="Rs. Crore")
        x = np.arange(len(cols))
        w = 0.35
        ax.bar(x - w/2, cfo, width=w, label="Operating Cash Flow (CFO)", color=COLOR_TEAL, zorder=2)
        if len(pat):
            ax.bar(x + w/2, pat, width=w, label="Net Profit (PAT)", color=COLOR_NAVY, alpha=0.85, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=35, ha="right")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_fcf_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df_cf = analysis.data.df_cf
    cols = _filter_numeric_cols(df_cf)
    fcf = df_cf.loc["Free Cash Flow", cols].values if "Free Cash Flow" in df_cf.index else []

    if len(cols) and len(fcf):
        fig, ax = _setup_figure(f"{analysis.symbol} — Free Cash Flow (FCF) Trend", ylabel="Rs. Crore")
        colors = [COLOR_GREEN if v >= 0 else COLOR_RED for v in fcf]
        bars = ax.bar(cols, fcf, color=colors, width=0.55, zorder=2)
        ax.axhline(0, color="#757575", linewidth=0.8, linestyle="--")
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h != 0:
                ax.text(bar.get_x() + bar.get_width()/2., h * 1.02 if h >= 0 else h * 0.95,
                        f"{h:,.0f}", ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=6.5, color="#212121")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_quarterly_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df_q = analysis.data.df_quarters
    if df_q is not None and not df_q.empty:
        cols = list(df_q.columns)[-8:]  # last 8 quarters
        sales = df_q.loc["Sales", cols].values if "Sales" in df_q.index else []
        pat = df_q.loc["Net Profit", cols].values if "Net Profit" in df_q.index else []

        if len(cols) and len(sales):
            fig, ax1 = _setup_figure(f"{analysis.symbol} — Quarterly Sales & Profit Momentum", ylabel="Sales (Rs. Cr)")
            ax2 = ax1.twinx()
            ax1.bar(cols, sales, color=COLOR_NAVY, alpha=0.75, width=0.45, label="Quarterly Sales")
            if len(pat):
                ax2.plot(cols, pat, color=COLOR_AMBER, marker="o", linewidth=2.2, label="Net Profit (Rs. Cr)")
                ax2.set_ylabel("Net Profit (Rs. Cr)", color="#424242", fontsize=8)
            ax1.set_xticks(range(len(cols)))
            ax1.set_xticklabels(cols, rotation=35, ha="right")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7.5)
            plt.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)


def generate_shareholding_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df_sh = analysis.data.df_shareholding
    if df_sh is not None and not df_sh.empty:
        cols = list(df_sh.columns)[-8:]  # last 8 quarters
        promoters = df_sh.loc["Promoters", cols].values if "Promoters" in df_sh.index else []
        fiis = df_sh.loc["FIIs", cols].values if "FIIs" in df_sh.index else []
        diis = df_sh.loc["DIIs", cols].values if "DIIs" in df_sh.index else []
        public = df_sh.loc["Public", cols].values if "Public" in df_sh.index else []

        if len(cols) and len(promoters):
            fig, ax = _setup_figure(f"{analysis.symbol} — Shareholding Pattern Trend", ylabel="Percentage (%)")
            if len(promoters):
                ax.plot(cols, promoters, marker="o", label="Promoters %", color=COLOR_NAVY, linewidth=2.0)
            if len(fiis):
                ax.plot(cols, fiis, marker="s", label="FIIs %", color=COLOR_TEAL, linewidth=1.8)
            if len(diis):
                ax.plot(cols, diis, marker="^", label="DIIs %", color=COLOR_GREEN, linewidth=1.8)
            if len(public):
                ax.plot(cols, public, marker="x", label="Public %", color=COLOR_GREY, linewidth=1.5, linestyle=":")
            ax.legend(loc="best", fontsize=7.5)
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)


def generate_peer_comparison_chart(analysis: ComprehensiveAnalysis, out_path: str):
    df_p = analysis.valuation.df_peer_comparison
    if df_p is not None and not df_p.empty and "Name" in df_p.columns and "P/E" in df_p.columns:
        names = df_p["Name"].head(6).tolist()
        pe_vals = [float(v) if str(v).replace('.', '').isdigit() else 0 for v in df_p["P/E"].head(6)]

        if names and any(pe_vals):
            fig, ax = _setup_figure(f"{analysis.symbol} vs Peers — P/E Multiple", ylabel="Price to Earnings (P/E)")
            colors = [COLOR_AMBER if analysis.symbol.lower() in n.lower() else COLOR_NAVY for n in names]
            bars = ax.bar(names, pe_vals, color=colors, width=0.55, zorder=2)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., h * 1.01, f"{h:.1f}x",
                            ha="center", va="bottom", fontsize=7, color="#212121")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)


def generate_all_charts(analysis: ComprehensiveAnalysis, output_dir: Path) -> Dict[str, str]:
    """Generate all 11 financial charts and save PNGs into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sym = analysis.symbol

    charts_map = {
        "revenue_trend": str(output_dir / f"{sym}_revenue_trend.png"),
        "net_profit_trend": str(output_dir / f"{sym}_net_profit_trend.png"),
        "eps_trend": str(output_dir / f"{sym}_eps_trend.png"),
        "opm_trend": str(output_dir / f"{sym}_opm_trend.png"),
        "roe_roce_trend": str(output_dir / f"{sym}_roe_roce_trend.png"),
        "debt_trend": str(output_dir / f"{sym}_debt_trend.png"),
        "cash_flow_quality": str(output_dir / f"{sym}_cash_flow_quality.png"),
        "free_cash_flow": str(output_dir / f"{sym}_free_cash_flow.png"),
        "quarterly_trend": str(output_dir / f"{sym}_quarterly_trend.png"),
        "shareholding_trend": str(output_dir / f"{sym}_shareholding_trend.png"),
        "peer_comparison": str(output_dir / f"{sym}_peer_comparison.png"),
    }

    try:
        generate_revenue_chart(analysis, charts_map["revenue_trend"])
        generate_net_profit_chart(analysis, charts_map["net_profit_trend"])
        generate_eps_chart(analysis, charts_map["eps_trend"])
        generate_opm_chart(analysis, charts_map["opm_trend"])
        generate_roe_roce_chart(analysis, charts_map["roe_roce_trend"])
        generate_debt_trend_chart(analysis, charts_map["debt_trend"])
        generate_cash_flow_chart(analysis, charts_map["cash_flow_quality"])
        generate_fcf_chart(analysis, charts_map["free_cash_flow"])
        generate_quarterly_chart(analysis, charts_map["quarterly_trend"])
        generate_shareholding_chart(analysis, charts_map["shareholding_trend"])
        generate_peer_comparison_chart(analysis, charts_map["peer_comparison"])
    except Exception as exc:
        pass

    # Filter only successfully created files
    return {k: p for k, p in charts_map.items() if os.path.exists(p)}
