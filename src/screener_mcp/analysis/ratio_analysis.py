"""
Ratio Analysis Engine.
Calculates and normalizes Profitability, Solvency, Efficiency, and Cash Flow Quality ratios.
Never invents data — uses NaN when inputs are missing.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from .growth_analysis import get_series_by_label


@dataclass
class RatioAnalysisResult:
    # Latest snapshot values
    roe_latest: float
    roce_latest: float
    roa_latest: float
    opm_latest: float
    npm_latest: float
    debt_to_equity_latest: float
    interest_coverage_latest: float
    cfo_to_pat_latest: float
    debtor_days_latest: float
    working_capital_days_latest: float
    cash_conversion_cycle_latest: float
    asset_turnover_latest: float
    fcf_latest: float
    # 3-year averages
    roe_3y_avg: float
    roce_3y_avg: float
    opm_3y_avg: float
    npm_3y_avg: float
    cfo_to_pat_3y_avg: float
    interest_coverage_3y_avg: float
    # Historical DataFrames
    df_profitability: pd.DataFrame
    df_solvency: pd.DataFrame
    df_efficiency: pd.DataFrame
    df_cash_flow_quality: pd.DataFrame
    df_all_ratios: pd.DataFrame


def _get_clean_array(series: Optional[pd.Series], max_periods: int = 10) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    cols = [c for c in series.index if str(c).strip().upper() != "TTM"]
    s = series[cols].dropna()
    return s.iloc[-max_periods:] if len(s) > max_periods else s


def analyze_ratios(
    df_pl: pd.DataFrame,
    df_bs: pd.DataFrame,
    df_cf: pd.DataFrame,
    df_ratios: pd.DataFrame,
) -> RatioAnalysisResult:
    """Compute comprehensive financial ratios across historical periods."""
    sales = get_series_by_label(df_pl, ["Sales", "Revenue"])
    op = get_series_by_label(df_pl, ["Operating Profit", "EBITDA"])
    pat = get_series_by_label(df_pl, ["Net Profit", "PAT"])
    interest = get_series_by_label(df_pl, ["Interest"])
    equity = get_series_by_label(df_bs, ["Equity Capital"])
    reserves = get_series_by_label(df_bs, ["Reserves"])
    borrowings = get_series_by_label(df_bs, ["Borrowings", "Total Debt"])
    total_assets = get_series_by_label(df_bs, ["Total Assets", "Total Liabilities"])
    cfo = get_series_by_label(df_cf, ["Cash from Operating Activity", "Operating Cash Flow"])
    fcf = get_series_by_label(df_cf, ["Free Cash Flow"])

    # Extract historical ratios if provided in Screener's ratios table
    roce_hist = get_series_by_label(df_ratios, ["ROCE"])
    debtor_days_hist = get_series_by_label(df_ratios, ["Debtor Days"])
    inv_days_hist = get_series_by_label(df_ratios, ["Inventory Days"])
    payable_days_hist = get_series_by_label(df_ratios, ["Days Payable"])
    ccc_hist = get_series_by_label(df_ratios, ["Cash Conversion Cycle"])
    wc_days_hist = get_series_by_label(df_ratios, ["Working Capital Days"])

    # Align common years
    common_years = []
    for df in [df_pl, df_bs, df_cf]:
        if df is not None and not df.empty:
            years = [c for c in df.columns if str(c).strip().upper() != "TTM"]
            if not common_years:
                common_years = years
            else:
                common_years = [y for y in common_years if y in years]

    if not common_years and df_pl is not None and not df_pl.empty:
        common_years = [c for c in df_pl.columns if str(c).strip().upper() != "TTM"]

    ratios_dict = {}

    for y in common_years:
        s_val = sales.get(y, np.nan) if sales is not None else np.nan
        op_val = op.get(y, np.nan) if op is not None else np.nan
        pat_val = pat.get(y, np.nan) if pat is not None else np.nan
        int_val = interest.get(y, np.nan) if interest is not None else np.nan
        eq_val = equity.get(y, np.nan) if equity is not None else np.nan
        res_val = reserves.get(y, np.nan) if reserves is not None else np.nan
        bor_val = borrowings.get(y, np.nan) if borrowings is not None else np.nan
        assets_val = total_assets.get(y, np.nan) if total_assets is not None else np.nan
        cfo_val = cfo.get(y, np.nan) if cfo is not None else np.nan
        fcf_val = fcf.get(y, np.nan) if fcf is not None else np.nan

        net_worth = (eq_val + res_val) if not np.isnan(eq_val) and not np.isnan(res_val) else np.nan

        # Profitability
        opm = (op_val / s_val * 100) if s_val and not np.isnan(s_val) and s_val > 0 else np.nan
        npm = (pat_val / s_val * 100) if s_val and not np.isnan(s_val) and s_val > 0 else np.nan
        roe = (pat_val / net_worth * 100) if net_worth and not np.isnan(net_worth) and net_worth > 0 else np.nan
        roa = (pat_val / assets_val * 100) if assets_val and not np.isnan(assets_val) and assets_val > 0 else np.nan

        # Solvency
        de = (bor_val / net_worth) if net_worth and not np.isnan(net_worth) and net_worth > 0 else (0.0 if bor_val == 0 else np.nan)
        ic = (op_val / int_val) if int_val and not np.isnan(int_val) and int_val > 0 else (999.0 if int_val == 0 and op_val > 0 else np.nan)

        # Cash flow quality
        cfo_pat = (cfo_val / pat_val) if pat_val and not np.isnan(pat_val) and pat_val > 0 else np.nan
        asset_turnover = (s_val / assets_val) if assets_val and not np.isnan(assets_val) and assets_val > 0 else np.nan

        ratios_dict[y] = {
            "OPM (%)": opm,
            "NPM (%)": npm,
            "ROE (%)": roe,
            "ROA (%)": roa,
            "Debt to Equity": de,
            "Interest Coverage": ic,
            "CFO / Net Profit": cfo_pat,
            "Asset Turnover": asset_turnover,
            "Free Cash Flow": fcf_val,
        }

    df_calculated = pd.DataFrame(ratios_dict)

    # Merge Screener historical ratios if available
    if df_ratios is not None and not df_ratios.empty:
        for idx in df_ratios.index:
            s_name = str(idx).strip()
            row_series = df_ratios.loc[idx]
            for col in df_calculated.columns:
                if col in row_series.index and not pd.isna(row_series[col]):
                    df_calculated.loc[s_name, col] = row_series[col]

    # Compute snapshot latest and 3Y averages
    def _latest(row_name: str, fallback_series: Optional[pd.Series] = None) -> float:
        if row_name in df_calculated.index:
            s = df_calculated.loc[row_name].dropna()
            if not s.empty:
                return float(s.iloc[-1])
        if fallback_series is not None and not fallback_series.empty:
            s = fallback_series.dropna()
            if not s.empty:
                return float(s.iloc[-1])
        return np.nan

    def _avg3(row_name: str, fallback_series: Optional[pd.Series] = None) -> float:
        if row_name in df_calculated.index:
            s = df_calculated.loc[row_name].dropna()
            if len(s) >= 3:
                return float(s.iloc[-3:].mean())
            elif not s.empty:
                return float(s.mean())
        if fallback_series is not None and not fallback_series.empty:
            s = fallback_series.dropna()
            if len(s) >= 3:
                return float(s.iloc[-3:].mean())
            elif not s.empty:
                return float(s.mean())
        return np.nan

    roe_latest = _latest("ROE (%)")
    roce_latest = _latest("ROCE %", roce_hist)
    roa_latest = _latest("ROA (%)")
    opm_latest = _latest("OPM (%)")
    npm_latest = _latest("NPM (%)")
    de_latest = _latest("Debt to Equity")
    ic_latest = _latest("Interest Coverage")
    cfo_pat_latest = _latest("CFO / Net Profit")
    fcf_latest = _latest("Free Cash Flow", fcf)
    debtor_days_latest = _latest("Debtor Days", debtor_days_hist)
    wc_days_latest = _latest("Working Capital Days", wc_days_hist)
    ccc_latest = _latest("Cash Conversion Cycle", ccc_hist)
    asset_turnover_latest = _latest("Asset Turnover")

    roe_3y_avg = _avg3("ROE (%)")
    roce_3y_avg = _avg3("ROCE %", roce_hist)
    opm_3y_avg = _avg3("OPM (%)")
    npm_3y_avg = _avg3("NPM (%)")
    cfo_pat_3y_avg = _avg3("CFO / Net Profit")
    ic_3y_avg = _avg3("Interest Coverage")

    # Grouped sub-tables
    prof_idx = [k for k in ["OPM (%)", "NPM (%)", "ROE (%)", "ROCE %", "ROA (%)"] if k in df_calculated.index]
    solv_idx = [k for k in ["Debt to Equity", "Interest Coverage"] if k in df_calculated.index]
    eff_idx = [k for k in ["Debtor Days", "Inventory Days", "Days Payable", "Cash Conversion Cycle", "Working Capital Days", "Asset Turnover"] if k in df_calculated.index]
    cf_idx = [k for k in ["CFO / Net Profit", "Free Cash Flow"] if k in df_calculated.index]

    df_prof = df_calculated.loc[prof_idx] if prof_idx else pd.DataFrame()
    df_solv = df_calculated.loc[solv_idx] if solv_idx else pd.DataFrame()
    df_eff = df_calculated.loc[eff_idx] if eff_idx else pd.DataFrame()
    df_cf_q = df_calculated.loc[cf_idx] if cf_idx else pd.DataFrame()

    return RatioAnalysisResult(
        roe_latest=roe_latest,
        roce_latest=roce_latest,
        roa_latest=roa_latest,
        opm_latest=opm_latest,
        npm_latest=npm_latest,
        debt_to_equity_latest=de_latest,
        interest_coverage_latest=ic_latest,
        cfo_to_pat_latest=cfo_pat_latest,
        debtor_days_latest=debtor_days_latest,
        working_capital_days_latest=wc_days_latest,
        cash_conversion_cycle_latest=ccc_latest,
        asset_turnover_latest=asset_turnover_latest,
        fcf_latest=fcf_latest,
        roe_3y_avg=roe_3y_avg,
        roce_3y_avg=roce_3y_avg,
        opm_3y_avg=opm_3y_avg,
        npm_3y_avg=npm_3y_avg,
        cfo_to_pat_3y_avg=cfo_pat_3y_avg,
        interest_coverage_3y_avg=ic_3y_avg,
        df_profitability=df_prof,
        df_solvency=df_solv,
        df_efficiency=df_eff,
        df_cash_flow_quality=df_cf_q,
        df_all_ratios=df_calculated,
    )
