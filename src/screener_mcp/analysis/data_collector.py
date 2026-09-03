"""
Unified Fundamental Data Collector.
Gathers complete, normalized financial statements from Screener.in and NSE India.
Normalizes data into structured Pandas DataFrames without inventing missing values.
Supports dynamic detection and extraction of all nested/sub-row schedules and shareholder breakdowns.
"""

import asyncio
import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from ..client import get_client
from ..core.nse_client import get_nse_client
from ..parsers.company import parse_full_page

logger = logging.getLogger(__name__)


def clean_label(label: str) -> str:
    """Normalize row label by stripping trailing +, raw whitespace, and non-breaking spaces."""
    if not label:
        return ""
    s = label.replace("\xa0", " ").strip()
    s = s.rstrip("+").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_numeric(val: Any) -> float:
    """Convert currency/percentage/string into float, NaN if invalid or missing."""
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(val) else np.nan

    s = str(val).strip().replace("\xa0", " ")
    if not s or s in {"-", "—", "N/A", "NA", "None", ""}:
        return np.nan

    s = s.replace(",", "").replace("%", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    try:
        return float(s)
    except ValueError:
        return np.nan


def table_to_dataframe(table_dict: dict[str, Any], index_col_name: str = "Metric") -> pd.DataFrame:
    """Convert yearly table {years: [...], rows: [{label, values}]} to flat top-level DataFrame."""
    years = table_dict.get("years", [])
    rows = table_dict.get("rows", [])
    if not years or not rows:
        return pd.DataFrame()

    cleaned_years = [re.sub(r"\s+", " ", y).strip() for y in years]
    data = {}
    row_labels = []

    for r in rows:
        raw_label = r.get("label", "") or r.get("category", "")
        norm_label = clean_label(raw_label)
        if not norm_label or norm_label in row_labels:
            norm_label = f"{norm_label} (2)" if norm_label in row_labels else norm_label

        row_labels.append(norm_label)
        raw_values = r.get("values", [])
        parsed_vals = []
        for i in range(len(cleaned_years)):
            v = raw_values[i] if i < len(raw_values) else None
            parsed_vals.append(parse_numeric(v))
        data[norm_label] = parsed_vals

    df = pd.DataFrame(data, index=cleaned_years).T
    df.index.name = index_col_name
    return df


def build_hierarchical_dataframe(
    table_dict: dict[str, Any],
    section_schedules: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a hierarchical 2-level DataFrame:
    Columns: ['Parent Category', 'Sub-category', Year1, Year2, ...]
    Preserves parent totals separately from child sub-rows.
    Child rows are never double-counted into parent rows.
    """
    years = table_dict.get("years", [])
    rows = table_dict.get("rows", [])
    if not years or not rows:
        return pd.DataFrame()

    cleaned_years = [re.sub(r"\s+", " ", y).strip() for y in years]
    records = []

    for r in rows:
        raw_label = r.get("label", "") or r.get("category", "")
        parent_clean = raw_label.rstrip(" +").strip()
        norm_label = clean_label(raw_label)
        raw_values = r.get("values", [])
        parsed_parent_vals = [
            parse_numeric(raw_values[i]) if i < len(raw_values) else np.nan
            for i in range(len(cleaned_years))
        ]
        parent_row_dict = dict(zip(cleaned_years, parsed_parent_vals))

        # Check if schedule exists for this parent
        sched_data = None
        for k, v in section_schedules.items():
            if k.lower() == parent_clean.lower() or k.lower() == norm_label.lower():
                sched_data = v
                break

        if sched_data and isinstance(sched_data, dict):
            # 1. Total / Parent row
            rec_parent = {
                "Parent Category": parent_clean,
                "Sub-category": f"Total {parent_clean}",
            }
            rec_parent.update(parent_row_dict)
            records.append(rec_parent)

            # 2. Child sub-rows
            for sub_name, sub_years in sched_data.items():
                if sub_name == "setAttributes" or not isinstance(sub_years, dict):
                    continue
                rec_sub = {
                    "Parent Category": parent_clean,
                    "Sub-category": sub_name.strip(),
                }
                for y in cleaned_years:
                    val = sub_years.get(y)
                    if val is None:
                        val = sub_years.get(y.replace(" ", ""))
                    rec_sub[y] = parse_numeric(val) if val is not None else np.nan
                records.append(rec_sub)
        else:
            # Leaf row without sub-components
            rec_leaf = {
                "Parent Category": parent_clean,
                "Sub-category": "-",
            }
            rec_leaf.update(parent_row_dict)
            records.append(rec_leaf)

    df_hier = pd.DataFrame(records)
    col_order = ["Parent Category", "Sub-category"] + [y for y in cleaned_years if y in df_hier.columns]
    return df_hier[col_order]


def build_hierarchical_shareholding(
    sh_dict: dict[str, Any],
    investors_dict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a hierarchical 2-level DataFrame for Shareholding:
    Columns: ['Parent Category', 'Sub-category', Quarter1, Quarter2, ...]
    Includes Total classification percentage and individual holders (> 0.05% holding).
    """
    quarters = sh_dict.get("quarters", [])
    rows = sh_dict.get("rows", [])
    if not quarters or not rows:
        return pd.DataFrame()

    cleaned_quarters = [re.sub(r"\s+", " ", q).strip() for q in quarters]
    records = []

    cat_map = {
        "promoter": "promoters",
        "fii": "foreign_institutions",
        "dii": "domestic_institutions",
        "government": "government",
        "public": "public",
    }

    for r in rows:
        cat_name = r.get("category", "")
        clean_cat = cat_name.rstrip(" +").strip()
        raw_vals = r.get("values", [])
        parsed_vals = [
            parse_numeric(raw_vals[i]) if i < len(raw_vals) else np.nan
            for i in range(len(cleaned_quarters))
        ]
        cat_row_dict = dict(zip(cleaned_quarters, parsed_vals))

        rec_parent = {
            "Parent Category": clean_cat,
            "Sub-category": f"Total {clean_cat} %",
        }
        rec_parent.update(cat_row_dict)
        records.append(rec_parent)

        api_key = None
        for k_sub, v_key in cat_map.items():
            if k_sub in clean_cat.lower():
                api_key = v_key
                break

        holders_data = investors_dict.get(api_key, {}) if api_key else {}
        if holders_data and isinstance(holders_data, dict):
            valid_holders = []
            for h_name, h_quarters in holders_data.items():
                if h_name == "setAttributes" or not isinstance(h_quarters, dict):
                    continue
                vals_h = [parse_numeric(h_quarters.get(q)) for q in cleaned_quarters if h_quarters.get(q) is not None]
                max_h = max([v for v in vals_h if not np.isnan(v)], default=0)
                if max_h >= 0.05:
                    valid_holders.append((h_name, h_quarters, max_h))

            valid_holders.sort(key=lambda x: x[2], reverse=True)
            for h_name, h_quarters, _ in valid_holders[:20]:
                rec_sub = {
                    "Parent Category": clean_cat,
                    "Sub-category": h_name.strip(),
                }
                for q in cleaned_quarters:
                    val = h_quarters.get(q)
                    rec_sub[q] = parse_numeric(val) if val is not None else np.nan
                records.append(rec_sub)

    df_hier = pd.DataFrame(records)
    col_order = ["Parent Category", "Sub-category"] + [q for q in cleaned_quarters if q in df_hier.columns]
    return df_hier[col_order]


@dataclass
class CollectedFinancialData:
    symbol: str
    overview: dict[str, Any]
    df_pl: pd.DataFrame
    df_bs: pd.DataFrame
    df_cf: pd.DataFrame
    df_quarters: pd.DataFrame
    df_ratios: pd.DataFrame
    df_shareholding: pd.DataFrame
    df_peers: pd.DataFrame
    compounding: dict[str, dict[str, str]]
    df_pl_hierarchical: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_bs_hierarchical: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_cf_hierarchical: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_quarters_hierarchical: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_shareholding_hierarchical: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_schedules: dict[str, Any] = field(default_factory=dict)
    announcements: list[dict[str, Any]] = field(default_factory=list)
    annual_reports: list[dict[str, Any]] = field(default_factory=list)
    retrieval_date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    financial_type: str = "consolidated"


async def fetch_peers_table(warehouse_id: str) -> pd.DataFrame:
    """Fetch live peer comparison table from Screener API."""
    if not warehouse_id:
        return pd.DataFrame()
    try:
        client = await get_client()
        html = await client.get_html(f"/api/company/{warehouse_id}/peers/")
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if not table:
            return pd.DataFrame()

        headers = [clean_label(th.get_text()) for th in table.find_all("th")]
        std_headers = []
        for h in headers:
            h_clean = re.sub(r"\s+", " ", h).strip()
            if "Name" in h_clean:
                std_headers.append("Name")
            elif "CMP" in h_clean:
                std_headers.append("CMP")
            elif "P/E" in h_clean:
                std_headers.append("P/E")
            elif "Mar Cap" in h_clean or "Market Cap" in h_clean:
                std_headers.append("Market Cap")
            elif "Div Yld" in h_clean:
                std_headers.append("Div Yield %")
            elif "NP Qtr" in h_clean:
                std_headers.append("Net Profit Qtr")
            elif "Qtr Profit Var" in h_clean:
                std_headers.append("Qtr Profit Var %")
            elif "Sales Qtr" in h_clean:
                std_headers.append("Sales Qtr")
            elif "Qtr Sales Var" in h_clean:
                std_headers.append("Qtr Sales Var %")
            elif "ROCE" in h_clean:
                std_headers.append("ROCE %")
            else:
                std_headers.append(h_clean)

        tbody = table.find("tbody")
        if not tbody:
            return pd.DataFrame()

        rows_data = []
        for tr in tbody.find_all("tr"):
            cells = [clean_label(td.get_text()) for td in tr.find_all("td")]
            if len(cells) == len(std_headers):
                rows_data.append(cells)

        if not rows_data:
            return pd.DataFrame()

        df_peers = pd.DataFrame(rows_data, columns=std_headers)
        for col in df_peers.columns:
            if col != "Name":
                df_peers[col] = df_peers[col].apply(parse_numeric)

        return df_peers
    except Exception as exc:
        logger.debug("Peers fetch error: %s", exc)
        return pd.DataFrame()


async def collect_fundamental_data(
    symbol: str,
    financial_type: str = "consolidated"
) -> CollectedFinancialData:
    """
    Ingest all available fundamental statements, schedules, and ratios for a stock.
    Detects and captures all nested schedules dynamically.
    """
    client = await get_client()
    symbol_upper = symbol.upper().strip()
    path = f"/company/{symbol_upper}/"
    if financial_type == "consolidated":
        path += "consolidated/"

    html = await client.get_html(path)
    soup = BeautifulSoup(html, "lxml")
    parsed = parse_full_page(html)

    overview = parsed.get("overview", {})
    company_id = overview.get("company_id", "")
    pl_dict = parsed.get("profit_loss", {})
    bs_dict = parsed.get("balance_sheet", {})
    cf_dict = parsed.get("cash_flow", {})
    qr_dict = parsed.get("quarterly_results", {})
    rh_dict = parsed.get("ratios_history", {})
    sh_dict = parsed.get("shareholding", {})
    compounding = parsed.get("compounding", {})

    # Convert top-level statements to DataFrames (for ratios & scoring)
    df_pl = table_to_dataframe(pl_dict)
    df_bs = table_to_dataframe(bs_dict)
    df_cf = table_to_dataframe(cf_dict)
    df_quarters = table_to_dataframe(qr_dict)
    df_ratios = table_to_dataframe(rh_dict)

    sh_quarters = sh_dict.get("quarters", [])
    sh_rows = sh_dict.get("rows", [])
    if sh_quarters and sh_rows:
        df_sh = table_to_dataframe({"years": sh_quarters, "rows": [{"label": r.get("category", ""), "values": r.get("values", [])} for r in sh_rows]})
    else:
        df_sh = pd.DataFrame()

    # ─── Dynamic Detection & Parallel Fetching of Sub-row Schedules ───────────
    schedules_to_fetch = []
    for btn in soup.find_all(attrs={"onclick": re.compile(r"showSchedule")}):
        onclick = btn.get("onclick", "")
        m = re.search(r"showSchedule\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]", onclick)
        if m:
            schedules_to_fetch.append((m.group(1), m.group(2)))

    shareholders_to_fetch = []
    for btn in soup.find_all(attrs={"onclick": re.compile(r"showShareholders")}):
        onclick = btn.get("onclick", "")
        m = re.search(r"showShareholders\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]", onclick)
        if m:
            shareholders_to_fetch.append((m.group(1), m.group(2)))

    async def _fetch_single_schedule(parent: str, section: str):
        if not company_id:
            return (section, parent, {})
        params = {"parent": parent, "section": section}
        if financial_type == "consolidated":
            params["consolidated"] = ""
        try:
            resp = await client.get_html(f"/api/company/{company_id}/schedules/", params=params)
            data = json.loads(resp)
            return (section, parent, data)
        except Exception:
            return (section, parent, {})

    async def _fetch_single_shareholder(classification: str, period: str):
        if not company_id:
            return (classification, period, {})
        try:
            resp = await client.get_html(f"/api/3/{company_id}/investors/{classification}/{period}/")
            data = json.loads(resp)
            return (classification, period, data)
        except Exception:
            return (classification, period, {})

    sched_tasks = [_fetch_single_schedule(p, s) for p, s in schedules_to_fetch]
    sh_tasks = [_fetch_single_shareholder(c, p) for c, p in shareholders_to_fetch if p == "quarterly"]

    fetched_scheds, fetched_sh = await asyncio.gather(
        asyncio.gather(*sched_tasks),
        asyncio.gather(*sh_tasks),
    )

    schedules_by_section = {}
    for sec, parent, data in fetched_scheds:
        schedules_by_section.setdefault(sec, {})[parent] = data

    investors_by_cat = {}
    for classification, period, data in fetched_sh:
        investors_by_cat[classification] = data

    # Build hierarchical statement DataFrames
    df_pl_hier = build_hierarchical_dataframe(pl_dict, schedules_by_section.get("profit-loss", {}))
    df_bs_hier = build_hierarchical_dataframe(bs_dict, schedules_by_section.get("balance-sheet", {}))
    df_cf_hier = build_hierarchical_dataframe(cf_dict, schedules_by_section.get("cash-flow", {}))
    df_quarters_hier = build_hierarchical_dataframe(qr_dict, schedules_by_section.get("quarters", {}))
    df_sh_hier = build_hierarchical_shareholding(sh_dict, investors_by_cat)

    # Fetch peers table
    warehouse_id = overview.get("warehouse_id", "")
    df_peers = await fetch_peers_table(warehouse_id)

    # NSE announcements & filings
    announcements = []
    annual_reports = []
    try:
        nse = await get_nse_client()
        announcements_task = nse.get_announcements(symbol_upper)
        reports_task = nse.get_annual_reports(symbol_upper)
        announcements, annual_reports = await asyncio.gather(
            announcements_task, reports_task, return_exceptions=True
        )
        if isinstance(announcements, Exception):
            announcements = []
        if isinstance(annual_reports, Exception):
            annual_reports = []
    except Exception as exc:
        logger.debug("NSE fetch skipped or failed: %s", exc)

    return CollectedFinancialData(
        symbol=symbol_upper,
        overview=overview,
        df_pl=df_pl,
        df_bs=df_bs,
        df_cf=df_cf,
        df_quarters=df_quarters,
        df_ratios=df_ratios,
        df_shareholding=df_sh,
        df_peers=df_peers,
        compounding=compounding,
        df_pl_hierarchical=df_pl_hier,
        df_bs_hierarchical=df_bs_hier,
        df_cf_hierarchical=df_cf_hier,
        df_quarters_hierarchical=df_quarters_hier,
        df_shareholding_hierarchical=df_sh_hier,
        raw_schedules=schedules_by_section,
        announcements=announcements if isinstance(announcements, list) else [],
        annual_reports=annual_reports if isinstance(annual_reports, list) else [],
        financial_type=financial_type,
    )
