"""
Excel Report Generation Engine (OpenPyXL).
Builds a publication-grade 14-worksheet workbook: <COMPANY>_Fundamental_Analysis.xlsx
with professional formatting, freeze panes, conditional formatting, and embedded charts.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from ..analysis.financial_analysis import ComprehensiveAnalysis
from .charts import generate_all_charts


# ─── Professional Color Palette ───────────────────────────────────────────────
FILL_HEADER = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
FONT_HEADER = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")

FILL_SECTION = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
FONT_SECTION = Font(name="Segoe UI", size=11, bold=True, color="1F4E79")

FILL_CARD = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
FONT_CARD_TITLE = Font(name="Segoe UI", size=9, bold=True, color="595959")
FONT_CARD_VALUE = Font(name="Segoe UI", size=16, bold=True, color="1F4E79")

FILL_ZEBRA = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")
FILL_WHITE = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

FILL_SEV_HIGH = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
FONT_SEV_HIGH = Font(name="Segoe UI", size=9, bold=True, color="721C24")

FILL_SEV_MED = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
FONT_SEV_MED = Font(name="Segoe UI", size=9, bold=True, color="856404")

FONT_DATA = Font(name="Segoe UI", size=9, color="212121")
FONT_BOLD_DATA = Font(name="Segoe UI", size=9, bold=True, color="212121")
FILL_PARENT = PatternFill(start_color="E9EEF4", end_color="E9EEF4", fill_type="solid")
FONT_PARENT = Font(name="Segoe UI", size=9, bold=True, color="1F4E79")

BORDER_THIN = Border(
    left=Side(style="thin", color="D3D3D3"),
    right=Side(style="thin", color="D3D3D3"),
    top=Side(style="thin", color="D3D3D3"),
    bottom=Side(style="thin", color="D3D3D3"),
)


def _autofit_columns(ws, min_width: int = 12, max_width: int = 50):
    """Adjust column widths dynamically."""
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            if len(val_str) > max_len and not cell.coordinate in ws.merged_cells:
                max_len = len(val_str)
        ws.column_dimensions[col_letter].width = max(min(max_len + 3, max_width), min_width)


def _format_cell_value(cell, val: Any, label: str = ""):
    """Apply numeric and percentage formats intelligently."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        cell.value = "-"
        cell.alignment = Alignment(horizontal="right")
        return

    if isinstance(val, (int, float)):
        cell.value = float(val)
        abs_val = abs(val)
        label_lower = str(label).lower()
        if "%" in label_lower:
            cell.number_format = "0.0%" if abs_val <= 1.0 else "0.0"
        elif "cagr" in label_lower or "yoy" in label_lower:
            cell.number_format = "0.0%" if abs_val <= 1.0 else "0.0"
        elif abs_val >= 1000:
            cell.number_format = "#,##0.0"
        elif abs_val >= 10:
            cell.number_format = "#,##0.00"
        else:
            cell.number_format = "0.00"
        cell.alignment = Alignment(horizontal="right")
    else:
        cell.value = str(val)
        cell.alignment = Alignment(horizontal="left")


def _write_dataframe_table(
    ws,
    df,
    start_row: int = 2,
    start_col: int = 1,
    title: str = "",
    include_index: bool = True
) -> int:
    """Write DataFrame to worksheet with headers, borders, and number formatting."""
    cur_row = start_row

    if title:
        cell_title = ws.cell(row=cur_row, column=start_col, value=title)
        cell_title.font = FONT_SECTION
        ws.row_dimensions[cur_row].height = 24
        cur_row += 1

    if df is None or df.empty:
        ws.cell(row=cur_row, column=start_col, value="No data available for this section.")
        return cur_row + 2

    # Headers
    col_offset = 0
    if include_index:
        idx_header = ws.cell(row=cur_row, column=start_col, value=df.index.name or "Metric")
        idx_header.fill = FILL_HEADER
        idx_header.font = FONT_HEADER
        idx_header.alignment = Alignment(horizontal="left", vertical="center")
        col_offset = 1

    for c_idx, col_name in enumerate(df.columns):
        header_cell = ws.cell(row=cur_row, column=start_col + col_offset + c_idx, value=str(col_name))
        header_cell.fill = FILL_HEADER
        header_cell.font = FONT_HEADER
        header_cell.alignment = Alignment(horizontal="right", vertical="center")

    ws.row_dimensions[cur_row].height = 22
    cur_row += 1

    # Data Rows
    for r_idx, (idx_val, row) in enumerate(df.iterrows()):
        fill = FILL_ZEBRA if r_idx % 2 == 1 else FILL_WHITE
        ws.row_dimensions[cur_row].height = 19

        if include_index:
            idx_cell = ws.cell(row=cur_row, column=start_col, value=str(idx_val))
            idx_cell.font = FONT_BOLD_DATA
            idx_cell.fill = fill
            idx_cell.border = BORDER_THIN
            idx_cell.alignment = Alignment(horizontal="left", vertical="center")

        for c_idx, col_name in enumerate(df.columns):
            val = row[col_name]
            cell = ws.cell(row=cur_row, column=start_col + col_offset + c_idx)
            cell.fill = fill
            cell.border = BORDER_THIN
            cell.font = FONT_DATA
            _format_cell_value(cell, val, label=f"{idx_val} {col_name}")

        cur_row += 1

    return cur_row + 1


def _write_hierarchical_statement_table(
    ws,
    df_hier: pd.DataFrame,
    start_row: int = 2,
    title: str = "",
) -> int:
    """
    Write hierarchical 2-level financial statement table:
    Column A: Parent Category
    Column B: Sub-category
    Columns C+: Historical Years / Quarters
    """
    cur_row = start_row

    if title:
        cell_title = ws.cell(row=cur_row, column=1, value=title)
        cell_title.font = FONT_SECTION
        ws.row_dimensions[cur_row].height = 24
        cur_row += 1

    if df_hier is None or df_hier.empty:
        ws.cell(row=cur_row, column=1, value="No data available for this section.")
        return cur_row + 2

    # Headers
    headers = list(df_hier.columns)
    for c_idx, col_name in enumerate(headers, 1):
        h_cell = ws.cell(row=cur_row, column=c_idx, value=str(col_name))
        h_cell.fill = FILL_HEADER
        h_cell.font = FONT_HEADER
        h_cell.alignment = Alignment(horizontal="left" if c_idx <= 2 else "right", vertical="center")

    ws.row_dimensions[cur_row].height = 22
    cur_row += 1

    # Data Rows
    for r_idx, (_, row) in enumerate(df_hier.iterrows()):
        parent_cat = str(row.get("Parent Category", "") or "")
        sub_cat = str(row.get("Sub-category", "") or "")

        is_parent = sub_cat.startswith("Total ") or sub_cat == "-" or "Total " in sub_cat
        fill = FILL_PARENT if is_parent else (FILL_ZEBRA if r_idx % 2 == 1 else FILL_WHITE)
        font = FONT_PARENT if is_parent else FONT_DATA
        ws.row_dimensions[cur_row].height = 20

        # Parent Category cell
        c_parent = ws.cell(row=cur_row, column=1, value=parent_cat)
        c_parent.font = font
        c_parent.fill = fill
        c_parent.border = BORDER_THIN
        c_parent.alignment = Alignment(horizontal="left", vertical="center")

        # Sub-category cell (slightly indented if child)
        sub_display = sub_cat if is_parent else f"   {sub_cat}"
        c_sub = ws.cell(row=cur_row, column=2, value=sub_display)
        c_sub.font = font
        c_sub.fill = fill
        c_sub.border = BORDER_THIN
        c_sub.alignment = Alignment(horizontal="left", vertical="center")

        # Year / Quarter data columns
        for c_idx, col_name in enumerate(headers[2:], 3):
            val = row[col_name]
            cell = ws.cell(row=cur_row, column=c_idx)
            cell.fill = fill
            cell.border = BORDER_THIN
            cell.font = font
            _format_cell_value(cell, val, label=f"{parent_cat} {sub_cat} {col_name}")

        cur_row += 1

    return cur_row + 1


# ─── Sheet 1: Summary Dashboard ───────────────────────────────────────────────

def _build_summary_sheet(ws, analysis: ComprehensiveAnalysis):
    ws.title = "Summary"
    ws.views.sheetView[0].showGridLines = True

    # Title Banner
    ov = analysis.data.overview
    name = ov.get("name", analysis.symbol)
    price = ov.get("current_price", "N/A")
    mcap = ov.get("key_ratios", {}).get("Market Cap", "N/A")
    sectors = ", ".join(ov.get("sectors", [])) or "N/A"

    ws.merge_cells("A1:G1")
    title_cell = ws["A1"]
    title_cell.value = f"  {name} ({analysis.symbol}) — Fundamental Analysis Executive Dashboard"
    title_cell.font = Font(name="Segoe UI", size=14, bold=True, color="FFFFFF")
    title_cell.fill = FILL_HEADER
    title_cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 36

    ws.merge_cells("A2:G2")
    sub_cell = ws["A2"]
    sub_cell.value = f"  Sector: {sectors}   |   Current Price: Rs {price}   |   Market Cap: Rs {mcap} Cr   |   Retrieved: {analysis.data.retrieval_date}   |   Mode: {analysis.data.financial_type.capitalize()}"
    sub_cell.font = Font(name="Segoe UI", size=9, color="595959")
    sub_cell.fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    sub_cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[2].height = 20

    # Overall Health Score Banner (Row 4 to 6)
    ws.merge_cells("A4:B5")
    score_card = ws["A4"]
    score_card.value = f"{analysis.scorecard.overall_score}/100"
    score_card.font = Font(name="Segoe UI", size=24, bold=True, color="1F4E79")
    score_card.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    score_card.alignment = Alignment(horizontal="center", vertical="center")

    ws.merge_cells("C4:G4")
    grade_cell = ws["C4"]
    grade_cell.value = f"Overall Rating: {analysis.scorecard.overall_grade}"
    grade_cell.font = Font(name="Segoe UI", size=12, bold=True, color="1F4E79")
    grade_cell.fill = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")
    grade_cell.alignment = Alignment(vertical="center")

    ws.merge_cells("C5:G5")
    conc_cell = ws["C5"]
    conc_cell.value = analysis.scorecard.conclusion
    conc_cell.font = Font(name="Segoe UI", size=9, color="424242")
    conc_cell.fill = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")
    conc_cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[4].height = 24
    ws.row_dimensions[5].height = 36

    # Category Pillar Cards (Row 7 to 8)
    cur_r = 7
    ws.cell(row=cur_r, column=1, value="Pillar Health Scores").font = FONT_SECTION
    cur_r += 1

    pillars = [
        ("Profitability", analysis.scorecard.profitability_score),
        ("Growth", analysis.scorecard.growth_score),
        ("Balance Sheet", analysis.scorecard.balance_sheet_score),
        ("Cash Flow", analysis.scorecard.cash_flow_score),
        ("Valuation", analysis.scorecard.valuation_score),
        ("Shareholding", analysis.scorecard.shareholding_score),
    ]

    for col_i, (p_name, p_score) in enumerate(pillars, 1):
        cell_label = ws.cell(row=cur_r, column=col_i, value=f"{p_name}\n({p_score.grade})")
        cell_label.font = FONT_CARD_TITLE
        cell_label.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell_label.fill = FILL_CARD
        cell_label.border = BORDER_THIN

        cell_val = ws.cell(row=cur_r + 1, column=col_i, value=f"{p_score.score:.0f}/100")
        cell_val.font = FONT_CARD_VALUE
        cell_val.alignment = Alignment(horizontal="center", vertical="center")
        cell_val.fill = FILL_CARD
        cell_val.border = BORDER_THIN

    ws.row_dimensions[cur_r].height = 26
    ws.row_dimensions[cur_r + 1].height = 30
    cur_r += 3

    # Strengths & Weaknesses (Row 11 onwards)
    ws.cell(row=cur_r, column=1, value="Major Investment Strengths").font = FONT_SECTION
    ws.cell(row=cur_r, column=5, value="Key Investment Risks / Weaknesses").font = FONT_SECTION
    cur_r += 1

    max_items = max(len(analysis.scorecard.major_strengths), len(analysis.scorecard.major_weaknesses), 1)
    for i in range(max_items):
        # Strength
        s_val = analysis.scorecard.major_strengths[i] if i < len(analysis.scorecard.major_strengths) else ""
        ws.merge_cells(start_row=cur_r + i, start_column=1, end_row=cur_r + i, end_column=4)
        c_s = ws.cell(row=cur_r + i, column=1, value=f"• {s_val}" if s_val else "")
        c_s.font = FONT_DATA
        c_s.fill = PatternFill(start_color="EDF7ED", end_color="EDF7ED", fill_type="solid") if s_val else FILL_WHITE
        c_s.alignment = Alignment(vertical="center")

        # Weakness
        w_val = analysis.scorecard.major_weaknesses[i] if i < len(analysis.scorecard.major_weaknesses) else ""
        ws.merge_cells(start_row=cur_r + i, start_column=5, end_row=cur_r + i, end_column=7)
        c_w = ws.cell(row=cur_r + i, column=5, value=f"• {w_val}" if w_val else "")
        c_w.font = FONT_DATA
        c_w.fill = PatternFill(start_color="FDEDED", end_color="FDEDED", fill_type="solid") if w_val else FILL_WHITE
        c_w.alignment = Alignment(vertical="center")
        ws.row_dimensions[cur_r + i].height = 20

    cur_r += max_items + 1

    # Red Flags Table Snippet
    ws.cell(row=cur_r, column=1, value=f"Red Flags Detected ({analysis.red_flags.flag_count})").font = FONT_SECTION
    cur_r += 1

    if analysis.red_flags.flags:
        rf_headers = ["Severity", "Metric", "Current Value", "Historical / Target", "Reason"]
        for c_idx, h in enumerate(rf_headers, 1):
            hc = ws.cell(row=cur_r, column=c_idx, value=h)
            hc.fill = FILL_HEADER
            hc.font = FONT_HEADER
        cur_r += 1

        for rf in analysis.red_flags.flags:
            sev_fill = FILL_SEV_HIGH if rf.severity == "HIGH" else FILL_SEV_MED
            sev_font = FONT_SEV_HIGH if rf.severity == "HIGH" else FONT_SEV_MED
            c1 = ws.cell(row=cur_r, column=1, value=rf.severity)
            c1.fill = sev_fill
            c1.font = sev_font
            c1.border = BORDER_THIN

            for c_idx, val in enumerate([rf.metric, rf.current_value, rf.historical_value, rf.reason], 2):
                c = ws.cell(row=cur_r, column=c_idx, value=val)
                c.font = FONT_DATA
                c.border = BORDER_THIN
            cur_r += 1
    else:
        ws.merge_cells(start_row=cur_r, start_column=1, end_row=cur_r, end_column=7)
        c_clean = ws.cell(row=cur_r, column=1, value="No forensic red flags detected. Company passes all critical warning checks.")
        c_clean.font = Font(name="Segoe UI", size=10, color="2E7D32", bold=True)
        c_clean.fill = PatternFill(start_color="EDF7ED", end_color="EDF7ED", fill_type="solid")

    ws.freeze_panes = "A3"
    _autofit_columns(ws, min_width=15, max_width=45)


# ─── Sheet 13: Charts Sheet ───────────────────────────────────────────────────

def _build_charts_sheet(ws, charts_map: Dict[str, str]):
    ws.title = "Charts"
    ws.views.sheetView[0].showGridLines = True

    ws.merge_cells("A1:N1")
    t = ws["A1"]
    t.value = "  Fundamental Analysis Visual Charts"
    t.font = Font(name="Segoe UI", size=14, bold=True, color="FFFFFF")
    t.fill = FILL_HEADER
    t.alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 32

    chart_keys = list(charts_map.keys())
    # Arrange in a 2-column grid
    positions = [
        "B3", "H3",
        "B22", "H22",
        "B41", "H41",
        "B60", "H60",
        "B79", "H79",
        "B98"
    ]

    for idx, key in enumerate(chart_keys):
        if idx >= len(positions):
            break
        img_path = charts_map[key]
        if os.path.exists(img_path):
            try:
                img = OpenpyxlImage(img_path)
                img.width = 520
                img.height = 290
                ws.add_image(img, positions[idx])
            except Exception:
                pass


# ─── Master Excel Generator ───────────────────────────────────────────────────

def export_fundamental_excel_report(
    analysis: ComprehensiveAnalysis,
    output_filepath: Optional[str] = None
) -> str:
    """
    Generate <COMPANY>_Fundamental_Analysis.xlsx with 14 comprehensive worksheets.
    """
    sym = analysis.symbol
    if not output_filepath:
        output_filepath = f"{sym}_Fundamental_Analysis.xlsx"

    wb = openpyxl.Workbook()
    # Default sheet
    ws_summary = wb.active

    # Temporary directory for Matplotlib charts
    temp_charts_dir = Path.cwd() / ".temp_charts" / sym
    charts_map = generate_all_charts(analysis, temp_charts_dir)

    # 1. Summary Dashboard
    _build_summary_sheet(ws_summary, analysis)

    # 2. Company Overview
    ws_ov = wb.create_sheet(title="Company Overview")
    ov_data = []
    ov = analysis.data.overview
    ratios = ov.get("key_ratios", {})
    ov_data.append({"Metric": "Company Name", "Value": ov.get("name", sym)})
    ov_data.append({"Metric": "BSE Code", "Value": ov.get("bse_code", "-")})
    ov_data.append({"Metric": "NSE Code", "Value": ov.get("nse_code", "-")})
    ov_data.append({"Metric": "Sector / Industry", "Value": ", ".join(ov.get("sectors", [])) or "-"})
    ov_data.append({"Metric": "Current Market Price", "Value": ov.get("current_price", "-")})
    ov_data.append({"Metric": "52-Week High", "Value": ov.get("52_week_high", "-")})
    ov_data.append({"Metric": "52-Week Low", "Value": ov.get("52_week_low", "-")})
    for k, v in ratios.items():
        ov_data.append({"Metric": k, "Value": v})
    df_ov = pd.DataFrame(ov_data).set_index("Metric")
    _write_dataframe_table(ws_ov, df_ov, start_row=2, title="Company Snapshot & Key Ratios")
    if ov.get("about"):
        r_about = len(df_ov) + 5
        ws_ov.cell(row=r_about, column=1, value="About Company").font = FONT_SECTION
        ws_ov.merge_cells(start_row=r_about + 1, start_column=1, end_row=r_about + 4, end_column=6)
        c_ab = ws_ov.cell(row=r_about + 1, column=1, value=ov["about"])
        c_ab.font = FONT_DATA
        c_ab.alignment = Alignment(wrap_text=True, vertical="top")
    ws_ov.freeze_panes = "B4"
    _autofit_columns(ws_ov)

    # 3. P&L
    ws_pl = wb.create_sheet(title="P&L")
    df_pl_to_write = analysis.data.df_pl_hierarchical if getattr(analysis.data, "df_pl_hierarchical", None) is not None and not analysis.data.df_pl_hierarchical.empty else analysis.data.df_pl
    if "Sub-category" in df_pl_to_write.columns:
        _write_hierarchical_statement_table(ws_pl, df_pl_to_write, start_row=2, title="Profit & Loss Statement (Rs. Crore)")
        ws_pl.freeze_panes = "C4"
    else:
        _write_dataframe_table(ws_pl, df_pl_to_write, start_row=2, title="Profit & Loss Statement (Rs. Crore)")
        ws_pl.freeze_panes = "B4"
    _autofit_columns(ws_pl)

    # 4. Balance Sheet
    ws_bs = wb.create_sheet(title="Balance Sheet")
    df_bs_to_write = analysis.data.df_bs_hierarchical if getattr(analysis.data, "df_bs_hierarchical", None) is not None and not analysis.data.df_bs_hierarchical.empty else analysis.data.df_bs
    if "Sub-category" in df_bs_to_write.columns:
        _write_hierarchical_statement_table(ws_bs, df_bs_to_write, start_row=2, title="Balance Sheet Statement (Rs. Crore)")
        ws_bs.freeze_panes = "C4"
    else:
        _write_dataframe_table(ws_bs, df_bs_to_write, start_row=2, title="Balance Sheet Statement (Rs. Crore)")
        ws_bs.freeze_panes = "B4"
    _autofit_columns(ws_bs)

    # 5. Cash Flow
    ws_cf = wb.create_sheet(title="Cash Flow")
    df_cf_to_write = analysis.data.df_cf_hierarchical if getattr(analysis.data, "df_cf_hierarchical", None) is not None and not analysis.data.df_cf_hierarchical.empty else analysis.data.df_cf
    if "Sub-category" in df_cf_to_write.columns:
        _write_hierarchical_statement_table(ws_cf, df_cf_to_write, start_row=2, title="Cash Flow Statement (Rs. Crore)")
        ws_cf.freeze_panes = "C4"
    else:
        _write_dataframe_table(ws_cf, df_cf_to_write, start_row=2, title="Cash Flow Statement (Rs. Crore)")
        ws_cf.freeze_panes = "B4"
    _autofit_columns(ws_cf)

    # 6. Quarterly Results
    ws_qr = wb.create_sheet(title="Quarterly Results")
    df_qr_to_write = analysis.data.df_quarters_hierarchical if getattr(analysis.data, "df_quarters_hierarchical", None) is not None and not analysis.data.df_quarters_hierarchical.empty else analysis.data.df_quarters
    if "Sub-category" in df_qr_to_write.columns:
        _write_hierarchical_statement_table(ws_qr, df_qr_to_write, start_row=2, title="Quarterly Financial Results (Rs. Crore)")
        ws_qr.freeze_panes = "C4"
    else:
        _write_dataframe_table(ws_qr, df_qr_to_write, start_row=2, title="Quarterly Financial Results (Rs. Crore)")
        ws_qr.freeze_panes = "B4"
    _autofit_columns(ws_qr)

    # 7. Ratios
    ws_rt = wb.create_sheet(title="Ratios")
    r_end = _write_dataframe_table(ws_rt, analysis.ratios.df_profitability, start_row=2, title="Profitability Ratios (%)")
    r_end = _write_dataframe_table(ws_rt, analysis.ratios.df_solvency, start_row=r_end + 1, title="Solvency & Leverage Ratios")
    r_end = _write_dataframe_table(ws_rt, analysis.ratios.df_efficiency, start_row=r_end + 1, title="Operating Efficiency Ratios")
    _write_dataframe_table(ws_rt, analysis.ratios.df_cash_flow_quality, start_row=r_end + 1, title="Cash Conversion & Quality Ratios")
    ws_rt.freeze_panes = "B4"
    _autofit_columns(ws_rt)

    # 8. Growth Analysis
    ws_ga = wb.create_sheet(title="Growth Analysis")
    _write_dataframe_table(ws_ga, analysis.growth.df_growth_summary, start_row=2, title="Compounded Annual Growth Rates (CAGR & YoY)")
    ws_ga.freeze_panes = "B4"
    _autofit_columns(ws_ga)

    # 9. Valuation
    ws_val = wb.create_sheet(title="Valuation")
    _write_dataframe_table(ws_val, analysis.valuation.df_valuation_summary, start_row=2, title="Current Valuation Multiples & Assessment")
    ws_val.freeze_panes = "B4"
    _autofit_columns(ws_val)

    # 10. Shareholding
    ws_sh = wb.create_sheet(title="Shareholding")
    df_sh_to_write = analysis.data.df_shareholding_hierarchical if getattr(analysis.data, "df_shareholding_hierarchical", None) is not None and not analysis.data.df_shareholding_hierarchical.empty else analysis.data.df_shareholding
    if "Sub-category" in df_sh_to_write.columns:
        _write_hierarchical_statement_table(ws_sh, df_sh_to_write, start_row=2, title="Quarterly Shareholding Pattern (%)")
        ws_sh.freeze_panes = "C4"
    else:
        _write_dataframe_table(ws_sh, df_sh_to_write, start_row=2, title="Quarterly Shareholding Pattern (%)")
        ws_sh.freeze_panes = "B4"
    _autofit_columns(ws_sh)

    # 11. Peer Comparison
    ws_peer = wb.create_sheet(title="Peer Comparison")
    _write_dataframe_table(ws_peer, analysis.valuation.df_peer_comparison, start_row=2, title="Sector Peer Comparison", include_index=False)
    ws_peer.freeze_panes = "B4"
    _autofit_columns(ws_peer)

    # 12. Red Flags
    ws_rf = wb.create_sheet(title="Red Flags")
    _write_dataframe_table(ws_rf, analysis.red_flags.df_red_flags, start_row=2, title=f"Automated Forensic Warning Flags ({analysis.red_flags.flag_count})", include_index=False)
    ws_rf.freeze_panes = "A4"
    _autofit_columns(ws_rf)

    # 13. Charts
    ws_charts = wb.create_sheet(title="Charts")
    _build_charts_sheet(ws_charts, charts_map)

    # 14. Raw Data
    ws_raw = wb.create_sheet(title="Raw Data")
    raw_info = [
        {"Property": "Symbol", "Raw Value": analysis.symbol},
        {"Property": "Retrieval Timestamp", "Raw Value": analysis.data.retrieval_date},
        {"Property": "Financial Mode", "Raw Value": analysis.data.financial_type},
        {"Property": "P&L Years Available", "Raw Value": ", ".join(analysis.data.df_pl.columns)},
        {"Property": "Balance Sheet Years", "Raw Value": ", ".join(analysis.data.df_bs.columns)},
        {"Property": "Cash Flow Years", "Raw Value": ", ".join(analysis.data.df_cf.columns)},
        {"Property": "Quarterly Periods", "Raw Value": ", ".join(analysis.data.df_quarters.columns)},
        {"Property": "NSE Announcements Count", "Raw Value": str(len(analysis.data.announcements))},
        {"Property": "NSE Annual Reports Count", "Raw Value": str(len(analysis.data.annual_reports))},
    ]
    df_raw = pd.DataFrame(raw_info).set_index("Property")
    _write_dataframe_table(ws_raw, df_raw, start_row=2, title="Raw Retrieval Metadata & Audit Record")
    _autofit_columns(ws_raw)

    # Save workbook
    abs_path = os.path.abspath(output_filepath)
    wb.save(abs_path)

    return abs_path
