#!/usr/bin/env python3
"""style_workbooks.py - one presentation for every supplementary workbook, values untouched.

    python3 scripts/style_workbooks.py <workbook.xlsx> [...]

Applies to each workbook: a README sheet with a title, wrapped introductory text and a contents
table whose sheet names link to the sheets; on every data sheet a bold shaded header row with a
filter, frozen at the header, column widths fitted to the content (capped), and a sheet-tab colour
by group (read from the README's group/contents columns). No cell value is changed: every value is
recorded before styling and compared after the workbook is reloaded; a difference fails the run.
"""
import sys
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HEAD_FILL = PatternFill("solid", fgColor="D9E1F2")
HEAD_FONT = Font(bold=True)
TITLE_FONT = Font(bold=True, size=14)
BODY_FONT = Font(size=11)
THIN = Side(style="thin", color="8EA9DB")
TAB = {"Multi-source review": "2F5597", "Retained measure": "548235", "PubMed-first analysis": "7F7F7F",
       "Supplementary Data": "2F5597", "README": "1F3864", "RUNS_columns": "BF9000"}


def snapshot(wb):
    out = {}
    for ws in wb.worksheets:
        out[ws.title] = [[c.value for c in row] for row in ws.iter_rows()]
    return out


def _key(v):
    return str(v or "").strip().lower().replace(" ", "_")


def _header_row(ws):
    """The contents table's header: below the title, first cell a known label, second cell filled."""
    return next((r for r in range(2, ws.max_row + 1)
                 if _key(ws.cell(r, 1).value) in ("table", "supplementary_data", "sheet", "item") and ws.cell(r, 2).value), None)


def style_readme(ws, wb):
    """Title, wrapped intro paragraphs merged across the table width, contents table with links."""
    max_col = max(ws.max_column, 6)
    # the contents table starts at the first row whose first cell is a header keyword
    for rng in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(rng))
    hdr_row = _header_row(ws)
    ws["A1"].font = TITLE_FONT
    for r in range(2, (hdr_row or ws.max_row + 1)):
        v = ws.cell(r, 1).value
        if v:
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=max_col)
            ws.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top")
            ws.cell(r, 1).font = BODY_FONT
            ws.row_dimensions[r].height = max(30, 15 * (len(str(v)) // 130 + 1))
    if hdr_row:
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(hdr_row, c)
            if cell.value is not None:
                cell.value = str(cell.value).replace("_", " ").capitalize() if str(cell.value).islower() else cell.value
                cell.font = HEAD_FONT; cell.fill = HEAD_FILL; cell.border = Border(bottom=THIN)
                cell.alignment = Alignment(vertical="center", wrap_text=True)
        headers = [_key(ws.cell(hdr_row, c).value) for c in range(1, ws.max_column + 1)]
        sheet_col = next((i + 1 for i, h in enumerate(headers) if h in ("sheet", "table")), 1)
        for r in range(hdr_row + 1, ws.max_row + 1):
            name = ws.cell(r, sheet_col).value
            if name and name in wb.sheetnames:
                cell = ws.cell(r, sheet_col)
                from openpyxl.worksheet.hyperlink import Hyperlink
                cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"'{name}'!A1", display=str(name))
                cell.font = Font(color="0563C1", underline="single")
            for c in range(1, ws.max_column + 1):
                ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical="top")
        ws.auto_filter.ref = f"A{hdr_row}:{get_column_letter(ws.max_column)}{ws.max_row}"
        ws.freeze_panes = ws.cell(hdr_row + 1, 1)
    widths = {}
    for c in range(1, ws.max_column + 1):
        h = headers[c - 1] if hdr_row else ""
        widths[c] = {"contents": 90, "content": 90, "title": 30, "group": 34, "cited_in": 42, "source_file": 40}.get(h, 16)
    for c, w in widths.items():
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.sheet_properties.tabColor = TAB["README"]


def style_table(ws, group):
    if ws.max_row < 1: return
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(1, c)
        cell.font = HEAD_FONT; cell.fill = HEAD_FILL; cell.border = Border(bottom=THIN)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
    for c in range(1, ws.max_column + 1):
        col = get_column_letter(c)
        longest = len(str(ws.cell(1, c).value or ""))
        for r in range(2, min(ws.max_row, 400) + 1):
            v = ws.cell(r, c).value
            if v is not None:
                longest = max(longest, len(str(v)))
        ws.column_dimensions[col].width = max(9, min(60, int(longest * 0.95) + 2))
    ws.row_dimensions[1].height = 30
    for key, colour in TAB.items():
        if group.startswith(key):
            ws.sheet_properties.tabColor = colour; break


def main():
    for path in sys.argv[1:]:
        wb = openpyxl.load_workbook(path)
        before = snapshot(wb)
        groups = {}
        if "README" in wb.sheetnames:
            ws = wb["README"]
            hdr_row = _header_row(ws)
            if hdr_row:
                headers = [_key(ws.cell(hdr_row, c).value) for c in range(1, ws.max_column + 1)]
                sc = next((i + 1 for i, h in enumerate(headers) if h in ("sheet", "table")), 1)
                gc = next((i + 1 for i, h in enumerate(headers) if h == "group"), None)
                for r in range(hdr_row + 1, ws.max_row + 1):
                    name = ws.cell(r, sc).value
                    if name: groups[name] = str(ws.cell(r, gc).value or "") if gc else "Supplementary Data"
            style_readme(ws, wb)
        for ws in wb.worksheets:
            if ws.title == "README": continue
            style_table(ws, groups.get(ws.title, ws.title))
        after = snapshot(wb)                    # checked before saving: nothing but README header labels may differ
        diffs = 0
        for name, rows in before.items():
            for i, row in enumerate(rows):
                for j, v in enumerate(row):
                    w = after[name][i][j] if i < len(after[name]) and j < len(after[name][i]) else None
                    if v != w and not (name == "README" and isinstance(v, str) and _key(w) == _key(v)):
                        diffs += 1
        if diffs:
            sys.exit(f"  {path}: {diffs} cell(s) would change; not saved")
        wb.save(path)
        print(f"  styled {path.split('/')[-1]}: {len(wb.sheetnames)} sheets, values unchanged")


if __name__ == "__main__":
    main()
