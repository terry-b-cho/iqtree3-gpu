#!/usr/bin/env python3
"""source_data.py - assemble the Source Data workbook from the per-panel CSVs.

    python3 scripts/source_data.py [<out.xlsx>]

The panel writer (`scripts/lib/figstyle.py`, `write_source_data`) records, for every published
panel, the values behind every drawn mark, read back from the matplotlib artists so the file cannot
disagree with the figure. This script collects those CSVs into one workbook with a README sheet and
one sheet per panel, named by the panel's figure and letter (Fig1a, FigS8a, ...), in manifest order.
Panels whose marks are drawings rather than data (the tree montages) carry a sheet that says which
tree files they were drawn from.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
"""
import os, csv, sys
import pandas as pd

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SD = os.path.join(REPO, "results", "figures", "source_data")
MANIFEST = os.path.join(REPO, "results", "figures", "FIGURE_MANIFEST.tsv")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results", "figures", "Source_Data.xlsx")
TREE_SCRIPT = "trees.py"


def main():
    man = pd.read_csv(MANIFEST, sep="\t", dtype=str).fillna("")
    readme, sheets = [], []
    for _, r in man.iterrows():
        stem = r["file"]; fig = r["figure"]; panel = r["panel"]
        sheet = f"Fig{fig}{panel}" + (f"_{stem.rsplit('_p', 1)[1]}" if "_p" in stem[-4:] or stem.rsplit("_", 1)[-1].startswith("p") and stem.rsplit("_", 1)[-1][1:].isdigit() else "")
        sheet = sheet[:31]
        csv_path = os.path.join(SD, stem + ".csv")
        if r["source_script"] == TREE_SCRIPT:
            df = pd.DataFrame({"note": [f"Tree drawings; the plotted objects are the maximum-likelihood tree files of the "
                                        f"{r['rows_used']} datasets shown, distributed per cell in the archived record "
                                        f"(Supplementary Note 2, §11.3: <cell>/gpu.treefile and <cell>/cpu.treefile)."]})
            kind = "tree files (see note)"
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path); kind = f"{len(df)} plotted values"
        else:
            df = pd.DataFrame({"note": ["no plotted values were recorded for this panel"]}); kind = "none recorded"
        readme.append(dict(sheet=sheet, figure=fig, panel=panel, title=r["title"], generating_script=r["source_script"],
                           source_table=r["source_table"], rows_used=r["rows_used"], data_date=r["data_date"], contents=kind))
        sheets.append((sheet, df))
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        intro = pd.DataFrame({"Source Data": [
            "One sheet per published panel. Columns: axes (index of the axes within the panel), axes_title, kind "
            "(scatter, line, bar, segment, band, image), series (legend label where the artist has one), x_label, "
            "y_label, x, y, width (bars), height_or_value (bar height, or the cell value of an image).",
            "Values are in data coordinates as drawn; a log axis does not change them. Fitted lines and confidence "
            "bands appear as 'line' and 'band' rows with their series label.",
            "The values are read back from the drawn artists by the panel writer (scripts/lib/figstyle.py), so a sheet "
            "cannot disagree with its panel. The underlying run table is in results/benchmarks/ of the archived record.",
        ]})
        intro.to_excel(xw, sheet_name="README", index=False)
        pd.DataFrame(readme).to_excel(xw, sheet_name="README", index=False, startrow=5)
        seen = set()
        for sheet, df in sheets:
            name = sheet; k = 2
            while name in seen:
                name = f"{sheet[:28]}_{k}"; k += 1
            seen.add(name)
            df.to_excel(xw, sheet_name=name, index=False)
    print(f"  wrote {os.path.relpath(OUT, REPO)}: {len(sheets)} panel sheets")


if __name__ == "__main__":
    main()
