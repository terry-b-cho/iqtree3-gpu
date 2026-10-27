# Figure outputs

**`panels/` is authoritative.** Every published figure is emitted there, under the naming contract
`Fig{NN}{p}_{slug}` / `FigS{NN}{p}_{slug}` (multipage montages append `_p{NN}`), as a vector PDF plus
a 300 dpi PNG, with a row in `FIGURE_MANIFEST.tsv` and a passing check from
`benchmark/figures_tables/figures.py audit`.

## Contents

| path | what |
|---|---|
| `panels/` | 25 panels x {pdf, png}: three main figures (Fig01-Fig03) and four supplementary (FigS01-FigS04), numbered in order of first citation; panel letters in the stems match the manuscript letters (two boxes that share a letter, such as Fig. 2A left and right, carry the same letter in their stems). Alternative drafts, when written, use the stem `<panel>_alternative<k>_<slug>`, sit in no composite and have no `Source_Data.xlsx` sheet; none is present at the moment. Every plotting box is a 40 x 40 mm square except the tree montages; the title slot holds only the thread count and n. |
| `source_data/`, `Source_Data.xlsx` | Source Data: the values behind every mark in every panel, read back from the drawn artists by the panel writer (one CSV per panel; the workbook has one sheet per panel and a README sheet). Written by `benchmark/figures_tables/tables.py source_data`; exactly one CSV per manifest row. |
| `arial/` | optional and not distributed (Monotype licence): the Arial TrueType files `ARIAL.TTF`, `ARIALBD.TTF`, `ARIALI.TTF`, `ARIALBI.TTF`. When present, `figstyle.apply()` registers them and every figure embeds Arial; when absent the stack falls through to Nimbus Sans, which is metrically identical, and the panel audit still rejects a DejaVu fallback. |
| `composites/` | One lettered figure per file, as the journal receives it: the PDF (vector, all pages), plus per page a 300 dpi PNG and an SVG converted from that PDF with its text as outlines (the SVG is what the READMEs embed, so figures render sharp on GitHub). Written by `figures.py composites`. |
| `FIGURE_MANIFEST.tsv` | One row per panel: file, figure, panel, title, generating script, source table, rows used, data date. Written by the pipeline, never by hand. |

Nothing else belongs directly in this directory. `benchmark/figures_tables/tables.py crosscheck` fails the build if
anything other than this README and `FIGURE_MANIFEST.tsv` appears at the top level, so the loose
ad-hoc output that used to accumulate here cannot come back.

## Regenerating

```
bash benchmark/figures_tables/refresh_all.sh
```

That runs the record builders, the key numbers, every figure section, the composites, the audit,
both workbooks and the map, in dependency order; `bash scripts/run_tests.sh --quick` then re-checks
the figure contract and the cross-file checks. One panel at a time:
`python3 benchmark/figures_tables/figures.py <section>` (sections and functions in `../FIGURE_MAP.md`).
