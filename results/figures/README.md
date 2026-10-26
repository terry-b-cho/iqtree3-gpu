# Figure outputs

**`panels/` is authoritative.** Every published figure is emitted there, under the naming contract
`Fig{NN}{p}_{slug}` / `FigS{NN}{p}_{slug}` (multipage montages append `_p{NN}`), as a vector PDF plus
a 300 dpi PNG, with a row in `FIGURE_MANIFEST.tsv` and a passing check from
`scripts/panels.py`.

`previews/` holds one assembled preview per figure plus the greyscale proof sheet. These are for
reading and for the accessibility check; the individual panels remain the submission unit.

## Contents

| path | what |
|---|---|
| `panels/` | 43 panels x {pdf, png}. Six main figures (Fig01-Fig06) and thirteen supplementary (FigS01-FigS13), numbered in order of first citation. |
| `previews/` | One assembled preview per figure, plus the greyscale proof sheet of the main figures (a production check, not a supplementary figure). |
| `source_data/`, `Source_Data.xlsx` | Source Data: the values behind every mark in every panel, read back from the drawn artists by the panel writer (one CSV per panel; the workbook has one sheet per panel and a README sheet). Written by `scripts/source_data.py`. |
| `FIGURE_MANIFEST.tsv` | One row per panel: file, figure, panel, title, generating script, source table, rows used, data date. Written by the pipeline, never by hand. |

Nothing else belongs directly in this directory. `scripts/crosscheck.py` fails the build if
anything other than this README and `FIGURE_MANIFEST.tsv` appears at the top level, so the loose
ad-hoc output that used to accumulate here cannot come back.

## Regenerating

```
bash scripts/refresh_all.sh
```

That runs the aggregator, the key-number derivation, the phase extraction, the six main-figure
generators, the supplementary generators, the preview assembly and the contract audit, in dependency
order. `bash scripts/run_tests.sh` then re-checks the figure contract and the manuscript's figure
references.
