# benchmark/figures_tables/ - the presentation tier

Turns the measured record in `benchmark/runs/` into every figure, table and number of the
manuscript. Nothing here runs IQ-TREE or needs a GPU. The code that produced the record is in
`benchmark/scripts/`.

| file | what it is |
|---|---|
| `refresh_all.sh` | regenerates everything, in dependency order (about 4 minutes) |
| `figures.py` | every panel, the lettered composites and the figure-contract audit; one section per former script |
| `tables.py` | the key numbers, the Source Data and Supplementary Data workbooks, the README map and the cross-checks; one section each |
| `lib/figdata.py` | the one analysis layer: row filters, complexity and work-unit definitions, every fit and interval |
| `lib/figstyle.py` | the one publication style: 6 pt Arial, the colour contract, the 40 x 40 mm plotting box, the panel writer |
| `lib/headline.py` | the build filter and replicate namespaces |

Run from the repository root:

```bash
bash benchmark/figures_tables/refresh_all.sh            # everything
python3 benchmark/figures_tables/figures.py             # every figure section, then composites and audit
python3 benchmark/figures_tables/figures.py decision    # one section (Fig. 2A)
python3 benchmark/figures_tables/tables.py key_numbers  # one section
python3 benchmark/figures_tables/tables.py              # key numbers, both workbooks, the map, the cross-checks
```

| `figures.py` section | writes | `tables.py` section | writes |
|---|---|---|---|
| `equivalence` | Fig. 1B, 1C | `key_numbers` | `runs/KEY_NUMBERS.tsv`, `.md` |
| `decision` | Fig. 2A | `source_data` | `figures/Source_Data.xlsx` |
| `mechanism` | Fig. 2B | `supplementary_data` | `runs/Supplementary_Data_benchmark.xlsx` |
| `phases_cost` | Fig. 3B, 3C, 3D | `style_workbooks <xlsx>` | the look of a workbook, values untouched |
| `hardware` | Supplementary Fig. S1D | `benchmark_map [--check]` | the map in `benchmark/README.md` |
| `supp_ported` | S1C, `runs/REPLICATES_20T.tsv` | `crosscheck` | the six cross-file checks |
| `supplementary` | Fig. 1A, 3A, S1A, S1B, S4 | | |
| `trees` | S2, S3 (US Letter pages) | | |
| `composites` | `figures/composites/` | | |
| `audit` | the figure contract (exit 1 on a violation) | | |

Which function draws which panel, which rows it uses and how to change a panel:
[`../FIGURE_MAP.md`](../FIGURE_MAP.md).
