# Figure and table map: input → code → panel

Every published panel and table, the function that draws or assembles it, the rows it uses and
every file it reads. The generated block in [`README.md`](README.md) is the file-level summary;
this page is the function-level one. Both are regenerated from the same record, and neither needs
IQ-TREE or a GPU: the whole presentation tier is `bash benchmark/figures_tables/refresh_all.sh`
(about 4 minutes) on the per-cell record already in `runs/`. All paths are relative to `benchmark/`
unless they start at the repository root; all commands run from the repository root.

## 1. The boundary

The benchmark has two tiers. `scripts/` holds the code that PRODUCED the record: the paired SLURM
harness (`scripts/harness/`), the dataset builders (`scripts/dev/`), and the two record builders
`aggregate_runs.py` (per-cell `RESULT.tsv` → `runs/RUNS.tsv`) and `phase_extract.py` (per-cell
IQ-TREE logs → `runs/PHASE_STEPS.tsv`). `figures_tables/` holds everything that PRESENTS the record:
`figures.py` (every panel, the lettered composites and the figure-contract audit), `tables.py`
(the key numbers, both workbooks, the README map and the cross-checks), the shared library `lib/`
(`figdata`, `figstyle`, `headline`) and `refresh_all.sh`, which runs the two record builders and
then every section of the two files in dependency order. Nothing in `figures_tables/` invokes
`iqtree3`, `sbatch`, `srun`, `nvidia-smi` or `sacct`. The complete input to every figure and table is:

| what | where |
|---|---|
| one row per paired run (226 rows, 182 on the reported build) | `runs/RUNS.tsv` |
| per-cell evidence: `RESULT.tsv`, `PHASES.tsv`, `equivalence.kv`, both arms' `.log`, `.stdout`, `.treefile`, `.contree`, `.iqtree` | `runs/paired/<cell>/` (226 cells) |
| profiler records | `profiling/` |
| source-genome metadata and study citations (Supplementary Data 0 only) | `data/genomes/genome_metadata.tsv`, `data/genomes/study_citations.tsv` |

No figure reads the alignments in `data/protein/` or `data/dna/`; they are harness inputs only.

### The two files

Each file is a set of SECTIONS, one per former generator script, run by name from the repository
root (`python3 benchmark/figures_tables/figures.py decision`, `python3 benchmark/figures_tables/tables.py key_numbers`),
or all in order with no argument. The manifest's `source_script` column records the file and
section (`figures.py (decision)`).

| file | section | writes |
|---|---|---|
| `figures.py` | `equivalence` | Fig. 1B, 1C |
| | `decision` | Fig. 2A (two boxes) |
| | `mechanism` | Fig. 2B (two boxes) |
| | `phases_cost` | Fig. 3B, 3C, 3D |
| | `hardware` | Supplementary Fig. S1D |
| | `supp_ported` | Supplementary Fig. S1C and `runs/REPLICATES_20T.tsv` |
| | `supplementary` | Fig. 1A, Fig. 3A, Supplementary Figs. S1A, S1B, S4 |
| | `trees` | Supplementary Figs. S2, S3 (Letter pages) |
| | `composites` | `figures/composites/` (the lettered figures: PDF and per-page PNG; an SVG of each README figure in `assets/readme/`) |
| | `audit` | the figure-contract audit (naming, 5/6/7/8 pt only, no DejaVu, manifest agreement) |
| `tables.py` | `key_numbers` | `runs/KEY_NUMBERS.tsv`, `runs/KEY_NUMBERS.md` |
| | `source_data` | `figures/Source_Data.xlsx` |
| | `supplementary_data` | `runs/Supplementary_Data_benchmark.xlsx` |
| | `style_workbooks` | the look of a workbook passed on the command line (values untouched) |
| | `benchmark_map` | the generated block of `README.md` (`--check` reports staleness) |
| | `crosscheck` | the six cross-file checks (exit 1 on a failure) |

### The dependency chain

```
runs/paired/<cell>/RESULT.tsv ──► scripts/aggregate_runs.py ──► runs/RUNS.tsv, RUNS.csv, RUNS_summary.tsv
runs/paired/<cell>/{cpu,gpu}.log ─► scripts/phase_extract.py ──► runs/PHASE_STEPS.tsv
RUNS.tsv + PHASE_STEPS.tsv + REPLICATES_20T.tsv + paired/<cell>/gpu.stdout + paired/<cell>/*.contree
  + profiling/*.tsv ──► tables.py key_numbers ──► runs/KEY_NUMBERS.tsv, KEY_NUMBERS.md
RUNS.tsv (+ the extra inputs listed per panel below)
  ──► figures.py equivalence, decision, mechanism, phases_cost, hardware, supp_ported, supplementary, trees
        each writes figures/panels/<stem>.pdf + .png, figures/source_data/<stem>.csv and a row of
        figures/FIGURE_MANIFEST.tsv; supp_ported also writes runs/REPLICATES_20T.tsv
figures/panels/*.pdf ──► figures.py composites ──► figures/composites/ (the lettered figures the journal receives)
figures/panels/*     ──► figures.py audit      (naming, 5/6/7/8 pt only, no DejaVu, manifest agreement)
figures/source_data/*.csv ──► tables.py source_data ──► figures/Source_Data.xlsx
runs/*.tsv + profiling/*.tsv + paired/<cell>/gpu.stdout + data/genomes/*.tsv
  + BENCHMARK_SCHEMA.md ──► tables.py supplementary_data ──► runs/Supplementary_Data_benchmark.xlsx (styled by tables.py style_workbooks)
FIGURE_MANIFEST.tsv + the composites and supplementary_data sections ──► tables.py benchmark_map ──► README.md (generated block)
```

### Two presentation rules every panel follows

**No narrative titles.** The title slot of every panel holds one line, `figstyle.stratum(threads,
n)`: the thread count(s) the panel pools and the n it draws, e.g. `20 threads, n = 145`. The n is
the same value the panel passes to `save_panel`, so the title and the manifest `rows_used` column
cannot disagree. Everything a title used to say (the finding, a quoted number) belongs to the
figure legend in the manuscript; the numbers remain in `runs/KEY_NUMBERS.tsv`. The facet grid
(S1D) keeps the facet key (`L40S (n=105)`) on each facet and carries the stratum line above the
grid (`figstyle.grid_note`); the tree montages (S2, S3) carry it as the page line.

**One plotting box, 40 x 40 mm.** Every panel's plotting area (the box the spines enclose,
excluding tick labels, axis labels, legends and the stratum line) is exactly `figstyle.AX_MM`
= 40 mm square, built with `figstyle.square_axes()` or, for a facet grid, `figstyle.square_grid()`
(every facet 40 mm). A panel may split its square into side-by-side boxes (`figstyle.split_axes`,
Fig. 3C) that together fill 40 x 40 mm. Decorations hang off the box and the save crops to them;
nothing is fitted afterwards (no `tight_layout`), and `save_panel` asserts the box size before
writing. The tree montages (S2, S3) are the only exception (`square=False`): they are whole US
Letter pages, 4 x 4 trees inside 0.8 in margins.

## 2. Row sets (defined once, in `figures_tables/lib/`)

| name | definition | n |
|---|---|---|
| ALL | `figdata.load(REPO)`: rows of `RUNS.tsv` whose `binary_sha256` does not start with `07c70cae` (the earlier build; see `BENCHMARK_SCHEMA.md`) | 182 |
| ALL226 | `figdata.load(REPO, current_build_only=False)` | 226 |
| EQ | `for_equivalence(ALL)`: ALL, the gate-failure cell retained and labelled | 182 |
| FITS | `for_fits(ALL)`: ALL minus the gate-failure cell `g__Ruminococcoides__T20__l40s_cx` and the six `__mdl_*` correctness cells | 175 |
| T20 | FITS with `threads == "20"` | 145 |
| T1 | FITS with `threads == "1"` | 30 |
| HEADLINE | `headline.is_headline`: current build, namespace `__l40s` or `__l40s_b2` (the size ladders) | 48 cells |
| REPL | `headline.is_replicate`: current build, namespace `l40s`, `l40s_b2`, `l40sq` or `l40s_r<k>` | |

Derived quantities, also defined once in `lib/figdata.py`: `complexity(r)` = (2·taxa−3) · patterns ·
rate_cats(model) · states² (abscissa of Fig. 2A, 3D, S1D); `work_units(r)` =
likelihood calls (`gpu_calls`) · patterns · states² · rate_cats(model) (abscissa of Fig. 2B);
`loglog_fit`, `crossover`, `slope_difference` (straight log-log fits and intervals: Fig. 2A, 3D, the
key numbers), `quadratic_fit`, `quadratic_curve`, `end_slope_difference` (the Fig. 2B cost-law
curves: local slope at the smallest and the largest workload, since every series is concave-up on
the log-log scale).

## 3. Figure panels

Panel letters in the file stems equal the manuscript letters; `FIGURES` and `SUPP` in the
`composites` section of `figures.py` decide the lettering, and the `crosscheck` section resolves
references against them. Two boxes that share one letter (a tuple in a composite row) carry the
same letter in their stems. "Rows" is the `rows_used` column of `figures/FIGURE_MANIFEST.tsv`.
Inputs are in addition to `runs/RUNS.tsv`. Functions are in `figures.py` under the section named.

### Main figures

| Figure | Panel stem | Section : function | Row set | Extra inputs | Rows |
|---|---|---|---|---|---|
| 1A | `Fig01a_coverage_stacked_bars` | `supplementary : coverage_bars` via `_stacked_bars` | ALL: runs per taxon count, stacked by workflow (hue) and thread count (hatch) | | 182 |
| 1B | `Fig01b_support_diff_dotplot` | `equivalence : equivalence_panel_b` via `split_supports` (one dot per run; colour alphabet, marker thread count; no tolerance drawn) | EQ | `runs/paired/<cell>/{cpu,gpu}.contree` (per-split supports, asserted equal to RUNS.tsv `support_max_absdiff`); RUNS.tsv `rf_dist` | 182 |
| 1C | `Fig01c_logl_reldiff_ecdf` | `equivalence : equivalence_panel_a` (ECDF per alphabet, difference on the vertical axis) | EQ | RUNS.tsv `logl_reldiff` | 182 |
| 2A left | `Fig02a_decision_t20` | `decision : panel(rows, "20", ...)` via `_decision_panel`; x and y ranges from `decision_limits(FITS)`, shared with the right box; each workflow's break-even ringed and written at the axis foot, shaded 95% bootstrap CI | T20 | | 145 |
| 2A right | `Fig02a_decision_t1` | `decision : panel(rows, "1", ...)`, same ranges, ticks and top axis | T1 | | 30 |
| 2B left | `Fig02b_cost_law_t20` | `mechanism : mechanism_panel_a` (per-arm fits in `_arm_fits` using `figdata.quadratic_fit`, the local slope reported at the smallest and largest workload, paired difference by `figdata.end_slope_difference`; fixed ranges `XLIM`, `YMAX` shared with the right box; circles only, open + dashed = IQ-TREE3, filled + solid = IQ-TREE3-GPU) | T20, all six cards pooled | | 145 |
| 2B right | `Fig02b_cost_law_t1` | `mechanism : mechanism_panel_b` (same helper, same ranges) | T1 | | 30 |
| 3A | `Fig03a_phase_step_heatmap` | `supplementary : phase_step_heatmap` (mean share of each step written on each alphabet's block) | FITS cells, IQ-TREE3 build | `runs/PHASE_STEPS.tsv` (every step) | 175 |
| 3B | `Fig03b_cores_busy` | `phases_cost : panel_c` (violin + jittered circles per build and alphabet) | T20, both builds | PHASE_STEPS.tsv (`total` wall_s, cpu_s) | 145 |
| 3C | `Fig03c_phase_share_cpu_time` | `phases_cost : phases_cost_panel_a`: two half-width boxes sharing one 40 mm square (`figstyle.split_axes`), left the wall share by phase (hues `figstyle.PHASE_COLORS`), right the per-run processor time consumed IQ-TREE3 / IQ-TREE3-GPU as a log-space violin (`_log_violin`) | T20 cells, both builds | PHASE_STEPS.tsv (modelfinder, tree_search, unattributed, total: wall_s, cpu_s) | 145 |
| 3D | `Fig03d_phase_speedup_complexity` | `phases_cost : phases_cost_panel_b`; series = model selection (amino acid), tree search (amino acid), tree search (nucleotide), each with `figdata.crossover` (OLS fit, break-even ring and axis-foot label, shaded 95% bootstrap CI) | T20, both builds; one point per (cell, phase) | PHASE_STEPS.tsv; `complexity(r)` | 212 phase-pairs |

### Supplementary figures

| Figure | Panel stem | Section : function | Row set | Extra inputs | Rows |
|---|---|---|---|---|---|
| S1A | `FigS01a_coverage_l40s` | `supplementary : coverage_bars_l40s` via `_stack` (dark grey 20 threads, light grey 1 thread; solid = amino acid, striped = nucleotide) | ALL, L40S runs | | 142 |
| S1B | `FigS01b_coverage_sweep` | `supplementary : coverage_bars_sweep` via `_stack` (one Okabe-Ito hue per sweep card) | ALL, the five sweep cards | | 40 |
| S1C | `FigS01c_replicates` | `supp_ported : replicates(rows)` via `replicate_sets`, both alphabets on one axis | FITS ∩ REPL, 20 threads, sizes with ≥ 2 runs | | 77 |
| S1D | `FigS01d_card_facet_speedup` | `hardware : panel_d` (one facet per card) | T20, faceted by card | | 145 |
| S2 pages | `FigS02a_trees_aa_p01` … | `trees : montage` → `draw`; cells chosen by `datasets()` / `_priority()`; US Letter pages, 4 x 4 trees inside 0.8 in margins, 6 pt type with 5 pt tip labels (tips labelled only when they fit; an unlabelled tree fills its cell) | every current-build amino-acid inference: one tree per dataset x thread count x card x fixed-model setting (same-settings replicates collapse to one) | `runs/paired/<cell>/gpu.treefile` (Bio.Phylo) | 16 per page |
| S3 pages | `FigS03a_trees_nt_p01` … | same, nucleotide | every current-build nucleotide inference, same rule | gpu.treefile | 16 per page |
| S4 | `FigS04a_model_coverage_data` | `supplementary : model_coverage` | ALL (deliberately includes the `__mdl_*` cells) | | 182 |

### Alternative views

Drafts for choosing a presentation, when written, come from the same section as the panel they
would replace, under the stem `<panel>_alternative<k>_<slug>`; they pass the same audits, have a
manifest row and a `figures/source_data/` CSV, but no Source Data sheet, no place in the README
map and no composite. None is present at the moment; earlier drafts are not distributed.

`split_supports` (Fig. 1B) re-reads both arms' consensus trees with the harness's own split parser
(`scripts/harness/paired_benchmark.sh`, equivalence block) and asserts that the per-run maximum it
finds equals the `support_max_absdiff` column of every run before drawing.

## 4. Tables

`runs/Supplementary_Data_benchmark.xlsx` is assembled by `tables.py supplementary_data` from its
`items` list, one sheet per item, every cell read back and compared with its source, then styled by
`tables.py style_workbooks`. Nothing in it needs IQ-TREE to regenerate.

| Sheet | Contents | Source | Produced by |
|---|---|---|---|
| Data 0 `MAG_SOURCES` | the 435 source MAGs across 14 genera: genus, species, study, citation, DOI, deposit, accessions, host, CheckM2 quality, checksum, alignments | `data/genomes/genome_metadata.tsv` joined to `data/genomes/study_citations.tsv` on `dataset` | `tables.py supplementary_data : mag_sources_table` |
| Data 1 `RUNS` | run table, 226 × 122 | `runs/RUNS.tsv` | `scripts/aggregate_runs.py` from `runs/paired/<cell>/RESULT.tsv` |
| Data 2 `RUNS_summary` | per-stratum mean and s.d. | `runs/RUNS_summary.tsv` | `scripts/aggregate_runs.py` |
| Data 3 `KEY_NUMBERS` | every quoted number | `runs/KEY_NUMBERS.tsv` | `tables.py key_numbers` |
| Data 4 `PHASE_STEPS` | per-step wall and CPU time, both arms, 226 cells | `runs/PHASE_STEPS.tsv` | `scripts/phase_extract.py` (regexes in `PATTERNS`) on `runs/paired/<cell>/{cpu,gpu}.log` |
| Data 5 `REPLICATES_20T` | replicate sets by alphabet and size | `runs/REPLICATES_20T.tsv` | `figures.py supp_ported : write_replicate_table` |
| Data 6 `ARENA_REPORTS` | the engine's arena report per run (device-memory demand, Methods) | computed in memory from `runs/paired/<cell>/gpu.stdout` + `RUNS.tsv` | `tables.py supplementary_data : arena_table` via the `key_numbers` section's `arena_report` |
| Data 7 `LAUNCH_LATENCY` | empty-kernel launch latency | `profiling/launch_latency_l40s.tsv` | `scripts/harness/launch_latency.sh` (record) |
| Data 8 `NSYS_KERNEL_SUMMARY` | Nsight Systems attribution | `profiling/nsys_kernel_summary.tsv` | `scripts/harness/nsys_kernel_time.sh` (record) |
| Data 9 `NSIGHT_SUMMARY` | header only, by design | `profiling/nsight_summary.tsv` | `scripts/harness/roofline_profile.sh` (record) |
| RUNS_columns | column dictionary of Data 1 | `BENCHMARK_SCHEMA.md` | `tables.py supplementary_data : column_dictionary` |

`figures/Source_Data.xlsx` (by `tables.py source_data`) has one sheet per manifest row, named
`Fig<figure><letter>` (plus `_pNN` for the tree pages, and the panel's slug where two boxes share a
letter: `Fig2a_decision_t20`, `Fig2a_decision_t1`), filled from `figures/source_data/<stem>.csv`.
Those CSVs are written automatically by `lib/figstyle.write_source_data` every time `save_panel`
runs, by reading the coordinates of every drawn artist back from the matplotlib figure, so the
workbook changes whenever a panel is redrawn. The `source_data/` directory holds exactly one CSV
per manifest row; a renamed or retired panel's CSV is removed with its panel.

## 5. Changing how something is presented

| Kind of change | Where | Then run |
|---|---|---|
| one panel's labels, annotations, legend, limits, marker size | the function named in section 3 | that section of `figures.py`, then `figures.py composites audit` |
| the stratum line (thread count, n) | `figstyle.stratum`; the n each panel passes | every figure section |
| the plotting-box size (40 mm), facet gap, canvas margins | `AX_MM`, `_GRID_GAP_IN`, `_MARGIN_IN` in `lib/figstyle.py` | every figure section, then `figures.py composites` (which refuses a page that would print below 5 pt) |
| type size, font stack, the two alphabet hues, phase colours, size ramp, arm encoding (fill + line), marker shapes, spines, ticks, DPI, PDF font embedding, `ratio_ticks` | `lib/figstyle.py` | every figure section (or `refresh_all.sh`) |
| which rows enter a panel (build filter, gate-failure cell, `__mdl_*` cells, strata, replicate namespaces); the complexity and work-unit definitions; the estimators | `lib/figdata.py`, `lib/headline.py` | `tables.py key_numbers` and every figure section |
| which panels form a figure, their rows and letters (a tuple shares one letter), composite titles, page scale limits | `FIGURES` and `SUPP` in the `composites` section of `figures.py` | `figures.py composites`, `tables.py benchmark_map crosscheck` |
| tree montages: tip-label shortening, scale bar, trees per page, page size and margins | the `trees` section (`short_label`, `nice_scale`, `draw`, `PAGE_IN`, `MARGIN_IN`, `NROW`, `NCOL`) | `figures.py trees` |
| workbook sheet descriptions and order | the `items` list in the `supplementary_data` section of `tables.py` | `tables.py supplementary_data` |
| workbook look (header fill, widths, tab colours) | the `style_workbooks` section of `tables.py` | `tables.py style_workbooks <xlsx>` |

```bash
cd <repository root>
python3 benchmark/figures_tables/figures.py <section>      # rewrites every panel that section owns (seconds to ~1 min)
python3 benchmark/figures_tables/figures.py composites     # the lettered figure files (needs pdftoppm)
python3 benchmark/figures_tables/figures.py audit          # the contract audit (exit 1 on a violation)
bash benchmark/figures_tables/refresh_all.sh               # or everything, ~4 min, no GPU
bash scripts/run_tests.sh --quick                          # invariants, figure contract, cross-checks, the README map
```

Notes. Run from the repository root: reads resolve from the file's own location, but `save_panel`
writes to the relative default `benchmark/figures/panels` unless `PANEL_DIR` is set, and
`PANEL_DIR=/some/scratch` is the way to draft one section's panels without touching `figures/`. A
section always rebuilds every panel in its `main()` list. The audit rejects a literal `fontsize=`
or `.grid(True)` in the panel-generating sections, any PDF type size outside 5/6/7/8 pt (mathtext
sub- and superscripts at 0.7× are allowed) and an embedded DejaVu font, which `figstyle.apply()`
also raises on when Arial, Helvetica and Nimbus Sans are all absent (Arial is used when its
TrueType files are placed in `figures/arial/`). `composites` refuses a layout whose uniform page
scale would print type below 5 pt.
