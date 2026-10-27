# Figure and table map: input → code → panel

Every published panel and table, the function that draws or assembles it, the rows it uses and
every file it reads. The generated block in [`README.md`](README.md) is the script-level summary;
this page is the function-level one. Both are regenerated from the same record, and neither needs
IQ-TREE or a GPU: the whole presentation tier is `bash benchmark/scripts/refresh_all.sh` (about
3 minutes) on the per-cell record already in `runs/`. All paths are relative to `benchmark/` unless
they start at the repository root; all commands run from the repository root.

## 1. The boundary

`scripts/refresh_all.sh` calls only the Python scripts in `scripts/`. None of them invokes
`iqtree3`, `sbatch`, `srun`, `nvidia-smi` or `sacct`. The only code that runs IQ-TREE lives in
`scripts/harness/` (the paired SLURM benchmark) and `scripts/dev/` (dataset builders), and the
refresh calls neither. The complete input to every figure and table is:

| what | where |
|---|---|
| one row per paired run (226 rows, 182 on the reported build) | `runs/RUNS.tsv` |
| per-cell evidence: `RESULT.tsv`, `PHASES.tsv`, `equivalence.kv`, both arms' `.log`, `.stdout`, `.treefile`, `.contree`, `.iqtree` | `runs/paired/<cell>/` (226 cells) |
| profiler records | `profiling/` |
| source-genome metadata and study citations (Supplementary Data 0 only) | `data/genomes/genome_metadata.tsv`, `data/genomes/study_citations.tsv` |

No figure reads the alignments in `data/protein/` or `data/dna/`; they are harness inputs only.

### The dependency chain

```
runs/paired/<cell>/RESULT.tsv ──► aggregate_runs.py ──► runs/RUNS.tsv, RUNS.csv, RUNS_summary.tsv
runs/paired/<cell>/{cpu,gpu}.log ─► phase_extract.py ──► runs/PHASE_STEPS.tsv
RUNS.tsv + PHASE_STEPS.tsv + REPLICATES_20T.tsv + paired/<cell>/gpu.stdout + paired/<cell>/*.contree
  + profiling/*.tsv + card_specs.tsv ──► key_numbers.py ──► runs/KEY_NUMBERS.tsv, KEY_NUMBERS.md
RUNS.tsv (+ the extra inputs listed per panel below)
  ──► equivalence.py, decision.py, mechanism.py, phases_cost.py                           (main figures)
  ──► supp_ported.py, supplementary.py, trees.py                                           (supplementary)
        each writes figures/panels/<stem>.pdf + .png, figures/source_data/<stem>.csv and a row of
        figures/FIGURE_MANIFEST.tsv; supp_ported.py also writes runs/REPLICATES_20T.tsv
figures/panels/*.png ──► assemble.py   ──► figures/previews/ (reading copies + greyscale proof)
figures/panels/*.pdf ──► composites.py ──► figures/composites/ (the lettered figures the journal receives)
figures/panels/*     ──► panels.py       (audit: naming, 5/6/7/8 pt only, no DejaVu, manifest agreement)
figures/source_data/*.csv ──► source_data.py ──► figures/Source_Data.xlsx
runs/*.tsv + profiling/*.tsv + card_specs.tsv + paired/<cell>/gpu.stdout + data/genomes/*.tsv
  + BENCHMARK_SCHEMA.md ──► supplementary_data.py ──► runs/Supplementary_Data_benchmark.xlsx (styled by style_workbooks.py)
FIGURE_MANIFEST.tsv + composites.py + supplementary_data.py ──► benchmark_map.py ──► README.md (generated block)
```

### Two presentation rules every panel follows

**No narrative titles.** The title slot of every panel holds one line, `figstyle.stratum(threads,
n)`: the thread count(s) the panel pools and the n it draws, e.g. `20 threads, n = 145`. The n is
the same value the panel passes to `save_panel`, so the title and the manifest `rows_used` column
cannot disagree. Everything a title used to say (the finding, a quoted number) belongs to the
figure legend in the manuscript; the numbers remain in `runs/KEY_NUMBERS.tsv`. Facet grids
(S6, S7B) keep the facet key (`L40S (n=105)`) on each facet and carry the
stratum line above the grid (`figstyle.grid_note`); the tree montages (S1, S2) carry it as the
page line.

**One plotting box, 40 x 40 mm.** Every panel's plotting area (the box the spines enclose,
excluding tick labels, axis labels, legends and the stratum line) is exactly `figstyle.AX_MM`
= 40 mm square, built with `figstyle.square_axes()` or, for a facet grid, `figstyle.square_grid()`
(every facet 40 mm, 4 mm apart). Decorations hang off the box and the save crops to them;
nothing is fitted afterwards (no `tight_layout`), and `save_panel` asserts the box size before
writing. The tree montages (S1, S2) are the only exception (`square=False`).

## 2. Row sets (defined once, in `scripts/lib/`)

| name | definition | n |
|---|---|---|
| ALL | `figdata.load(REPO)`: rows of `RUNS.tsv` whose `binary_sha256` does not start with `07c70cae` (the superseded build) | 182 |
| ALL226 | `figdata.load(REPO, current_build_only=False)` | 226 |
| EQ | `for_equivalence(ALL)`: ALL, the gate-failure cell retained and labelled | 182 |
| FITS | `for_fits(ALL)`: ALL minus the gate-failure cell `g__Ruminococcoides__T20__l40s_cx` and the six `__mdl_*` correctness cells | 175 |
| T20 | FITS with `threads == "20"` | 145 |
| T1 | FITS with `threads == "1"` | 30 |
| L40S | T20 with `card == "NVIDIA_L40S"` | |
| HEADLINE | `headline.is_headline`: current build, namespace `__l40s` or `__l40s_b2` (the size ladders) | 48 cells |
| REPL | `headline.is_replicate`: current build, namespace `l40s`, `l40s_b2`, `l40sq` or `l40s_r<k>` | |

Derived quantities, also defined once in `lib/figdata.py`: `complexity(r)` = (2·taxa−3) · patterns ·
rate_cats(model) · states² (abscissa of Fig. 2A/B, 4B, S6); `work_units(r)` =
likelihood calls (`gpu_calls`) · patterns · states² · rate_cats(model) (abscissa of Fig. 3A/B);
`loglog_fit`, `crossover`, `slope_difference` (straight log-log fits and intervals: Fig. 2, the
key numbers), `quadratic_fit`, `quadratic_curve`, `end_slope_difference` (the Fig. 3 cost-law
curves: local exponent at the smallest and the largest workload, since every series is
concave-up on the log-log scale).

## 3. Figure panels

Panel letters in the file stems equal the manuscript letters; `composites.py` (`FIGURES`, `SUPP`)
is what decides the lettering, and `crosscheck.py` resolves references against it. "Rows" is the
`rows_used` column of `figures/FIGURE_MANIFEST.tsv`. Inputs are in addition to `runs/RUNS.tsv`.

### Main figures

| Figure | Panel stem | Script : function | Row set | Extra inputs | Rows |
|---|---|---|---|---|---|
| 1A | `Fig01a_logl_reldiff_ecdf` | `equivalence.py : panel_a` (ECDF per alphabet, difference on the vertical axis) | EQ | RUNS.tsv `logl_reldiff` | 182 |
| 1B | `Fig01b_support_diff_dotplot` | `equivalence.py : panel_b` via `split_supports` (one dot per run; colour alphabet, marker thread count; no tolerance drawn) | EQ | `runs/paired/<cell>/{cpu,gpu}.contree` (per-split supports, asserted equal to RUNS.tsv `support_max_absdiff`); RUNS.tsv `rf_dist` | 182 |
| 2A | `Fig02a_decision_t20` | `decision.py : panel(rows, "20", ...)` via `_decision_panel`; x and y ranges from `limits(FITS)`, shared with 2B | T20 | | 145 |
| 2B | `Fig02b_decision_t1` | `decision.py : panel(rows, "1", ...)`, same ranges, ticks and top axis as 2A | T1 | | 30 |
| 3A | `Fig03a_cost_law` | `mechanism.py : panel_a` (per-arm fits in `_arm_fits` using `figdata.quadratic_fit`, the local exponent reported at the smallest and largest workload, paired difference by `figdata.end_slope_difference`; x and y ranges from `limits(FITS)`, shared with 3B; circles only, open + dashed = IQ-TREE3, filled + solid = IQ-TREE3-GPU) | L40S headline fit; T20 pooled six cards as the grey reference | | 145 |
| 3B | `Fig03b_cost_law_t1` | `mechanism.py : panel_b` (same helper, same ranges) | T1 | | 30 |
| 4A | `Fig04a_phase_share_cpu_time` | `phases_cost.py : panel_a`: two half-width boxes sharing one 40 mm square (`figstyle.split_axes`), left the wall share by phase (hues `figstyle.PHASE_COLORS`), right the per-run processor time consumed IQ-TREE3 / IQ-TREE3-GPU as a log-space violin (`_log_violin`) | T20 cells, both builds | `runs/PHASE_STEPS.tsv` (modelfinder, tree_search, unattributed, total: wall_s, cpu_s) | 145 |
| 4B | `Fig04b_phase_speedup_complexity` | `phases_cost.py : panel_b`; series = model selection (amino acid), tree search (amino acid), tree search (nucleotide), each with `figdata.crossover` (OLS fit, break-even ring and axis-foot label, shaded 95% bootstrap CI) | T20, both builds; one point per (cell, phase) | PHASE_STEPS.tsv; `complexity(r)` | 212 phase-pairs |
| 4C | `Fig04c_cores_busy` | `phases_cost.py : panel_c` (violin + jittered circles per build and alphabet) | T20, both builds | PHASE_STEPS.tsv (`total` wall_s, cpu_s) | 145 |

### Supplementary figures

| Figure | Panel stem | Script : function | Row set | Extra inputs | Rows |
|---|---|---|---|---|---|
| S1 pages | `FigS01a_trees_aa_p01` … | `trees.py : montage` → `draw`; cells chosen by `datasets()` / `_priority()`; US Letter pages, 4 x 4 trees inside 0.8 in margins, 6 pt type with 5 pt tip labels (tips labelled only when they fit; an unlabelled tree fills its cell) | every current-build amino-acid inference: one tree per dataset x thread count x card x model (same-settings replicates collapse to one) | `runs/paired/<cell>/gpu.treefile` (Bio.Phylo) | 16 per page |
| S2 pages | `FigS02a_trees_nt_p01` … | same, nucleotide | every current-build nucleotide inference, same rule | gpu.treefile | 16 per page |
| S3 | `FigS03a_model_coverage_data` | `supplementary.py : model_coverage` | ALL (deliberately includes the `__mdl_*` cells) | | 182 |
| S4 | `FigS04a_replicates` | `supp_ported.py : replicates(rows)` via `replicate_sets`, both alphabets on one axis (no letter in the composite) | FITS ∩ REPL, 20 threads, sizes with ≥ 2 runs | | 77 |
| S5 | `FigS05a_phase_step_heatmap` | `supplementary.py : phase_step_heatmap` (mean share of each step written on each alphabet's block) | FITS cells, IQ-TREE3 build | `runs/PHASE_STEPS.tsv` (every step) | 175 |
| S6 | `FigS06a_card_facet_speedup` | `hardware.py : panel_d` (no letter in the composite) | T20, faceted by card | | 145 |
| S7A | `FigS07a_coverage_stacked_bars` | `supplementary.py : coverage_bars` via `_stacked_bars` | ALL: runs per taxon count, stacked by workflow (hue) and thread count (hatch) | | 182 |
| S7B | `FigS07b_coverage_by_card` | `supplementary.py : coverage_bars_by_card` via `_stack`: two boxes, L40S alone (grey, lightness = thread count) and the five sweep cards stacked (one Okabe-Ito hue per card); solid = amino acid, striped = nucleotide | ALL | | 182 |

### Alternative views (not in any composite)

Drafts for choosing a presentation. Each is written by the same generator as the panel it
replaces, from the same rows, under the stem `<panel>_alternative<k>_<slug>`, and passes the same
audits; it has a manifest row and a `figures/source_data/` CSV, but no Source Data sheet
(`source_data.py` skips `_alternative` stems, so `Fig1a` is the published panel), no place in the
README map (which is composite-driven) and no composite. Adopting one means naming it in
`composites.py` and retiring the rest.

| Replaces | Stem | Function | What it shows |
|---|---|---|---|
| (none at present) | | | every draft has been adopted or retired; the retired ones are in the private archive |

The Fig. 1A and 1B drafts were decided on 2026-10-26: the ECDF and the run dot plot became the
panels, the histograms and the other drafts went to the archive. `split_supports` (Fig. 1B)
re-reads both arms' consensus trees with the harness's own split parser
(`scripts/harness/paired_benchmark.sh`, equivalence block) and asserts that the per-run maximum it
finds equals the `support_max_absdiff` column of every run before drawing.

## 4. Tables

`runs/Supplementary_Data_benchmark.xlsx` is assembled by `supplementary_data.py` from its `items`
list, one sheet per item, every cell read back and compared with its source, then styled by
`style_workbooks.py`. Nothing in it needs IQ-TREE to regenerate.

| Sheet | Contents | Source | Produced by |
|---|---|---|---|
| Data 0 `MAG_SOURCES` | the 435 source MAGs across 14 genera: genus, species, study, citation, DOI, deposit, accessions, host, CheckM2 quality, checksum, alignments | `data/genomes/genome_metadata.tsv` joined to `data/genomes/study_citations.tsv` on `dataset` | `supplementary_data.py : mag_sources_table` |
| Data 1 `RUNS` | run table, 226 × 122 | `runs/RUNS.tsv` | `aggregate_runs.py` from `runs/paired/<cell>/RESULT.tsv` |
| Data 2 `RUNS_summary` | per-stratum mean and s.d. | `runs/RUNS_summary.tsv` | `aggregate_runs.py` |
| Data 3 `KEY_NUMBERS` | every quoted number | `runs/KEY_NUMBERS.tsv` | `key_numbers.py` |
| Data 4 `PHASE_STEPS` | per-step wall and CPU time, both arms, 226 cells | `runs/PHASE_STEPS.tsv` | `phase_extract.py` (regexes in `PATTERNS`) on `runs/paired/<cell>/{cpu,gpu}.log` |
| Data 5 `REPLICATES_20T` | replicate sets by alphabet and size | `runs/REPLICATES_20T.tsv` | `supp_ported.py : write_replicate_table` |
| Data 6 `ARENA_REPORTS` | the engine's arena report per run (device-memory demand, Methods) | computed in memory from `runs/paired/<cell>/gpu.stdout` + `RUNS.tsv` | `supplementary_data.py : arena_table` via `key_numbers.arena_report` |
| Data 7 `LAUNCH_LATENCY` | empty-kernel launch latency | `profiling/launch_latency_l40s.tsv` | `harness/launch_latency.sh` (record) |
| Data 8 `NSYS_KERNEL_SUMMARY` | Nsight Systems attribution | `profiling/nsys_kernel_summary.tsv` | `harness/nsys_kernel_time.sh` (record) |
| Data 9 `NSIGHT_SUMMARY` | header only, by design | `profiling/nsight_summary.tsv` | `harness/roofline_profile.sh` (record) |
| RUNS_columns | column dictionary of Data 1 | `BENCHMARK_SCHEMA.md` | `supplementary_data.py : column_dictionary` |

`figures/Source_Data.xlsx` (by `source_data.py`) has one sheet per manifest row, named
`Fig<figure><letter>` (plus `_pNN` for the S2 pages), filled from `figures/source_data/<stem>.csv`.
Those CSVs are written automatically by `lib/figstyle.write_source_data` every time `save_panel`
runs, by reading the coordinates of every drawn artist back from the matplotlib figure, so the
workbook changes whenever a panel is redrawn.

## 5. Changing how something is presented

| Kind of change | Where | Then run |
|---|---|---|
| one panel's labels, annotations, legend, limits, marker size | the function named in section 3 | that generator, then `composites.py`, then `panels.py` |
| the stratum line (thread count, n) | `figstyle.stratum`; the n each panel passes | every generator |
| the plotting-box size (40 mm), facet gap, canvas margins | `AX_MM`, `_GRID_GAP_IN`, `_MARGIN_IN` in `lib/figstyle.py` | every generator, then `composites.py` (which refuses a page that would print below 5 pt) |
| type size, font stack, the two alphabet hues, phase colours, size ramp, arm encoding (fill + line), marker shapes, spines, ticks, DPI, PDF font embedding, `ratio_ticks` | `lib/figstyle.py` | every generator (or `refresh_all.sh`) |
| which rows enter a panel (build filter, gate-failure cell, `__mdl_*` cells, strata, replicate namespaces); the complexity and work-unit definitions; the estimators | `lib/figdata.py`, `lib/headline.py` | `key_numbers.py` and every generator |
| which panels form a figure, their rows and letters, composite titles, page scale limits | `FIGURES` and `SUPP` in `composites.py` | `composites.py`, `benchmark_map.py`, `crosscheck.py` |
| preview layout | `assemble.py` | `assemble.py` |
| tree montages: tip-label shortening, scale bar, trees per page, panel inches | `trees.py` (`short_label`, `nice_scale`, `draw`, `PER_PAGE`, `PANEL_IN`) | `trees.py` |
| workbook sheet descriptions and order | the `items` list in `supplementary_data.py` | `supplementary_data.py` |
| workbook look (header fill, widths, tab colours) | `style_workbooks.py` | `style_workbooks.py <xlsx>` |

```bash
cd <repository root>
python3 benchmark/scripts/<generator>.py     # rewrites every panel that script owns (seconds to ~1 min)
python3 benchmark/scripts/composites.py      # the lettered figure files (needs pdftoppm)
python3 benchmark/scripts/panels.py          # the contract audit (exit 1 on a violation)
bash benchmark/scripts/refresh_all.sh        # or everything, ~3 min, no GPU
bash scripts/run_tests.sh --quick            # invariants, figure contract, cross-checks, the README map
```

Notes. Run the generators from the repository root: reads resolve from each script's own location,
but `save_panel` writes to the relative default `benchmark/figures/panels` unless `PANEL_DIR` is
set, and `PANEL_DIR=/some/scratch` is the way to draft one script's panels without touching
`figures/`. A generator always rebuilds every panel in its `main()` list. `panels.py` rejects a
literal `fontsize=` or `.grid(True)` in the audited scripts, any PDF type size outside 5/6/7/8 pt
(mathtext sub- and superscripts at 0.7× are allowed) and an embedded DejaVu font, which
`figstyle.apply()` also raises on when Arial, Helvetica and Nimbus Sans are all absent.
`composites.py` refuses a layout whose uniform page scale would print type below 5 pt.
