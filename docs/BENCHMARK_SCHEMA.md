# Benchmark data schema - what we record for every run

> **RULE (IQ-TREE3-GPU, non-negotiable).** Every benchmark job is monitored and recorded as one complete
> row in `results/benchmarks/RUNS.tsv` per this schema - no exceptions. A run missing GPU-engagement
> proof or an equivalence result is marked `INVALID`/`gate_pass=NA` and excluded from figures, never
> silently dropped. No number reaches the manuscript that is not a cell in `RUNS.tsv`. This rule is
> restated in the Methods of the accompanying manuscript.

**Every benchmark job is one datapoint = one row.** A reader trusts a figure only if they can audit the
row behind each point: what hardware and software produced it, what the problem was, proof the GPU was
actually used, the exact timings, and proof the GPU tree equals the CPU tree. This file is the data
dictionary; it defines every column, its source, and why it matters. It is published with the data.

## Published artifacts

| file | one row per | purpose |
|---|---|---|
| `results/benchmarks/RUNS.tsv` | **job (datapoint)** | the master record; every column below; append-only, never edited |
| `results/benchmarks/RUNS_summary.tsv` | (data_type, threads, card) | mean +/- sd and n across replicates, for the paper's summary tables |
| `BENCHMARK_SCHEMA.md` (this file) | column | the data dictionary so every field is interpretable |
| per-run raw (`paired/<cell>/`) | file | audit trail behind the row: `RESULT.tsv`, `*.log`, `*.util`, `*.treefile`, `*.iqtree` |

A figure cites `RUNS.tsv`; a summary table cites `RUNS_summary.tsv`; both are reproducible from the raw
per-run artifacts. No number appears in the manuscript that is not a cell in `RUNS.tsv`.

## Schema (grouped by what it establishes)

Legend: **source** = harness (paired benchmark), smi (`nvidia-smi`), slurm (`sacct`), iqlog (IQ-TREE
`.log`), iqfile (`.iqtree`), stats (GPUSTATS line). **status** = HAVE (already emitted) / ADD (to add).

### A. Identity and provenance (so the run is uniquely traceable)
| column | source | status | notes |
|---|---|---|---|
| run_id | slurm | ADD | SLURM job id; the primary key |
| datetime_utc | harness | ADD | run start, ISO-8601 UTC |
| dataset | harness | HAVE (`genus`) | dataset/genus name |
| data_type | harness | ADD | `protein` or `dna` |
| data_source | harness | ADD | `real_MAG` or `alisim_simulated` (with true model). The harness labels by alignment path and mislabelled the `data/dna/dna_*` sets as `real_MAG`; the aggregator sets every `data_type=dna` row to `alisim_simulated` (round 8, R26) |
| seed | harness | ADD | RNG seed (`-seed`), so the search is reproducible |
| command_line | harness | ADD | exact IQ-TREE invocation for each arm |

### B. Software and build (so the exact tool is reconstructable)
| column | source | status | notes |
|---|---|---|---|
| iqtree_version | iqlog | ADD | e.g. IQ-TREE 3.x.x |
| tool_git_commit | harness | ADD | commit that built the binary. Recorded values refer to the development history that preceded this repository's first commit and do not resolve here; `binary_sha256` is the binding identity |
| binary_sha256 | harness | ADD | hash of the `iqtree3` binary actually run. The recorded hashes come from a build whose only difference from the current `src/` is the text of the engine's log banner (renamed to `[IQ-TREE3-GPU]`); a fresh build therefore hashes differently while computing identically |
| build_cuda / build_gcc / sm_archs | harness | ADD | CUDA + compiler versions, compiled GPU archs |
| replay_cache | stats | ADD | `off` (default) / `on`; documents the engine config |

### C. Hardware (so results are attributable to a platform)
| column | source | status | notes |
|---|---|---|---|
| card | smi | HAVE | GPU model (e.g. NVIDIA_L40S) |
| gpu_vram_total_gb | smi | ADD | total device memory |
| cuda_driver_version | smi | ADD | driver version |
| compute_capability | smi | ADD | sm arch (e.g. 8.9) |
| cpu_model | slurm/harness | HAVE | CPU the arms ran on |
| cpu_cores_alloc | slurm | ADD | cores allocated (`-c`); context for `-T` |
| node | harness | HAVE | compute node (both arms; same-node proof) |
| partition | slurm | ADD | e.g. gpu (non-preemptible) vs gpu_requeue |
| preemptible / restarts | slurm | ADD | preemption count; >0 on a timing run flags the row for review |

### D. Problem / alignment (the biology of the datapoint)
| column | source | status | notes |
|---|---|---|---|
| taxa | harness | HAVE | number of sequences (tree size) |
| sites | iqlog | HAVE | alignment columns |
| patterns | iqlog | HAVE | distinct site patterns (the real work unit; DNA patterns bounded by 4^taxa) |
| informative_sites | iqlog | ADD | parsimony-informative sites |
| singleton_sites | iqlog | ADD | singleton sites |
| constant_sites | iqlog | ADD | constant sites (carry the invariant mass) |
| gap_ambiguity_pct | iqlog | ADD | missing-data fraction |
| model | harness/iqfile | ADD | substitution model actually used (e.g. GTR+F+G4, LG+F+R4) |
| rate_categories | iqfile | ADD | number of rate categories (+G/+R); drives per-site work |

### E. Run configuration
| column | source | status | notes |
|---|---|---|---|
| threads | harness | HAVE | `-T N` (never compare across thread budgets) |
| cpu_config / gpu_config | harness | HAVE | human-readable arm labels |
| gpu_flags | harness | ADD | `-gpu`, `--gpu-strict` |
| support_settings | harness | ADD | `--ufboot N` / `--alrt N` if used |
| model_search | derived from command | HAVE | How the substitution model was chosen, recorded explicitly on every row for provenance. **Our benchmark uses `full_ModelFinder_TESTNEW`** for protein (the default IQ-TREE workflow -- ModelFinder is ~91% of the CPU runtime and the phase the GPU accelerates most, so it is the defensible headline benchmark) and `fixed_model` for DNA (`GTR+F+G4`, no search). The classifier also recognises `bounded_ModelFinder` (a restricted `-mset`/`-mrate` set) should it ever be used, so runs are never conflated. |
| mf_flags | derived from command | HAVE | any model-search-restriction flags on the command; `none` for our runs (full ModelFinder / fixed model). Both arms of a cell always use the identical setting, so the speedup is a fair same-model comparison. |

### F. GPU engagement proof (the reader must be able to reject a silent CPU run)
| column | source | status | notes |
|---|---|---|---|
| engaged | harness | HAVE | OK only if banner AND non-zero utilisation; else INVALID |
| gpu_banner | iqlog | ADD | "GPU acceleration ACTIVE" seen |
| pct_gpu | stats | HAVE | % of eligible likelihood calls that ran on the GPU (target 100) |
| gpu_calls / cpu_calls | stats | ADD/HAVE | device vs CPU-fallback call counts (cpu_calls must be 0 under --gpu-strict) |
| kernel_launches | stats | ADD | raw device launches issued |
| newton_ondevice | stats | ADD | on-device Newton loops run (the DNA speedup mechanism) |
| device_errors | stats | ADD | must be 0 |
| util_nsamples / util_mean / util_sd / util_max | smi | ADD (mean HAVE) | nvidia-smi utilisation distribution sampled during the GPU arm (proves the device did work, not just that code dispatched) |
| peak_vram_mb | smi | ADD | peak device memory; supports the VRAM-vs-size scaling claim |

### G. Timing (algorithm runtime only, never queue/wait time)
| column | source | status | notes |
|---|---|---|---|
| cpu_wall_s / gpu_wall_s | iqlog | HAVE | IQ-TREE's own reported wall-clock per arm |
| speedup | harness | HAVE | cpu_wall_s / gpu_wall_s (same node, same binary) |
| cpu_search_s / gpu_search_s | iqlog | HAVE | tree-search phase |
| cpu_modelfinder_s / gpu_modelfinder_s | iqlog | ADD | ModelFinder phase (when `-m TESTNEW`) |
| cpu_bootstrap_s / gpu_bootstrap_s | iqlog | ADD | UFBoot/SH-aLRT phase (when requested) |
| cpu_totalcpu_s | slurm | ADD | CPU-seconds (vs wall) = parallel efficiency of the CPU arm |
| peak_ram_mb | slurm | ADD | peak host RAM (MaxRSS) |

### H. Correctness / equivalence (the GPU tree must equal the CPU tree)
| column | source | status | notes |
|---|---|---|---|
| cpu_logl / gpu_logl | iqlog | HAVE | final log-likelihood, both arms |
| logl_absdiff / logl_reldiff | harness | ADD | |cpu-gpu| and relative; gate is reldiff <= 1e-4 |
| rf_dist | harness | ADD | Robinson-Foulds distance between the two trees; gate is 0 |
| rf_norm | harness | ADD | RF normalised by 2(taxa-3), for cross-size comparability |
| support_max_absdiff | harness | ADD | max abs difference in UFBoot/SH-aLRT support over shared splits |
| gate_pass | harness | ADD | TRUE iff reldiff<=1e-4 AND rf_dist=0 AND support agrees; the single trust bit |

### I. CPU-vs-GPU fairness controls (add to every paired run)

The CPU arm is the baseline the speedup is divided by, so its speed must be shown to be *legitimate* -
not a crippled or contended reference. A reviewer will ask: was the CPU using its best vectorisation,
and was the node loaded?

| column | source | why it matters |
|---|---|---|
| cpu_isa | iqlog | the SIMD instruction set IQ-TREE actually used (SSE/AVX/AVX2/AVX512). A speedup over an SSE baseline on an AVX-512 CPU would be dishonest; IQ-TREE prints this. **Biggest single omission.** |
| cpu_base_ghz / cpu_max_ghz | /proc, lscpu | nominal and turbo clocks; frames the per-core baseline |
| cpu_sustained_ghz | perf/turbostat | actual sustained clock during the CPU arm (turbo may not hold under load) |
| cpu_threads_logical / smt_on | lscpu | physical vs hyperthread cores; `-T 20` on 20 physical != on 10c/20t |
| numa_nodes / thread_pinning | lscpu, env | NUMA layout and whether threads were pinned (affects memory locality) |
| node_exclusive / node_load1 | slurm, /proc/loadavg | was the node exclusive, or were other jobs contending for cores and memory bandwidth during the arms? A shared node inflates the CPU time. |
| cpu_energy_j | RAPL (if readable) | CPU-side joules for the run |
| gpu_power_mean_w / gpu_power_max_w | smi | GPU power draw during the GPU arm |
| gpu_energy_j | smi integral | GPU-side joules; enables **energy-per-tree and perf-per-watt** - "faster" is not the whole story |
| h2d_d2h_mb | stats | host<->device bytes transferred; shows PCIe is not the bottleneck |

### J. GPU-vs-GPU card-sweep attribution (add when the card is the variable)

When only the GPU changes, the reader needs (a) the card's intrinsic specs to explain *why* it wins or
loses, (b) proof it ran at full clocks (not throttled), and (c) control of the **host confound** - a
different GPU usually means a different node and therefore a different host CPU, which changes the
host-side portion of the run.

| column | source | why it matters |
|---|---|---|
| gpu_fp64_tflops / gpu_fp32_tflops | card spec table | the central prediction: protein is fp64 compute-bound, so fp64 throughput should rank the cards |
| gpu_mem_bandwidth_gbs | card spec table | the DNA story is memory-bound; bandwidth should rank the cards there |
| gpu_mem_type / bus_width / vram_gb | smi, spec | HBM vs GDDR, bus width - context for bandwidth |
| gpu_sm_count / gpu_arch | smi, spec | Ampere/Ada/Hopper/Volta; DMMA (fp64 tensor) availability |
| gpu_sm_clock_mean / gpu_mem_clock_mean | smi | **sustained** clocks during the run |
| gpu_temp_max_c / gpu_throttle | smi | REAL slowdown flag (power cap / HW slowdown / thermal). The benign GpuIdle reason (0x1) is masked: a launch-bound workload idles the GPU between kernels by nature, and that must not read as "throttled". Raw per-second samples are kept in each cell's `cardsweep_gpuJ.samples` so the flag is re-derivable |
| gpu_power_cap_w / mig_mode / ecc_on | smi | was it power-capped, MIG-partitioned (a slice != a full card), ECC on (cuts effective bandwidth)? |
| host_cpu_model / host_isa | per node | the host CPU for THIS card (the confound); report it so host-side time is attributable |
| gpu_kernel_time_s | stats/nsight | **GPU-kernel-only** time, isolated from host time, so the card comparison is not polluted by host-CPU differences across nodes |
| perf_per_watt / perf_per_usd | derived | trees-per-joule and trees-per-dollar (card MSRP) - what a "which card should I buy" reader actually needs |
| cuda_runtime / driver (per card) | smi | held constant across cards, or the difference is documented |

### K. Measurement rigor (applies to every datapoint)

| column | source | why it matters |
|---|---|---|
| n_replicates | harness | **a single timing run is not publication-grade.** Run each cell >=3 times; report mean and sd of wall time. Scheduler, thermal and co-tenancy noise are real. |
| wall_sd_s / wall_cv | harness | spread across replicates; a high coefficient of variation flags an unstable measurement |
| warmup_discarded | harness | first run (cold caches / JIT / clock spin-up) discarded |
| self_rf | harness | same seed run twice -> RF must be 0 to itself (determinism check) |
| branch_len_max_absdiff / branch_len_corr | harness | RF=0 proves the *topology* matches; branch lengths must match too (max abs diff and correlation between the two trees) - topology equivalence without branch-length equivalence would be a silent defect |

### L. DNA-vs-AA characterisation (why the two data types behave differently)

Protein (20-state) is compute-bound; DNA (4-state) is memory/launch-bound. The claim is only credible
if each run carries the numbers that show *which regime it is in*. The key instrument: nvidia-smi reports
**two** utilisations - SM (compute) and memory-controller - and logging both makes the regime visible.

| column | source | why it matters |
|---|---|---|
| num_states | iqlog | 4 vs 20 - the root of the whole difference |
| per_pattern_macs | derived | states^2 x rate_categories (16 vs 400 per category) - arithmetic per site |
| pattern_saturation | derived | distinct_patterns / min(sites, states^taxa); DNA saturates fast (bounded by 4^taxa), AA rarely - explains why DNA has little work to parallelise |
| util_sm_mean / util_sm_max | smi | GPU **compute** utilisation (utilization.gpu) - high for AA |
| util_mem_mean / util_mem_max | smi | GPU **memory-controller** utilisation (utilization.memory) - high for DNA. The compute-vs-memory contrast is the DNA/AA story in one pair of columns. |
| achieved_dram_gbs / achieved_fp64_gflops | nsight (profiling-once) | roofline coordinates: DNA near the bandwidth roof, AA near the fp64 roof |
| roofline_bound | nsight (profiling-once) | classification: `compute` / `memory` / `launch` bound |

### M. Per-phase decomposition (which step costs the most, and which the GPU helps the most)

A single run is several phases with very different cost and very different GPU benefit. The overall
speedup is an Amdahl average over them, so the decomposition is what makes the headline number
*explainable and extrapolatable*. Emit one sub-row per phase (long form) per arm, phases:
`setup | modelfinder | tree_search | branch_opt | bootstrap | alrt`.

| column (per phase, per arm) | source | why it matters |
|---|---|---|
| phase_wall_s | iqlog timestamps | time in this phase |
| phase_frac_total | derived | share of the run - identifies the **dominant** phase (ModelFinder is often 79-97%) |
| phase_speedup | derived | CPU/GPU for this phase - identifies which step the GPU helps **most** (AA: likelihood-heavy search/ModelFinder; DNA: the branch_opt Newton loop) |
| phase_gpu_engaged | stats | some phases are host-only (parsimony start, topology bookkeeping) |
| phase_util_sm / phase_util_mem | smi (windowed) | where compute vs memory pressure lives |
| phase_kernel_launches / phase_newton_ondevice | stats | DNA launch-overhead concentrates in branch_opt; shows why on-device Newton is the lever |
| phase_peak_vram_mb | smi | which phase sets the memory high-water mark |

Run-level derived from the decomposition:
| column | source | why it matters |
|---|---|---|
| accelerable_fraction | derived | (CPU-arm ModelFinder wall + CPU-arm tree-search wall) / CPU-arm total wall, from `cpu.log` (paired_benchmark.sh lines ~502-525); fixed-model runs have no ModelFinder phase, which contributes 0. It is a WALL-TIME share of the CPU arm, not a CPU-time share, and is the Amdahl ceiling on overall speedup |
| amdahl_predicted_speedup | derived | 1 / (serial + accel/phase_speedup); predicted vs measured overall is a consistency check |
| dominant_phase / best_speedup_phase | derived | the one-line answer to "what costs most" and "what the GPU helps most" |
| n_likelihood_evals (per arm) | iqlog | hardware-independent work count; equal across arms == apples-to-apples search |
| n_iterations / n_trees_evaluated | iqlog | search-trajectory length; if CPU and GPU differ, the "same operation" claim needs the RF=0 + logL tie to carry it |

## Paths inside the record

`RUNS.tsv` carries two different kinds of path, and the difference matters if you are checking that
this repository contains no machine-specific absolute paths.

**Pointers — stored repository-relative.** `result_path` and `phases_file` locate a cell's evidence
inside this repository. They are *regenerated* by `scripts/aggregate_runs.py` on every
aggregation pass, so they are stored relative to the repository root and resolve in any clone. Join them
to the repository root to get a usable path.

**Provenance — stored exactly as run, and left alone.** `command_line_cpu` and `command_line_gpu`
record the literal command each arm executed during the original campaign, absolute input paths
included, as do the per-cell `cpu.log`, `gpu.log` and `.iqtree` files that IQ-TREE itself wrote.
These are **not configuration**: nothing in the pipeline reads a path out of them, and rewriting
them would mean distributing altered program output while claiming it is the record of what ran.
They are deliberately left byte-for-byte as the campaign produced them, and they are the reason a
grep for an absolute path finds hits under `results/benchmarks/paired/`.

If you re-run the benchmark yourself, your own commands are recorded the same way, in your own
paths, and the pointer columns stay relative.

## Collection guarantee

- Columns in groups **A, C, D, E, F, G, H are MANDATORY in every run.** A run missing engagement proof
  (F) or an equivalence result (H) is recorded `gate_pass=NA` / `engaged=INVALID` and is **excluded from
  figures**, never silently dropped.
- The GPU arm always runs `--gpu-strict`, so a partial-CPU run cannot be produced; `cpu_calls>0` or a
  missing banner marks the row INVALID.
- Timing rows come from **non-preemptible** partitions; `restarts>0` flags a row for review.
- Both arms of a comparison run in **one job on one node** with the **same binary** (differing only by
  `-gpu`); `node` and `binary_sha256` are recorded to prove it.

## Paths inside the record

`command_line_cpu` / `command_line_gpu`, the per-cell logs and the checkpoint files quote the
absolute paths of the machine the campaign ran on. Repository-internal paths are given relative to
this repository's root as installed; the two out-of-tree build directories that held the binaries
are shown under the neutral prefix `/path/to/builds/`. Neither substitution touches a measured value.

## Aggregation for figures

`RUNS_summary.tsv` reports, per (data_type, threads, card): n (replicate count), mean and sd of speedup,
mean pct_gpu, mean util, and the equivalence pass rate. Figures plot per-run points from `RUNS.tsv` and
overlay the summary mean +/- sd. Reported as mean +/- sd with n, including unfavourable datapoints.

## Keeping the data current (the per-job update rule)

The data must stay live and trustworthy as jobs complete. The lifecycle, and the rule for each stage:

1. **Per job -> per cell (atomic).** Each benchmark job writes exactly one `RESULT.tsv` (123 columns) into
   its own `paired/<cell>/` directory, plus `PHASES.tsv`, `*.treefile`, `*.log`, `*.util*`. One job owns
   one cell, so there is no write contention. The GPU arm always runs `--gpu-strict`, so a cell that did
   not truly engage the GPU is written `engaged=INVALID` rather than a misleading number.
2. **Cells -> `RUNS.tsv` (idempotent aggregation).** `scripts/aggregate_runs.py` scans every
   cell, upserts one row per (run_id, cell) into the append-only master `RUNS.tsv`, and regenerates
   `RUNS_summary.tsv`. It writes atomically (temp file + rename) and is safe to run any number of times
   -- re-running only refreshes; it never double-counts. **Rule: run the aggregator after every batch of
   completed jobs** (and it is cheap enough to run on every progress check). During an active campaign,
   run it on a short interval so the master is never stale.
3. **`RUNS.tsv` -> figures/manuscript (committed snapshot).** **Rule: commit `RUNS.tsv` and
   `RUNS_summary.tsv` before cutting any figure or quoting any number**, so every published point traces
   to a versioned snapshot a reader can check out. A figure PDF and the `RUNS.tsv` commit it came from
   move together.

Invariants that make the data trustworthy at any moment:
- A row exists for a job only once its cell wrote a `RESULT.tsv`; a crashed/preempted job leaves no
  half-row (its cell simply has no `RESULT.tsv` until it completes on retry).
- `gate_pass` is `NA` unless engagement AND equivalence were both recorded; such rows are kept (for
  transparency) but excluded from every summary mean and every figure.
- `RUNS.tsv` is append-only in spirit: a re-run of a cell updates that cell's row in place (keyed by
  run_id/cell); superseded engines' cells are moved to a labelled archive dir, never silently deleted.
- A `schema_version` note at the top of `RUNS.tsv` records which schema (this file's revision) produced
  the columns, so the dataset stays interpretable as the schema grows.

## Status

Current `RESULT.tsv` emits the HAVE columns (25). The ADD columns are the remaining work to make each
row fully self-describing (task: extend `paired_benchmark.sh` + an aggregator that appends to
`RUNS.tsv` and regenerates `RUNS_summary.tsv`). Until then, `RUNS.tsv` is assembled from the per-run
artifacts + sacct/nvidia-smi at aggregation time.
