# benchmark/scripts/harness/ - the benchmarking harness (cluster-specific)

The paired CPU-vs-GPU benchmark and card sweep behind every number in `benchmark/runs/RUNS.tsv`.
**These scripts contain no absolute paths.** Like everything else in the repository they resolve the
repository root at run time through `scripts/lib/paths.sh`, and their `#SBATCH -o/-e` directives are
repo-relative on purpose, so a job launched from any clone writes into that clone's `logs/`.

What *is* specific to our site is the scheduler itself: SLURM with Lmod, the partition and GPU
resource strings, and the card-to-partition mapping. Those are settings rather than paths, and they
are configurable in one place — set `SLURM_PARTITION`, `SLURM_GRES`, `SLURM_MODULES` or `NO_LMOD` in
`config/site.env` (see `config/site.env.example`) and every script here follows. The defaults
reproduce the environment the recorded campaign ran on.

The harness is published for transparency and reproducibility of method. Running it end to end still
assumes a SLURM cluster with NVIDIA GPUs; expect to adapt the submitters for a different scheduler.

| script | role |
|---|---|
| `paired_benchmark.sh` | one benchmark cell: both arms on one node, same binary, `--gpu-strict`, engagement-gated; emits the full-schema `RESULT.tsv` row |
| `submit.sh` | submits the size-sweep cells |
| `card_sweep.sh` | GPU-vs-GPU card sweep: runs paired_benchmark.sh per card + samples group-J fields (clocks, throttle, power) |
| `card_specs.tsv` | static per-card intrinsic specs (fp64 TFLOPS, bandwidth, ...) joined into each card row |
| `gpu_engagement_check.sh`, `validate_gpu_engaged.sh` | engagement probes (GPUSTATS-based) |
| `fresh_build_accept.sh` | release acceptance: fresh clone -> build -> protein + DNA correctness |

Methodology rules and the full column dictionary: `../../BENCHMARK_SCHEMA.md`.
