# Contributing to IQ-TREE3-GPU

> **Primary repository:** <https://github.com/kosticlab/iqtree3-gpu> (authoritative; open issues and
> pull requests there). A synchronized mirror is kept at <https://github.com/terry-b-cho/iqtree3-gpu>.
> Upstream: [IQ-TREE 3](https://github.com/iqtree/iqtree3); report problems in IQ-TREE itself there.

## Reporting a problem

Open an issue at <https://github.com/kosticlab/iqtree3-gpu/issues> and include:

1. The output of `scripts/iqtree3-gpu-verify.sh` (30 seconds, prints the engagement and equivalence
   verdict). This answers most questions immediately.
2. `nvidia-smi`, `nvcc --version`, `gcc --version`.
3. The exact command you ran and the first 40 lines of the `.log` IQ-TREE wrote.

## The one rule that matters

**This tool's contract is algorithmic equivalence to the CPU path, not bit-identity.** Any change
that can alter a tree topology or move a log-likelihood by more than 1e-4 relative is a breaking
change, no matter how much faster it is. A PR that touches the kernel must show:

- `scripts/iqtree3-gpu-verify.sh` passing (RF = 0, logL within tolerance, GPU genuinely engaged), and
- `scripts/kernel_unit_test.sh` passing (partials, theta, logL, df, ddf against an in-file CPU
  reference at ~1e-12).

Run `bash scripts/run_tests.sh --quick` before opening a PR; `--gpu` if you have a device.

## Things that are easy to get wrong

- **`-T N` is not symmetric between the two arms.** On the CPU it is N worker threads over disjoint
  pattern ranges; on the GPU a single launch computes all patterns per node and `-T` drives host-side
  work only. Never describe a comparison as "N-thread CPU vs N-thread GPU".
- **Never claim the GPU ran because the banner says so.** Engagement is proven by GPUSTATS
  (`pct_gpu = 100`, `cpu_calls = 0`), which is what `--gpu-strict` enforces.
- **Benchmark timings come from IQ-TREE's own wall clock**, never from SLURM queue time, and both
  arms must run in one job on one node from one binary.
- **Never overwrite a successful measurement.** `benchmark/runs/paired/<cell>/` is irreplaceable;
  the harness refuses to overwrite a cell whose `RESULT.tsv` has `gate_pass=TRUE` unless `FORCE=1`.
- **Do not hand-edit numbers into the documentation.** They are generated into
  `benchmark/runs/KEY_NUMBERS.md` by `benchmark/figures_tables/tables.py key_numbers`.

## Style

Match the surrounding code. Shell scripts are `bash`, resolve their own location through
`scripts/lib/paths.sh`, and never hardcode an absolute path. Python is standard-library plus
matplotlib/numpy; no new runtime dependencies without discussion.
