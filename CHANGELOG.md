# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/). Nothing has been released publicly yet.

## [Unreleased]

The tool is feature-complete and validated; the remaining work before a public release is packaging
and the accompanying manuscript. See [`results/benchmarks/KEY_NUMBERS.md`](results/benchmarks/KEY_NUMBERS.md) for the measured numbers and
`results/benchmarks/KEY_NUMBERS.md` for the current measured numbers (regenerated from data, never
typed by hand).

### Added
- GPU likelihood kernel for amino-acid and DNA single-matrix reversible models, applied to IQ-TREE 3
  as an ~400-line patch across 8 files plus 4 new CUDA sources (`src/`).
- `--gpu-strict` engagement gate: the run fails rather than silently falling back to the CPU.
- On-device Newton branch-length optimisation, removing per-branch kernel-launch overhead.
- Paired benchmark harness: both arms in one job, on one node, from one binary (`scripts/benchmark/`).
- `scripts/lib/paths.sh` + `config/site.env.example` so the repository runs outside its original cluster.
- `scripts/key_numbers.py`, making every number quoted in the docs a build product.

### Validated
- Topology identical (RF = 0) in every recorded run; worst log-likelihood relative difference
  1.893e-06 against a 1e-4 contract.
- Model generality: amino-acid models selected by ModelFinder, plus DNA under GTR+F+G4, HKY+F+G4,
  GTR+F+R4, GTR+F+I+G4 and JC.
- Six GPUs spanning roughly a 19x range in fp64 throughput.

### Known limitations
- At 20 threads the GPU is slower than the CPU for most DNA trees; the payoff case is protein, and
  DNA only above roughly 1e9 tree complexity. See [`results/benchmarks/KEY_NUMBERS.md`](results/benchmarks/KEY_NUMBERS.md).
- One recorded cell fails the bootstrap-support arm of the equivalence gate while remaining exact in
  topology and likelihood; the per-run evidence is distributed under `results/benchmarks/paired/`.
- Profile-mixture, codon, partitioned and `+ASC` models fall back to the CPU (still exact).

## [0.1.0-dev] - unreleased
Pre-release development. No tagged versions yet.
