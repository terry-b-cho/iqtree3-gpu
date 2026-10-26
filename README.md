# IQ-TREE3-GPU

An exact CUDA likelihood backend for [IQ-TREE 3](https://github.com/iqtree/iqtree3), with the
benchmark campaign that measures it. One flag, `-gpu`, moves the phylogenetic likelihood onto an
NVIDIA GPU; everything else about an IQ-TREE run is unchanged, and the result is held to the CPU's:
identical topology (Robinson-Foulds distance 0) and log-likelihood within 1e-4 relative difference,
checked on every recorded run.

| | |
|---|---|
| Use the tool | [`USER_GUIDE.md`](USER_GUIDE.md) (build, flags, verification, troubleshooting), [`DEPENDENCIES.md`](DEPENDENCIES.md) |
| Try it in two minutes | `( cd examples && ./run_example.sh )`, walkthrough in [`examples/README.md`](examples/README.md) |
| Read the measurements | [`results/benchmarks/KEY_NUMBERS.md`](results/benchmarks/KEY_NUMBERS.md), figures in [`results/figures/`](results/figures/README.md), column definitions in [`BENCHMARK_SCHEMA.md`](BENCHMARK_SCHEMA.md) |
| Reproduce them | [`REPRODUCING.md`](REPRODUCING.md): verify a build, regenerate every figure without a GPU, or re-run the campaign |
| Inspect the inputs | [`data/README.md`](data/README.md): 435 source genomes with full accession metadata, 35 amino-acid and 26 nucleotide alignments |
| Cite | [`CITATION.cff`](CITATION.cff); IQ-TREE 3 must be cited alongside this work |

## Build

```bash
# on an Lmod cluster load gcc before cuda, e.g.  module load gcc/14.2.0 cuda/12.8 cmake/3.31.2
./build_iqtree_gpu.sh                      # clones IQ-TREE 3 at the pinned commit, applies src/iqtree3_gpu.patch, builds ./iqtree3_gpu_build/iqtree3
bash scripts/iqtree3-gpu-verify.sh ./iqtree3_gpu_build/iqtree3   # 30 s on a GPU: engagement, RF = 0, log-likelihood
```

Then run IQ-TREE as usual with `-gpu --gpu-strict` (strict mode fails loudly instead of silently
falling back to the CPU):

```bash
./iqtree3_gpu_build/iqtree3 -gpu --gpu-strict -s alignment.faa -m TESTNEW -T 20
```

## What is in this repository

```
src/                 the CUDA backend: a ~400-line patch to IQ-TREE 3 plus four new CUDA/C++ sources, and its design note
build_iqtree_gpu.sh  the one-command build against the pinned upstream commit
scripts/             the benchmark harness (scripts/benchmark/), the analysis and figure scripts, the test suite
data/                every benchmark input, with provenance and metadata (data/README.md)
results/             the measured record: RUNS.tsv (226 paired CPU-vs-GPU runs), per-cell evidence, profiling, figures
examples/            a worked example on a small real alignment, with its expected output
config/              per-site settings (partition, modules, binaries); the tracked example reproduces the measured environment
containers/          Dockerfile and Apptainer definition
```

The tool itself has no dependency on `data/` or `results/`; they are the published evidence behind
the benchmark claims. `bash scripts/run_tests.sh --quick` checks the repository's invariants
without a GPU; `--gpu` runs the equivalence gate on real alignments.

## Status

Pre-release (`VERSION`, [`CHANGELOG.md`](CHANGELOG.md)). The engine is feature-complete and
validated across six GPUs; profile-mixture, codon, partitioned and `+ASC` models fall back to the
CPU path and remain exact. Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md). License: GPL-2.0,
inherited from IQ-TREE ([`LICENSE`](LICENSE)).
