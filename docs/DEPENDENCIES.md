# Dependencies and tested hardware

Everything needed to build and run IQ-TREE3-GPU, with the **exact versions this tool was built and
validated against**. Where a version is known to matter, that is stated; where a range is fine, that
is stated too. Nothing here is aspirational — every version below was used to produce the measured
record in `benchmark/runs/`.

---

## 1. Build-time requirements

| Requirement | Version used here | Notes |
|---|---|---|
| **git** | 2.52.0 | any recent version |
| **CMake** | 3.31.2 | **>= 3.18 required** (needs `CMAKE_CUDA_ARCHITECTURES`) |
| **CUDA toolkit** | 12.8 (`nvcc` V12.8.61) | 12.x expected to work; 11.x untested |
| **C++ compiler** | GCC 14.2.0 | must be a version your CUDA release accepts — see the pitfall below |
| **Eigen** | 3.4.0 | header-only; used if installed, otherwise `build_iqtree_gpu.sh` downloads it (2.6 MB, checksum-verified) |
| **Boost** | 1.87.0, headers only | header-only; used if installed, otherwise downloaded by the build script (155 MB once); no compiled Boost libraries needed |
| [**IQ-TREE 3 source**](https://github.com/iqtree/iqtree3) | commit `6401e06905f65a11ccd6e45f369a1987d657d600` | pinned by `build_iqtree_gpu.sh`; the patch is made against exactly this commit |

Resulting binary: `IQ-TREE version 3.1.3 for Linux x86 64-bit`.

### The one pitfall that bites people
**nvcc rejects host compilers newer than it supports.** CUDA 12.8 accepts GCC up to 14.x; a newer
system GCC produces `unsupported GNU version! gcc versions later than N are not supported`. On an
Lmod cluster load **gcc first, then cuda**, so the CUDA module resolves against the right compiler:

```bash
module load gcc/14.2.0 cuda/12.8 cmake/3.31.2     # gcc FIRST
```

`build_iqtree_gpu.sh` checks git, CMake, nvcc and g++ before cloning anything and names whatever is
missing; Eigen and Boost are located or fetched after the clone.

---

## 2. Runtime requirements

The binary dynamically links these. They must be findable **at run time**, not only at build time:

| Library | From | Why |
|---|---|---|
| `libcudart.so.12` | CUDA toolkit | CUDA runtime |
| `libcublas.so.12` | CUDA toolkit | cuBLAS — the `--dating mcmctree` Hessian uses `dsyrk` |
| `libcublasLt.so.12` | CUDA toolkit | pulled in by cuBLAS |
| `libstdc++.so.6` | the GCC used to build | must be **at least as new** as the build compiler's |

Check yours resolve:

```bash
ldd ./iqtree3_gpu_build/iqtree3 | grep -E 'cudart|cublas|stdc'      # every line must show a real path
```

Any line reading `not found` means the CUDA (or GCC) libraries are not on `LD_LIBRARY_PATH`. Fix with
`module load cuda/12.8` or `export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH`.

**Optional, not required by the tool:**

| Tool | Version | Used for |
|---|---|---|
| python3 | 3.9.25 | the exactness checker — **standard library only**, no pip installs |
| R + `ape` | R 4.4.2 | drawing tree figures only; everything else works without R |
| git-lfs | any | only to fetch the source genomes in `benchmark/data/genomes/` (the alignments are ordinary files) |

---

## 3. GPU requirements

**A CUDA GPU with double-precision (fp64) support.** The likelihood kernel is fp64 throughout —
that is not a tuning choice: log-likelihoods here are of order 1e6 and the equivalence gate is 1e-4
absolute, which fp32 (~7 significant digits) cannot hold.

### Compute capability
By default the build covers every card below (`70;75;80;86;89`), so one binary runs natively on any
of them. Pass a single architecture for a smaller, faster-compiling binary:

```bash
./build_iqtree_gpu.sh                               # default: multi-arch, ~14 MB binary
./build_iqtree_gpu.sh ./iqtree3_gpu_build 89        # one card family (L40S here)
```

| Card | Compute capability | Pass |
|---|---|---|
| V100 / V100S | 7.0 | `70` |
| RTX 6000 / RTX 8000 | 7.5 | `75` |
| A100 | 8.0 | `80` |
| A6000 / RTX 3090 / A40 | 8.6 | `86` |
| L40 / L40S / RTX 4090 | 8.9 | `89` |

Building for the wrong architecture is not always an error: the driver can JIT-compile from embedded
PTX, so the binary still runs — just not as native code, and with a JIT cost at start-up. If you care
about performance, build for the card you will actually run on. Verify what a binary contains with:

```bash
cuobjdump --list-elf ./iqtree3_gpu_build/iqtree3 | grep -oP 'sm_\d+' | sort -u
```

### Hardware this tool was developed and benchmarked on
All work was carried out on the Harvard HMS O2 cluster. These are the GPUs that were available to us,
and therefore the only ones any performance claim in this repository can speak to:

| GPU | VRAM | Compute capability | fp64 class |
|---|---|---|---|
| **NVIDIA A100** | **80 GB** | 8.0 | fp64-strong (~9.7 TFLOPS) |
| **NVIDIA L40S** | **48 GB** | 8.9 | fp64-weak (1/64 rate, ~1.4 TFLOPS) |
| **NVIDIA A40** | **48 GB** | 8.6 | fp64-weak |
| **NVIDIA A6000** | **48 GB** | 8.6 | fp64-weak |
| **NVIDIA RTX 8000** | **48 GB** | 7.5 | fp64-weak |
| **NVIDIA V100S** | **32 GB** | 7.0 | fp64-strong (~8.2 TFLOPS) |
| **NVIDIA RTX 6000** | 24 GB | 7.5 | fp64-weak |
| **NVIDIA V100** | 16 GB | 7.0 | fp64-strong (~7.8 TFLOPS) |
| **NVIDIA L40** | 45 GB | 8.9 | fp64-weak |

We did **not** have access to H100/H200, GH200, MI300, or consumer RTX 40/50-series cards, so this
repository makes no claims about them.

**fp64 class matters for this kernel.** Consumer and workstation cards (L40S, A40, A6000, RTX)
deliberately run fp64 at 1/64 of their fp32 rate; datacenter cards (A100, V100/V100S) run it at 1/2.
Since the kernel is entirely double precision, that is a large architectural difference rather than a
detail. The protein path is fp64 compute-bound, so fp64-strong datacenter cards are *expected* to win
for protein; DNA is memory/launch-bound, so that argument does not carry to it. Both are predictions —
which class wins in practice is MEASURED across six cards: by GPU-arm wall time the A100 is fastest,
but an L40S finishes within 1.22x of it at one seventh the nominal fp64 rate, and the fp64-weakest
cards (A40, RTX 8000) are 2.4-3.2x slower. A 19x spread in nominal fp64 produces only a 3.7x spread
in measured wall time (r^2 = 0.75), so the datasheet number is a poor purchasing rule. Nucleotide
work under a fixed model is at or below parity at 20 threads on every card (the per-card runs are
in the run table, `benchmark/runs/RUNS.tsv`, and the within-card speedups in Supplementary Fig. S1D).

### How much VRAM do I need?
The device arena is approximately:

```
slots x patterns x categories x states x 8 bytes,   slots ~ (taxa - 2) + 2
(states = 20 for amino acid, 4 for nucleotide -- so a DNA arena is ~5x smaller for the same tree)
```

For scale: 30 taxa x 233k patterns x 10 rate categories is about **10 GB**. The engine keeps the arena
within 70% of device memory by lowering the rate-category capacity one step at a time (never below the
model's own count) and prints a line saying so; if even that does not fit, the run falls back to the CPU
kernel rather than failing — correct results, no acceleration. There is **no hard limit on taxa, sites,
or alignment size**; the arena is sized from IQ-TREE's own `max_lh_slots`.

---

## 4. What the GPU path supports

Accelerated (identical result to CPU IQ-TREE):

- **Amino acid** (20 state), single-matrix reversible models: LG, WAG, VT, JTT, PMB, Q.\*, Dayhoff, mtMAM, ...
- **Nucleotide** (4 state), reversible models: JC, K80, HKY, TN, GTR and the rest of the reversible DNA set
- `+F` empirical frequencies, `+G` gamma rates, `+R` FreeRate, `+I` invariant sites
- `-m TESTNEW` (ModelFinder), `--ufboot`, `--alrt`, `--dating mcmctree`
- Any thread count (`-T N`)
- `--gpu-newton host|device`: the branch-length Newton loop on the host with GPU derivatives (default;
  the benchmarked configuration) or entirely on the GPU (opt-in; same routine, not benchmarked)

> **Nucleotide GPU support is verified and shipped in the production binary.** The 4-state path (DNA is
> memory/launch-bound, so it uses a different thread mapping) builds
> correct, fully-GPU-engaged trees — RF = 0 vs CPU with logL identical and the GPU genuinely engaged
> (pct_gpu = 100, cpu_calls = 0) across the full n = 4..30 paired sweep (both 1 and 20 threads), and it
> is built by `build_iqtree_gpu.sh`. Measured speedup (fixed model, L40S): at 20 threads nucleotide
> work is usually below parity (median 0.80x over 78 runs, slower in 63 of them); at 1 thread the
> median is 3.19x. **Above roughly 1e9 tree complexity it does pay**, reaching 1.57x, which is where
> the fitted break-even sits (95% CI 7.4e8-1.7e9). Nucleotide is memory- and launch-bound, so the
> fp64-strong-hardware argument does not transfer from protein
> (`benchmark/figures/panels/Fig02a_decision_t20.png`).

Falls back to the CPU kernel automatically — still correct, just not accelerated:

- Codon, morphological and binary data
- Profile-mixture models (C10-C60, LG+C20, ...)
- Partitioned analyses / super-trees, tree mixtures, mixture-length models
- Ascertainment-bias correction (`+ASC`), rooted trees, non-reversible models, site-specific models

Run with `--gpu-strict` to make any such fallback a hard error instead of silent — recommended for
benchmarking, where a silent fallback would mean unknowingly comparing CPU against CPU.

### Known untested paths
Honest limits of what has been exercised, so you know where to be careful:

- **Resume from checkpoint.** Every validation run so far passed `-redo`. A run that resumes from a
  `.ckp.gz` rebuilds the GPU context from the restored tree, which *should* be correct — the context
  is derived entirely from the tree and alignment — but that path has never been executed. If you
  resume a long `-gpu` run, verify the result against a `-redo` run before trusting it.
- **Thread counts.** `-T 1`, `2`, `4`, `8` and `20` are covered by the equivalence matrix; other
  values use the same code path but have not been run individually.
- **`-safe` (SAFE_LH scaling).** Implemented and unit-tested, and covered by one equivalence cell.
  It auto-engages only at >= 2000 taxa, so it is otherwise reachable only by passing `-safe`.
