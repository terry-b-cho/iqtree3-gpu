# IQ-TREE3-GPU — user guide

A CUDA likelihood backend for [**IQ-TREE 3**](https://github.com/iqtree/iqtree3). One promise:

> **The GPU gives you the same tree as CPU IQ-TREE — log-likelihood within 1e-4, identical topology
> (Robinson-Foulds distance 0), matching SH-aLRT/UFBoot support.**

You turn it on with one flag: `-gpu`. Everything else about your command is unchanged. If a run is
ever not equivalent to the CPU, that is a bug — and the tooling below lets you check it yourself
before you trust a result.

---

## 1. What it does

IQ-TREE builds maximum-likelihood phylogenetic trees. The cost is dominated by the likelihood
calculation, evaluated millions of times during model selection (ModelFinder), tree search and
bootstrapping. This backend moves that calculation onto an NVIDIA GPU while reproducing IQ-TREE's own
quantities. It is not a re-implementation: it repoints IQ-TREE's four likelihood callbacks at a CUDA
implementation of the *same* Felsenstein recurrence — same eigen-basis partials, same substitution
matrices, same rate categories, same power-of-two rescaling, analytic branch-length derivatives — and
leaves the tree search untouched.

The equivalence target is **algorithmic**, not bit-level. Bit-identity is unattainable here: CPU
IQ-TREE is not bit-reproducible against *itself* across thread counts, because its derivative
accumulates under an OpenMP reduction. Design details: [`src/GPU_KERNEL_DESIGN.md`](../src/GPU_KERNEL_DESIGN.md).

## 2. Requirements

See **[`DEPENDENCIES.md`](DEPENDENCIES.md)** for exact versions, runtime libraries, compute
capabilities, VRAM sizing and the cards this was tested on. Summary: a CUDA GPU with fp64, CUDA 12.8
+ a gcc your CUDA accepts (14.2.0 here), CMake >= 3.18, Eigen 3.4 and Boost headers. On an Lmod
cluster, **load gcc before cuda**.

## 3. Build (one command)

```bash
module load gcc/14.2.0 cuda/12.8 cmake/3.31.2      # on an Lmod cluster; gcc first
./build_iqtree_gpu.sh                              # both args optional: default is a multi-arch build
# pin one arch for a smaller binary: ./build_iqtree_gpu.sh ./iqtree3_gpu_build 89   # 89=L40S 86=A6000 80=A100 70=V100 75=RTX6000/8000
```

This clones pristine IQ-TREE 3 at the pinned commit, applies `src/iqtree3_gpu.patch`, drops in
`src/gpu_new_files/`, and builds `./iqtree3_gpu_build/iqtree3`. (If Eigen/Boost are not auto-found,
set `EIGEN3_INCLUDE_DIR` and `Boost_INCLUDE_DIR` first.)

## 4. New here? Run the worked example (~2 min)

It builds a tree on the GPU **and** on the CPU from a small real alignment, streams the live output,
proves the two trees are identical (RF = 0), and draws the tree:

```bash
( cd examples && ./run_example.sh )
```

Walkthrough with an annotated transcript: [`examples/README.md`](../examples/README.md).

## 5. Verify on YOUR GPU (30 seconds)

Before any real analysis:

```bash
./scripts/iqtree3-gpu-verify.sh ./iqtree3_gpu_build/iqtree3
```

It checks two things **in order**: first *engagement* (the GPU kernel was actually installed and ran,
not merely that a device was reported), then *equivalence* (`rf_dist: 0`, matching log-likelihood).
Do this once per machine, and again after any rebuild.

## 6. Run your analysis

Take your normal IQ-TREE command and add `-gpu`:

```bash
module load gcc/14.2.0 cuda/12.8            # runtime needs libcudart and libcublas
./iqtree3_gpu_build/iqtree3 -gpu -T 20 \
    -s aln.faa -m TESTNEW --ufboot 1000 --alrt 1000 --prefix mytree
```

On start-up it reports the device, which models are accelerated, and what falls back. At exit it
reports per-call accounting (see section 8).

- **No GPU on the node?** It prints a notice and runs on the CPU, so the same script works everywhere.
- **Everything else is unchanged:** input, model string, `--ufboot`, `--alrt`, `--prefix`, seeds, and
  all output files behave exactly as in stock IQ-TREE.

### `-T N` is not symmetric between the two arms

On the **CPU**, `-T 20` runs 20 worker threads over disjoint pattern ranges. On the **GPU**, packet 0
computes *all* patterns in a single launch per node and the other packets return immediately, so `-T`
drives **host-side work only** — model setup, traversal bookkeeping, and any CPU-fallback models. It
does not mean "20 threads feeding the GPU".

`-T N` does not change the GPU result. But when you report a comparison, write it as
**"CPU, N threads" vs "GPU + N host threads"** — never "N-thread CPU vs N-thread GPU", which implies a
symmetry that does not exist.

## 7. What runs on the GPU (and what safely falls back)

| Model / data | Runs on | Result |
|---|---|---|
| Amino-acid single-matrix reversible models (LG/WAG/VT/JTT/PMB/Q.*/Dayhoff/mtMAM ...) with `+F` `+G` `+R` `+I` | GPU | equivalent |
| `-m TESTNEW` (ModelFinder) | GPU per eligible candidate; CPU for any ineligible one | equivalent |
| `--ufboot`, `--alrt` | GPU | equivalent support values |
| `--dating mcmctree` | GPU (G matrix device-resident, Hessian via cuBLAS `dsyrk`) | equivalent |
| **Nucleotide** reversible models (JC/K80/HKY/TN/GTR ...) with `+F` `+G` `+R` `+I` | GPU **(verified RF=0, GPU engaged across the full n=4..30 paired sweep; shipped in the binary)** | equivalent |
| Codon / morphological / binary data, profile mixtures (C10-C60, LG+C20 ...), partitioned analyses, tree mixtures, mixlen, `+ASC`, `+E`, rooted trees, non-reversible and site-specific models | CPU fallback | equivalent |

You never check eligibility yourself: `-gpu` is always safe. A few runs are *refused* at kernel
selection instead of falling back silently — `-robust-phy` / `-robust-alrt` redefine the tree
log-likelihood as an order statistic over per-**site** likelihoods, which a pattern-indexed engine
cannot produce. You are told why, once, rather than paying for a GPU and getting a CPU run.

Full list, plus known-untested paths (checkpoint resume, thread counts other than 1/2/4/8/20,
`-safe`): [`DEPENDENCIES.md`](DEPENDENCIES.md#4-what-the-gpu-path-supports).

## 8. Proving the GPU actually ran

The `GPU acceleration ACTIVE` banner is printed once, at kernel selection. It proves the kernel was
**installed** — nothing more. Every entry point re-tests eligibility on *every* call and can delegate
to the CPU silently, so a run can print that banner, show non-zero `nvidia-smi` utilisation, and still
evaluate an arbitrary fraction of its likelihoods on the CPU. One ineligible ModelFinder candidate is
enough.

The engine therefore counts every call and prints a machine-readable line at exit:

```
[IQ-TREE3-GPU] GPUSTATS gpu_calls=N cpu_calls=M pct_gpu=... rebuilds=... rebinds=... \
                  ncat_switches=... packets_skipped=... device_errors=... strict=0|1
```

Delegations are also reported by *reason* (`model-is-mixture`, `not-amino-acid`, `+ASC`, ...), so a
partial-GPU run tells you what to change.

**`--gpu-strict`** implies `-gpu` and is **computationally identical** to it — same eligibility test,
same kernels, same launches, same results, no speed difference. The only difference is failure
behaviour: any CPU fallback or device error aborts the run naming the failed condition, and the
process exits non-zero unless accounting shows 100 % GPU. Use it for benchmarking and CI, where a
silent fallback would mean unknowingly comparing CPU against CPU.

```bash
iqtree3 -gpu --gpu-strict -T 20 -s aln.faa -m LG+F+R4 --prefix mytree
```

## 9. Confirm a real run was equivalent (optional, for publications)

Run once on CPU (no `-gpu`) and once on GPU with the **same** `-s`, `-m`, `-seed`, then compare with
the dependency-free checker (system python3, no extra packages):

```bash
python3 scripts/rf_logl_check.py mytree_cpu mytree_gpu mycheck
# -> VERDICT[mycheck]: PASS   rf_dist=0   logl_reldiff=0.0e+00
```

PASS requires RF distance = 0 and log-likelihood relative difference <= 1e-4.

Three deeper checks ship with the repository, in increasing scope:

```bash
sbatch scripts/kernel_unit_test.sh                     # kernel vs in-file CPU reference (~1e-16)
GENUS=g__CAG-269 sbatch scripts/differential.sh        # per-site, per-rate-category logL, pinned model
GENUS=g__CAG-269 sbatch scripts/equivalence_matrix.sh  # models x -T 1/2/4/8/20, support, -safe
```

The differential is the one that catches errors an end-to-end RF check cannot: it compares `-wslr`
per-site log-likelihoods *per rate category* on a fixed topology under a **fully pinned** model. A
wrong `_pattern_lh_cat` interleaving leaves the tree log-likelihood correct while corrupting `+R`/`+I`
EM and empirical-Bayes site rates. Pinning matters: with `-te` alone IQ-TREE still optimises model
parameters, and the two arms then land on slightly different optima — observed per-site differences up
to 4.6e-02 while the *total* logL agreed to 6.5e-08.

## 10. Performance — how it is measured, and what is known

Speed is **actual algorithm runtime** — IQ-TREE's own "Total wall-clock time used" plus per-phase
timings — never cluster queue or wait time.

The benchmark (`benchmark/scripts/harness/paired_benchmark.sh`, submitted via `submit.sh`) is **paired**:

- both arms run **in one job on one node**, so CPU, memory and node are identical by construction;
- both arms run the **same binary**, differing only by `-gpu` (without the flag the swap block never
  runs, so it executes the stock CPU kernel);
- the page cache is warmed before either arm, so the second arm gets no I/O advantage;
- the GPU arm runs `--gpu-strict` and the cell is written INVALID unless engagement is proven
  numerically from GPUSTATS;
- work is recorded in **distinct site patterns**, not sites. For protein the two are close; for DNA
  patterns are bounded by 4^taxa and collapse hard (one simulated set: 195456 sites, 183 patterns).

The set is 12 genera at **12 distinct tree sizes, n = 4..30**, so speedup can be *fitted* against tree
size instead of asserted. That is a property of the benchmark, not a limit of the tool: the device
arena is sized from IQ-TREE's own `max_lh_slots`, and when the arena sized for the run's
rate-category *capacity* would exceed 70% of card memory the engine lowers that capacity one
category at a time, never below the model's own count, then allocates (see "Which card, and how much
memory" below for the rule and its limits).

**What is measured today.** Across **182 paired runs** on the reported build the GPU returned a
topologically identical tree in every case (RF = 0 in 182 of 182; 100 of 182 log-likelihoods
bit-identical, worst relative difference `1.897e-06`). Acceleration depends on the **workflow**, not
the alphabet: every amino-acid run used full ModelFinder and every nucleotide run a pinned model, so
these data cannot separate the two. Amino acid + ModelFinder gains a median **3.73x at 20 threads**
(n = 67) and **34.21x at 1 thread** (n = 17); nucleotide under a fixed model is usually *slower* at
20 threads (median **0.80x**, slower in 63 of 78 runs) and gains **3.19x at 1 thread**. Break-even
sits at a tree complexity of **1.1e7** (95% CI 3.3e6-2.1e7) for amino acid with ModelFinder and
**1.0e9** (7.4e8-1.7e9) for nucleotide under a fixed model. Across six cards nominal double-precision
throughput ranks them only loosely (r^2 = 0.75): an L40S finishes within **1.22x of an A100** at one
seventh its fp64 rate. Current numbers:
[`benchmark/runs/KEY_NUMBERS.md`](../benchmark/runs/KEY_NUMBERS.md); figures:
[`benchmark/figures/panels/`](../benchmark/figures/panels/).

**Choosing hardware.** The kernel is fp64 throughout and, now that it is coalesced, compute-bound in
fp64 for protein (arithmetic intensity ~5 FLOP/B against an L40S balance point of ~1.6). The design
analysis therefore predicts fp64-strong cards (A100 ~9.7 TFLOPS, V100S ~8.2) should beat fp64-weak
ones (L40S ~1.4 TFLOPS, 1/64 rate) by only ~25-30% (Amdahl-capped by the 27.9% device-kernel share). **That prediction has now been MEASURED** — the
card sweep (`benchmark/scripts/harness/card_sweep.sh`) is complete on 6 cards. It will not
transfer to DNA, whose arithmetic intensity is ~1.04 FLOP/B and therefore memory/launch-bound
(`src/` carries the kernel design notes).

## 10a. Practical guidance: when to use it, how many cores to ask for, which card

This section moved here from the manuscript's Discussion (round 8), because it is operational
advice rather than a result. Every number is a key in
[`benchmark/runs/KEY_NUMBERS.md`](../benchmark/runs/KEY_NUMBERS.md).

### When to use it, and when not to

**Amino-acid analyses with model selection, at production thread counts.** The clear case. The
median run gained 3.73-fold at 20 threads (n = 67) and the largest gate-passing analysis in the
benchmark went from 17.6 hours to 3.4 hours. Model selection is both the dominant phase and the
phase that accelerates most, so the benefit lands where these analyses spend their time.

**Nucleotide analyses under a fixed model.** These usually do *not* pay at 20 threads: the median
was 0.80-fold and the device was slower in 63 of 78 runs. Break-even sits at a tree complexity
(branches x patterns x rate categories x states^2) of about 1.0e9 (95% CI 7.4e8 to 1.7e9), which a
gene-scale alignment does not approach: a 50-taxon, 2,000-pattern nucleotide analysis with four
rate categories sits near 1.2e7, two orders below the boundary. Run such analyses on the host. The
statement is conditional: it applies to a fixed model, and nucleotide data with model selection has
not been measured.

**Single-thread or workstation deployment.** Where the host is not already running twenty-way
parallel, the gains are an order of magnitude larger (median 34.2-fold for amino acid at one thread,
n = 17; 3.19-fold for nucleotide). If your alternative is one core rather than a cluster node you are
in a different regime from the 20-thread headline numbers.

### Ask for fewer cores

On shared infrastructure the rationed quantity is processor-time, and the accelerated run leaves
most of its cores idle. For `g__Colicola`, the largest gate-passing analysis, the host arm consumed
348 core-hours and the accelerated arm 5.8, for the identical tree; both were *billed* for twenty
cores times wall time (352 against 69 core-hours), so the saving actually delivered was far smaller
than the saving made available.

**On an amino-acid workflow, request fewer cores alongside `-gpu`.** The accelerated arm kept a
median of 2.0 cores busy; requesting 8 covers the 90th percentile of the benchmark runs and would have
cut that job's bill from 69 to roughly 28 core-hours. Two caveats: 10% of the amino-acid runs kept
8 or more cores busy, so a two-core request (the naive reading of the median) would slow them; and
the advice **does not transfer to nucleotide work**, where the accelerated arm kept a median 10.3
cores busy (90th percentile 17.5) and cutting the request would slow the run outright.

### Which card, and how much memory

Nominal double-precision throughput is a poor purchasing guide. An L40S, with one seventh of an
A100's nominal double-precision rate, finished within 1.22-fold of it on the shared datasets and
ahead of two cards whose double-precision rates are comparable to the A100's; a 19-fold spread in
the specification produced a 3.7-fold spread in measured wall time. For this workload the
datacentre part buys proportionally little. Vendor specifications used here are in
`benchmark/scripts/harness/card_specs.tsv`, one source per row.

**Device memory.** The engine's arena is

    arena bytes = slots x patterns x states x rate-category capacity x 8 B  (+ one small counter per slot)

where *slots* is IQ-TREE's own partial-likelihood slot count plus two (one slot per internal node when
memory allows, so about the number of taxa), and the *capacity* is the largest category count the
run may evaluate: **ten under ModelFinder** (`+R10` is the top of the search), the model's own count
under a fixed model (four for `+G4`). Measured against that denominator the arena costs 8.0 bytes per
entry in every benchmark run. The rule the engine applies at start-up is: if `slots x slot_bytes(capacity)`
exceeds 70% of the card's memory, lower the capacity one category at a time **but never below the
model's own count**, then allocate. The 70% therefore bounds only the *headroom*; a model that needs
more than 70% at its own category count is allocated anyway, and `--gpu-strict` then fails on the
CUDA allocation error rather than falling back to the host. In the benchmark six nucleotide runs had
their capacity reduced (ten to between four and eight categories); five ended inside the budget and
the largest (100 taxa x 3.30 M patterns, GTR+F+G4) reached the four-category floor at 39.65 GB on a
44.4 GiB L40S and completed.

Patterns (millions) that fit the 70% budget, from the formula above:

| device VRAM | AA, ModelFinder (cap 10), 50 taxa | 100 | 200 | AA, fixed +G4 (cap 4), 50 taxa | 100 | 200 | NT, fixed +G4 (cap 4), 50 taxa | 100 | 200 |
|---|---|---|---|---|---|---|---|---|---|
| 16 GB | 0.13 | 0.07 | 0.03 | 0.34 | 0.17 | 0.09 | 1.7 | 0.9 | 0.4 |
| 24 GB | 0.20 | 0.10 | 0.05 | 0.50 | 0.26 | 0.13 | 2.5 | 1.3 | 0.6 |
| 48 GB | 0.40 | 0.21 | 0.10 | 1.01 | 0.51 | 0.26 | 5.0 | 2.6 | 1.3 |
| 80 GB | 0.67 | 0.34 | 0.17 | 1.68 | 0.86 | 0.43 | 8.4 | 4.3 | 2.2 |

For comparison, the largest amino-acid alignment in the benchmark has 170,890 patterns and 66 taxa
(0.17 M patterns, inside every row above at either capacity), and the largest nucleotide set,
4.84 M patterns at 28 taxa, needs about 19 GB at capacity four (30 slots x 4.84 M x 4 x 4 x 8 B), inside
the 48 GB card's budget; the 100-taxon, 3.30 M-pattern set is the one that exceeded it and ran. If your
run reports a capacity reduction at start-up it is telling you that ModelFinder's `+R10` headroom did
not fit; the analysis is unaffected unless the selected model itself needs more categories than the
reduced capacity, in which case the engine allocates the model's count regardless.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `error while loading shared libraries: libcudart.so.12` | `module load cuda/12.8` before running. |
| `error while loading shared libraries: libcublas.so.12` (or `libcublasLt.so.12`) | Same fix — the binary links cuBLAS for the `--dating mcmctree` Hessian. CUDA's `lib64` must be on `LD_LIBRARY_PATH` **at run time**, not just at build time. Check with `ldd ./iqtree3_gpu_build/iqtree3 \| grep -E 'cudart\|cublas'` — no line may read `not found`. |
| `GLIBCXX_3.4.31 not found` | `module load gcc/14.2.0` before running. |
| `-gpu requested but no CUDA device is visible` | No GPU allocated; request `--gres=gpu:1` or run without `-gpu`. |
| Start-up says a model is on CPU fallback | Expected for mixture / DNA / codon / `+ASC` / partitioned; the result is still equivalent. `GPUSTATS` at exit names the reason. |
| `--gpu-strict FAILED: N of M likelihood calls ran on the CPU` | The run was not a pure GPU measurement. The reason lines above it say which condition failed; either drop `--gpu-strict` or choose an eligible model. |
| Slower than the CPU below ~2e7 tree complexity (branches x patterns x rate categories x states^2) - measured for BOTH alphabets; 2e7-1e9 returns ~1-2x, above 1e9 protein returns 3.3-4.7x and DNA 1.1-1.6x | Expected - the GPU cannot amortise launch latency on small problems (an empty kernel launch and synchronisation on the L40S was measured directly; see `benchmark/profiling/launch_latency_l40s.tsv` and Supplementary Fig. S18 for the per-invocation view) |
| Slower than the CPU on an alignment with few distinct patterns | Expected: when patterns are few the per-node launch latency dominates the kernel work (this is the memory/launch-bound regime DNA lives in). Judge size by `Number of distinct site patterns` in the `.iqtree` report, not by the site count. |
| Verifier prints FAIL | Do not use the build; recheck the build and the CUDA/gcc modules, and report it. |

## 12. Where things are

The repository has two parts. The root is the tool; `benchmark/` is the published benchmark behind
the manuscript and is not needed to build, verify or use the tool.

```
README.md · docs/                        the tool: what it is, how to use it, what it needs
build_iqtree_gpu.sh                      one-command build (clone upstream + patch + compile)
src/                                     the GPU patch, new CUDA sources, and the kernel design note
examples/                                bundled small dataset + live worked example  <- start here
scripts/                                 verify (iqtree3-gpu-verify.sh), equivalence checks, run_tests.sh
scripts/lib/paths.sh                     the ONE place the repository root is resolved
config/site.env.example                  the ONE place site settings live (partition, modules, deps)
validation/                              where the validation harnesses write YOUR results
containers/                              Dockerfile, apptainer.def
benchmark/README.md                      the published benchmark: which script makes which figure and table
benchmark/data/                          benchmark alignments (public MAG data; simulated DNA sweep)
benchmark/runs/                          RUNS.tsv, the per-run record, KEY_NUMBERS.md and the derived tables
benchmark/figures/                       every published figure, its Source Data, FIGURE_MANIFEST.tsv
benchmark/scripts/                       the figure and table generators; harness/ is the SLURM benchmark
logs/                                    SLURM job logs land here (gitignored)
```

## Reproducing the published benchmarks

See [`benchmark/REPRODUCING.md`](../benchmark/REPRODUCING.md) and the figure and table map in
[`benchmark/README.md`](../benchmark/README.md). In short:

- **Quick test** - `scripts/iqtree3-gpu-verify.sh` and `examples/run_example.sh` ship in the
  repository and run in minutes.
- **All figures** - `bash benchmark/scripts/refresh_all.sh` regenerates every figure from the
  measured record, which ships in full. No GPU needed.
- **The measurements** - 12 DNA alignments and all 35 amino-acid alignments ship in the
  repository. Fourteen larger simulated DNA alignments (~2.4 GB; 11 of them above GitHub's
  100 MB per-file limit) are deposited on Zenodo as one folder, or can be regenerated locally
  from the recipe in each `benchmark/data/dna/<set>/PROVENANCE.md`.
