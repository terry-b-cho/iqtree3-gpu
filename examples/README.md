# Worked example — check the tool works, in ~2 minutes

**Purpose: confirm the GPU plug-in runs correctly on your machine and produces the same tree as
standard CPU IQ-TREE.** This is a *functional* check, not a speed test — see
[Why this example isn't about speed](#why-this-example-isnt-about-speed) below.

```bash
./run_example.sh          # from this directory (examples/)
```

or, from the repository root:

```bash
( cd examples && ./run_example.sh )
```

That's it. (If you haven't built the binary yet, see [Setup](../README.md#setup) first.)

Three small datasets from different genera are bundled, so you can run more than one:

```bash
./run_example.sh ""  data/example_Nanosyncoccus.faa
./run_example.sh ""  data/example_Lachnospira.faa
```

---

## What the example does

| Step | What happens | Why it matters |
|---|---|---|
| 1 | Builds a tree **on the GPU** (`-gpu`) | you see the GPU actually engage |
| 2 | Builds the same tree **on the CPU** | the reference |
| 3 | Compares the two | proves **RF = 0** — the GPU gave an identical tree |
| 4 | Draws the tree (PDF/PNG) | you see your result, not just numbers |

## The datasets

Three tiny alignments from different genera, each **8 taxa × ~3,400–4,000 amino-acid columns**
(~30 KB), cut down from real public gut-microbiome MAG alignments so a run finishes in seconds:

| file | source genus | taxa | columns |
|---|---|---|---|
| `data/example_CAG-269.faa` *(default)* | CAG-269 | 8 | 3,997 |
| `data/example_Nanosyncoccus.faa` | Nanosyncoccus | 8 | 3,374 |
| `data/example_Lachnospira.faa` | Lachnospira | 8 | 3,620 |

Full provenance and citation: [`data/PROVENANCE.md`](data/PROVENANCE.md).

> **On the trees:** tips are **bacterial** genomes labelled by the host they were recovered from.
> **Do not expect tips to group by host species.** Gut symbiont clades may or may not track host
> relatedness — recent host switching and genuinely non-codiversifying lineages are biologically real,
> so a tree that doesn't mirror host phylogeny is not a wrong tree. (Whether codiversification occurs
> is a research question, and 8 genomes could not settle it either way.) Use these runs to confirm the
> software works and matches [`expected_output/`](expected_output/) — not to draw biological
> conclusions.

## What you'll see (live output)

The run streams IQ-TREE's own progress. These are the lines worth watching — **actual output from a
real run on an NVIDIA L40S**:

```
 [1/4] GPU run   ->  iqtree3 -gpu -m LG+G4
[IQ-TREE3-GPU] 1 CUDA device(s) visible; device 0: NVIDIA L40S (sm_89)
[IQ-TREE3-GPU] GPU acceleration ACTIVE on NVIDIA L40S.   <- the GPU kernel is installed
[IQ-TREE3-GPU selftest] IQ-TREE logL=-33829.329090  CPU-recompute=-33829.329090  rel=2.796e-15
[IQ-TREE3-GPU selftest] GPU-recompute=-33829.329090  rel-vs-CPUrecompute=0.000e+00
                                                          ^^^ device reproduces the CPU log-likelihood
...ModelFinder / tree search / ultrafast bootstrap progress...
Maximum-likelihood tree: .../examples/out/gpu.treefile
  -> GPU run finished in 34s
  -> GPU kernel engaged (verified)

 [2/4] CPU run   ->  iqtree3 -m LG+G4
  -> CPU run finished in 33s

 [3/4] Exactness check
VERDICT[example]: PASS   rf_dist=0   logl_a=-33476.5026  logl_b=-33476.5026  logl_reldiff=0.00e+00

 [4/4] Tree figure
wrote example_tree.pdf + .png  (8 tips)
```

**`rf_dist=0` is the point:** the GPU and CPU trees are the same tree, and their log-likelihoods agree
within tolerance. The target is *algorithmic* equivalence (RF = 0, logL within 1e-4, matching
support), not bit-identity — see [Validation](../README.md#validation).

> **Read `rf_dist=0` together with the engagement line.** On its own it does not prove the GPU did
> anything: if the GPU kernel is not installed, both runs execute the same CPU kernel and agree
> trivially. `GPU acceleration ACTIVE` (printed only from inside the kernel-swap block) is what
> tells you the device was actually used — the device-count line and the self-test appear either
> way, because the self-test calls the CUDA path directly, bypassing the likelihood dispatch. The
> script checks this for you and prints `GPU kernel engaged (verified)` or a warning.

If your machine has no GPU, IQ-TREE prints a notice and runs on the CPU — the script still completes
(both runs are then CPU, and the check still passes, with the warning above telling you it was not a
GPU test).

## Why this example isn't about speed

This dataset is deliberately tiny (8 taxa, ~4 k columns) so it finishes in seconds. At that size the
GPU has almost nothing to chew on and per-kernel launch overhead dominates, so **GPU ≈ CPU here**
(34 s vs 33 s in the run above). **That is expected and is not a sign that anything is wrong.**

The question this example answers is the one that matters first: *does this new GPU plug-in run
correctly on my machine, and does it give me the same tree as the CPU-only IQ-TREE I already trust?*
The answer should be `rf_dist=0`.

For what the speedup actually is on realistic data, see the measured benchmarks in
[`../results/`](../results/) — reported honestly, per genus, with the exact GPU card, and never
including cluster queue time.

## Your output

Everything lands in `out/`:

| File | What it is |
|---|---|
| `gpu.treefile` | the maximum-likelihood tree, Newick — **this is your result** |
| `gpu.iqtree` | full report: chosen model, log-likelihood, branch supports |
| `gpu.log` | the complete run log |
| `cpu.*` | the same, from the CPU run (identical tree) |
| `example_tree.pdf/.png` | the tree drawn (needs R + `ape`; optional) |

Compare with [`expected_output/`](expected_output/) to confirm your build reproduces ours.

> **Tree figure is optional.** Drawing the tree uses R with the `ape` package. If `Rscript` isn't
> available the script skips that step and still succeeds — a pre-rendered reference is committed at
> `expected_output/example_tree.png`.

## Now use it on your own data

Take the IQ-TREE command you already run and add **`-gpu`**:

```bash
iqtree3 -gpu -T 20 -s your_alignment.faa -m TESTNEW --ufboot 1000 --alrt 1000 --prefix myrun
```

- `-m TESTNEW` lets ModelFinder choose the best-fit model (the example uses a fixed `LG+G4` only to
  stay fast).
- `-T N` sets threads. It **never changes the result**, but it is **not symmetric** between the
  two arms: on the CPU it runs N workers over disjoint pattern ranges; on the GPU a single launch
  computes all patterns per node, so `-T` drives **host-side work only**. See
  [USER_GUIDE](../docs/USER_GUIDE.md) — "`-T N` is not symmetric between the two arms".
- Amino-acid **and** DNA/nucleotide single-matrix reversible models are supported. Ineligible models
  (profile/mixture, codon, partitioned, `+ASC`) fall back to the CPU automatically, still exact — so
  `-gpu` is always safe.

See the [USER GUIDE](../docs/USER_GUIDE.md) for the full reference.
