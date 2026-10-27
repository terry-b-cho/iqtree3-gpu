# IQ-TREE3-GPU — kernel design (rewrite, 2026-07-21)

## Objective

Run [IQ-TREE 3](https://github.com/iqtree/iqtree3)'s maximum-likelihood tree search with the likelihood engine **actually executing on the
GPU**, performing the **same algorithm** as CPU IQ-TREE, faster.

**Equivalence gate (the real one):** `logL within 1e-4`, `RF = 0`, matching SH-aLRT/UFBoot support.
**Not** bit-identity. Bit-identity is unattainable and meaningless here: `computeLikelihoodDerv`
accumulates under an OpenMP `reduction(+:all_df,all_ddf)` (`phylokernelnew.h:2382`), so CPU IQ-TREE is
not bit-reproducible against *itself* across thread counts. Matching bits would additionally require
reproducing Agner Fog's VCL `exp`/`log` verbatim, every reduction tree shape, and an upstream
associativity typo in `dotProductDualVec` (`phylokernelnew.h:254`). Algorithmic identity is the goal:
same formulation, same quantities, same decisions.

Dropping bit-identity is what makes a fast kernel possible — it frees the memory layout.

## Why the previous kernel was slow (root cause)

1. **Uncoalesced layout.** `partial[(pat*ncat + cat)*20 + s]` puts *state* fastest. Consecutive
   threads working on consecutive patterns read addresses `ncat*20*8 = 800 B` apart, so each warp
   touches 32 separate cache lines instead of a few. Measured 121-449 ms/eval against a ~20 ms
   roofline estimate — consistent with a ~10-30x bandwidth penalty.
2. **Launch fragmentation.** `-T N` split the pattern range across N host threads, each driving its
   own CUDA stream, turning one large node kernel into N small ones plus mutex-guarded shared
   bookkeeping. The GPU is already the parallel device; splitting patterns across host threads
   subtracts parallelism from each launch and adds overhead.
3. **Wrong basis.** Partials were computed in probability space with explicit `P(t)` matrices. IQ-TREE
   stores partials in the **eigen basis**, which is what makes derivatives analytic. The
   probability-space choice forced finite-difference derivatives, which are numerically hopeless for
   a second derivative (`ddf` error 1e5-1e9 at realistic branch lengths) and caused
   `NUMERICAL ERROR: Wrong computeFuncDerv`.

**Withdrawn conclusion.** An earlier per-card sweep on the fragmented, uncoalesced kernel concluded
"launch-latency-bound, prefer high-clock consumer cards over A100". For protein this is now reversed:
the rewritten kernel is fp64 compute-bound (see Roofline). Do not use that card-preference guidance.

## Formulation (mirrors IQ-TREE exactly)

Scope: amino acid (`nstates=20`), single reversible non-mixture matrix, `+F/+G/+R/+I`.

Eigen convention (`modelmarkov.cpp:1558-1562`), both row-major:
`evec[x*20+k] = E[x][k]/sqrt(pi_x)`, `inv_evec[k*20+x] = E[x][k]*sqrt(pi_x)`, `P(t) = evec·e^{Λt}·inv_evec`.

**Partials are stored in the eigen basis:** `L~[c][k] = SUM_x inv_evec[k][x] · L_c(x)`.

Per-branch precompute, per category `c` (`phylokernelnew.h:1017-1031`):

    echild[(c*20 + x)*20 + i] = evec[x*20+i] · exp(eval[i] · rate_c · len)

Internal-internal recursion, per category:

    u_x       = ( SUM_i echild_l[(c*20+x)*20+i] · L~_l[c][i] )
              * ( SUM_i echild_r[(c*20+x)*20+i] · L~_r[c][i] )
    L~[c][k]  = SUM_x inv_evec[k*20+x] · u_x

Tip children pre-collapse to `partial_lh_leaf[state*block + c*20 + x]`; `STATE_UNKNOWN` is forced to
exactly `1.0`. Tip partials themselves are `tip_partial_lh[state*20+i] = SUM_j inv_evec[i*20+j]·state_lk[j]`
(24 states x 20, incl. AA ambiguity B/Z/J).

**Theta** (cached per branch, `phylokernelnew.h:2186-2192`): elementwise `theta[i] = L~_node[i] · L~_dad[i]`
over the whole block. Tip case reuses the same 20-element tip vector for every category.

**Analytic derivatives** (`phylokernelnew.h:2322-2345`) — this replaces finite differences entirely:

    cof  = eval[k] · rate_c
    val0 = exp(cof·t) · prop_c        (prop_c already carries the (1-p_inv) factor)
    val1 = cof · val0
    val2 = cof · val1

    lh_ptn  = SUM_c SUM_k val0·theta       df_ptn = SUM_c SUM_k val1·theta
    ddf_ptn = SUM_c SUM_k val2·theta

    L_p = |lh_ptn| + ptn_invar[p]
    df  = SUM_p freq_p · (df_ptn/L_p)
    ddf = SUM_p freq_p · ( ddf_ptn/L_p - (df_ptn/L_p)^2 )

One pass, versus three full likelihood evaluations for finite differences — the correct method is
also ~3x cheaper on this path.

**Branch logL:** `logL_p = log(|SUM_c SUM_k val0·theta| + ptn_invar[p]) + min_scale_p·log(2^-256)`,
`tree_lh = SUM_p ptn_freq[p]·logL_p`. Note IQ-TREE forms `exp(eval[k]·(rate_c·t))` here but
`exp((eval[k]·rate_c)·t)` in the derivative — a real difference in rounding, irrelevant at 1e-4.

**Scaling.** Power-of-two via `ldexp(x, 256)`, threshold `2^-256`. Default AA path is
**`NORM_LH` (`safe_numeric == false`)** — one `UBYTE` counter per *pattern*, rescale tested once per
pattern on the max over the whole block (`phylotreesse.cpp:101` only forces safe mode at >=2000 taxa
or non-AA/DNA). `SAFE_LH` uses one counter per (pattern, category) with the test inside the category
loop. Constant sites (`ptn_invar != 0`) are never scaled. When combining at a branch, categories are
renormalized to the *minimum* scale count and any category more than one step below is **hard-zeroed**
— a real numerical behaviour to replicate, not a safety net.

## GPU design

**Layout — pattern fastest-varying, so warps coalesce:**

    partial[ (c*20 + s) * npat_pad + p ]        // p contiguous
    theta  [ (c*20 + s) * npat_pad + p ]

`npat_pad` rounds `npat` up to a warp multiple. Consecutive threads read consecutive doubles: one
128 B transaction per warp instead of 32 scattered lines.

**Thread mapping:** one thread per `(pattern, category, state)` => `npat · ncat · 20` threads
(~7M for a typical genus). Each thread computes one output state via a 20-element dot product.
Massive occupancy, fully coalesced.

**Shared memory:** stage the 20x20 `echild_l`, `echild_r` and `inv_evec` blocks (400 doubles = 3.2 KB
each) once per block; every thread reuses them.

**One launch per node covering ALL patterns.** No per-host-thread pattern splitting, no per-thread
stream pool, no mutex. The GPU provides the parallelism.

**Device-resident everything:** partials, theta, eigen matrices, tip partials, `ptn_freq`,
`ptn_invar`, scale counters. Only scalars (`logL`, `df`, `ddf`) cross the bus, via a device reduction.

## Roofline

Per node per (pattern, category): ~2400 FLOP against ~480 B of traffic => arithmetic intensity
~5 FLOP/B. L40S fp64 balance point is ~1.6 FLOP/B, so once coalesced this is **compute-bound in fp64**.

- Node: `npat·ncat·2400` FLOP. For npat=70k, ncat=5 => 840 MFLOP => ~0.6 ms on L40S (1.4 TFLOPS fp64).
- Full tree (~2n nodes, n=20 => ~38): ~23 ms/eval, versus 132 ms/eval measured on the old kernel.

**Measured** (protein paired sweep, complete, 12/12 sizes, L40S — an fp64-weak card): 1.58-4.67x over
the CPU, growing with tree size (n17 = 4.67x). Since the kernel is compute-bound in fp64, fp64-strong
datacenter cards (A100, V100S) do lead, but only by ~25-30% over the L40S class -- **measured** in the completed 6-card sweep and bounded by the 27.9% device-kernel share (nsys). This
confirms the reversal of the withdrawn consumer-card preference above.

fp64 is required: logL magnitudes are ~1e6 and the gate is 1e-4 absolute, which fp32 (~7 significant
digits) cannot hold.

## Verification ladder

Differential harness running both kernels **in the same process**, comparing intermediate buffers
node-by-node, so divergence is caught at its origin rather than as a crash 1000 nodes later:

1. GPU partial vs CPU partial for one node (rel. tol ~1e-12)
2. `theta_all` matches
3. `df`/`ddf` match analytically (thread count pinned — the CPU derivative reduction is thread-order
   dependent)
4. One `optimizeOneBranch` follows the same trajectory
5. End-to-end: `logL` within 1e-4, `RF = 0`, matching support

Plus, non-negotiable after the dead-code incident, every performance run must prove GPU use
**numerically**, not by a banner:

- `GPUSTATS gpu_calls=N cpu_calls=0 pct_gpu=100.00 ...` at exit. The `GPU acceleration ACTIVE` banner
  is printed once at kernel selection and proves only that the kernel was **installed** — every entry
  point re-tests eligibility per call and can delegate to the CPU silently, so a run can print the
  banner, show non-zero `nvidia-smi` utilisation, and still evaluate much of its work on the CPU.
- `--gpu-strict` for benchmarking: computationally identical to `-gpu`, but any fallback or device
  error aborts, so a partly-CPU measurement cannot be produced at all.
- CPU and GPU arms in the **same job on the same node**, using the **same binary** differing only by
  the `-gpu` flag, with the page cache warmed first.
