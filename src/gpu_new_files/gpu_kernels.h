/*
 * gpu_kernels.h -- plain C API to the CUDA likelihood backend (IQ-TREE3-GPU).
 *
 * Included by phylotreegpu.cpp (g++ side). No CUDA headers leak through here, so the
 * IQ-TREE translation units never see nvcc-only constructs and nvcc never sees
 * vectorclass/AVX intrinsics or Eigen. That separation is why this header is plain C.
 *
 * -------------------------------------------------------------------------------------
 * SCOPE of the engine below: amino acids (nstates == 20, SEQ_PROTEIN) and nucleotides
 * (nstates == 4, SEQ_DNA), under a SINGLE reversible non-mixture rate matrix, with
 * +F / +G / +R / +I. Mixture models, SITE_MODEL, ascertainment-bias correction,
 * non-reversible models, codon and morphological data are NOT handled and must fall back
 * to the CPU kernel.
 *
 * The two state counts share ONE engine: every entry point below is state-agnostic and
 * takes nstates from the config. Only the two hottest kernels have a state-count-specific
 * shape, for the reason spelled out in gpu_kernels.cu -- protein is compute-bound in fp64
 * while DNA, with 24x less arithmetic against only 5x fewer bytes, is memory-bound.
 * -------------------------------------------------------------------------------------
 *
 * FORMULATION (mirrors phylokernelnew.h algorithmically, not bit-for-bit).
 *
 * Below, N stands for nstates (20 for protein, 4 for DNA).
 *
 *   Partials are stored in the EIGEN basis, exactly as IQ-TREE stores them:
 *
 *       Ltilde[c][k] = SUM_x inv_evec[k*N + x] * L_c(x)
 *
 *   This is the single most important correctness point. Storing partials in
 *   probability space would force finite-difference derivatives; in the eigen basis the
 *   branch-length derivatives are ANALYTIC, because the only place the branch length t
 *   enters is the diagonal factor exp(eval[k]*rate_c*t) applied to theta.
 *
 *   Per-branch precompute (one "echild slot" per branch being used):
 *
 *       echild[(c*N + x)*N + i] = evec[x*N + i] * exp(eval[i] * rate_c * len)
 *
 *   Bifurcating recursion, per rate category c:
 *
 *       u_x      = ( SUM_i echild_l[(c*N+x)*N+i] * Ltilde_l[c][i] )
 *                * ( SUM_i echild_r[(c*N+x)*N+i] * Ltilde_r[c][i] )
 *       Ltilde[c][k] = SUM_x inv_evec[k*N+x] * u_x
 *
 *   A tip child collapses ahead of time to
 *       partial_lh_leaf[state*block + c*N + x] = SUM_i echild[(c*N+x)*N+i]*tip_lh[state][i]
 *   with the STATE_UNKNOWN row forced to exactly 1.0 over the whole block.
 *
 *   The GPU never interprets an ambiguity code. tip_partial_lh arrives with every IUPAC row
 *   already collapsed by the host (ModelDNA::computeTipLikelihood for DNA, ModelProtein's
 *   B/Z/J rows for amino acids), so the state count changes only the number of rows and the
 *   row pitch of that table -- 24 x 20 for protein, 19 x 4 for DNA.
 *
 * MEMORY LAYOUT -- pattern is the FASTEST-varying index:
 *
 *       partial[(c*N + s) * pattern_stride + p]
 *       theta  [(c*N + s) * pattern_stride + p]
 *       scale  [p]                      (NORM_LH: one UBYTE per pattern)
 *       scale  [c * pattern_stride + p] (SAFE_LH: one UBYTE per pattern and category)
 *
 *   pattern_stride is npat rounded up to a multiple of 32. This layout is the whole
 *   point of the rewrite: consecutive threads work on consecutive patterns, so a warp
 *   touches one or two 128-byte lines instead of 32 scattered ones. The CPU layout
 *   (state fastest) put consecutive patterns 800 bytes apart and cost roughly an order
 *   of magnitude in effective bandwidth.
 *
 * SCALING semantics (replicated exactly, including the parts that look like bugs):
 *
 *   - Threshold 2^-256, applied with ldexp(x, 256). A pattern is rescaled when the
 *     maximum |partial| is BELOW the threshold AND ptn_invar[p] == 0. Constant sites are
 *     never scaled. Note there is deliberately no "and nonzero" guard: an all-zero block
 *     is still "rescaled" (a no-op on the values) and still bumps the counter, because
 *     that is what IQ-TREE does.
 *   - NORM_LH (safe_numeric == 0, the default path for BOTH 20 and 4 states -- see
 *     phylotreesse.cpp:102-103, which excludes num_states 4 and 20 alike from the
 *     "force safe" clause): ONE counter per pattern, and the
 *     underflow test happens ONCE PER PATTERN on the maximum over the WHOLE block, i.e.
 *     over all categories at once.
 *   - SAFE_LH (safe_numeric == 1): one counter per (pattern, category), test inside the
 *     category loop.
 *   - The tip-tip (cherry) case performs NO scaling at all and zeroes the counters.
 *   - When two subtrees meet at a branch (theta), the per-category scale counts are
 *     renormalised to their MINIMUM: a category exactly one step above the minimum has
 *     its theta multiplied by 2^-256, and a category MORE than one step above is
 *     HARD-ZEROED. That is a real numerical behaviour of IQ-TREE, not a safety net, and
 *     it is reproduced here.
 *
 * ANALYTIC DERIVATIVES (never finite differences):
 *
 *       cof  = eval[k] * rate_c
 *       val0 = exp(cof*t) * prop_c      (prop_c ALREADY carries the (1 - p_invar) factor)
 *       val1 = cof * val0
 *       val2 = cof * val1
 *       lh_ptn/df_ptn/ddf_ptn = SUM over (c,k) of val0/val1/val2 * theta
 *       L_p  = |lh_ptn| + ptn_invar[p]
 *       df   = SUM_p freq_p * (df_ptn / L_p)
 *       ddf  = SUM_p freq_p * (ddf_ptn / L_p - (df_ptn / L_p)^2)
 *
 * BRANCH LOG-LIKELIHOOD:
 *
 *       logL_p  = log(|SUM_(c,k) val0*theta| + ptn_invar[p]) + min_scale_p * log(2^-256)
 *       tree_lh = SUM_p ptn_freq[p] * logL_p
 *
 * All device buffers stay resident; only the scalars (logL, df, ddf) come back, via a
 * two-stage device reduction with a FIXED block count, so the result is bit-reproducible
 * run to run for a given problem size.
 */
#ifndef IQTREE_GPU_KERNELS_H
#define IQTREE_GPU_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------
 * Device introspection
 * ---------------------------------------------------------------------------------- */
int         iqtree_gpu_device_count(void);
const char* iqtree_gpu_device_name(void);

/* Free and total global memory of device 0, in bytes. Either pointer may be NULL. Returns 0
   on success. Used by the integration layer to bound the ncat capacity below. */
int         iqtree_gpu_device_mem(size_t *free_bytes, size_t *total_bytes);

/* Human-readable description of the last failure from any iqtree_gpu_lik_* call. */
const char* iqtree_gpu_lik_last_error(void);

/* ------------------------------------------------------------------------------------
 * Likelihood engine
 * ---------------------------------------------------------------------------------- */

/* Opaque engine handle. One per tree/alignment/model triple. */
typedef struct IqtreeGpuLik IqtreeGpuLik;

typedef struct {
    int nstates;         /* 20 (amino acids) or 4 (DNA); anything else is rejected   */
    int ncat;            /* number of rate categories (site_rate->getNRate())        */
    int npat;            /* number of alignment patterns carrying ptn_freq           */
    int num_tip_states;  /* rows of tip_partial_lh, i.e. aln->STATE_UNKNOWN + 1:
                            24 for AA (20 + B,Z,J + unknown), 19 for DNA (4 + the
                            twelve IUPAC codes + three unreachable-but-dense rows +
                            unknown). The table MUST stay dense: the row index IS the
                            state value, so rows the parser never emits still exist. */
    int state_unknown;   /* row index of STATE_UNKNOWN; 23 for AA, 18 for DNA        */
    int safe_numeric;    /* 0 = NORM_LH (the default for both 4 and 20), 1 = SAFE_LH */
    int echild_slots;    /* number of per-branch precompute slots the caller needs   */
    int tip_slots;       /* number of leaves whose state columns are held on device  */
    int device;          /* CUDA device ordinal                                      */
    int theta_slots;     /* independent theta buffers; 0 is read as 1 (see below)    */
    int ncat_cap;        /* categories to SIZE the arena for; < ncat is read as ncat */
} IqtreeGpuLikConfig;

/* WHY THE ARENA IS SIZED FOR A CAPACITY IN CATEGORIES.  Every buffer here is indexed
   buf[(cat*nstates + state)*pattern_stride + pattern], so a buffer allocated for ncat_cap
   categories contains the layout for any ncat <= ncat_cap as an EXACT PREFIX: nothing is
   re-indexed, only fewer (cat,state) planes are touched.  That makes a change of category
   count a pointer-arithmetic update rather than a reallocation, which matters because
   ModelFinder (-m TESTNEW) sweeps +G4 and +R2..+R10 across ~1200 candidate models, i.e. it
   changes ncat hundreds of times per run.  Rebuilding the engine for each of those meant a
   cudaFree + cudaMalloc + cudaMemset of the whole multi-GB arena every time.
   Use iqtree_gpu_lik_set_ncat() to re-point an existing engine at a smaller (or equal)
   category count; anything above ncat_cap still needs a fresh engine. */

/* WHY MORE THAN ONE THETA SLOT.  IQ-TREE keeps exactly one theta buffer (theta_all) and
   guards its reuse with theta_computed, which is set at the end of computeLikelihoodDerv
   and cleared at five specific sites.  computeLikelihoodBranch, by contrast, never touches
   theta_all at all -- it forms its per-category products on the fly.  The GPU engine wants
   theta for BOTH paths, so the integration layer gives the branch path a second, private
   theta slot; that way a computeLikelihoodBranch call cannot invalidate the theta that
   theta_computed is still promising to computeLikelihoodDerv / computeLikelihoodFromBuffer.
   Reproducing that lifetime exactly is the difference between correct Newton iterations and
   a stale-theta bug that looks like a convergence failure. */

/* One child of a multifurcating (degree > 3) node. */
typedef struct {
    int                  echild_slot; /* slot holding this child's echild matrices     */
    int                  is_tip;      /* nonzero -> use tip_slot, else d_partial       */
    int                  tip_slot;    /* leaf state column, when is_tip                */
    const double*        d_partial;   /* device partial of the child, when !is_tip     */
    const unsigned char* d_scale;     /* device scale counters of the child, !is_tip   */
} IqtreeGpuChild;

/* Lifecycle. Returns 0 on success, negative on failure (see iqtree_gpu_lik_last_error). */
int  iqtree_gpu_lik_create(IqtreeGpuLik** out, const IqtreeGpuLikConfig* cfg);
void iqtree_gpu_lik_destroy(IqtreeGpuLik* ctx);

/* ---- Launch accounting (STEP 1 of the DNA launch-overhead work) ----
   Process-wide counters, not per-engine: kernel_launches is every real CUDA kernel launch
   this backend has issued; graph_replays is every replay of a captured CUDA graph. Together
   they confirm the "millions of tiny launches" diagnosis for DNA and MEASURE any launch-
   reduction fix (kernel_launches must fall, graph_replays must rise). The GPUSTATS line in
   phylotreegpu.cpp prints them. reset() zeroes both. */
long long iqtree_gpu_lik_launch_count(void);
long long iqtree_gpu_lik_graph_replay_count(void);
void      iqtree_gpu_lik_reset_launch_count(void);

/* ---- CUDA-graph capture / replay (STEP 2) ----
   Opaque handle to an instantiated, replayable capture of a launch DAG. Capture records every
   kernel launch issued between capture_begin() and capture_end() -- unchanged kernels, order,
   arguments and device pointers -- so a replay is equivalent BY CONSTRUCTION to re-issuing the
   launches, at ~1 driver call instead of one per launch. See the DESIGN NOTE and the validity/
   invalidation contract in gpu_kernels.cu. Every function returns 0 on success, negative on
   failure. NO synchronous CUDA call (blocking memcpy/memset, cudaDeviceSynchronize, cudaMalloc)
   may occur between begin and end; capture the partial + theta sweep, and read results back
   AFTER capture_end/replay, never inside the captured region. */
typedef struct IqtreeGpuGraph IqtreeGpuGraph;
int  iqtree_gpu_lik_graph_capture_begin(IqtreeGpuLik* ctx);
int  iqtree_gpu_lik_graph_capture_end(IqtreeGpuLik* ctx, IqtreeGpuGraph** out);
int  iqtree_gpu_lik_graph_launch(IqtreeGpuLik* ctx, IqtreeGpuGraph* graph);
void iqtree_gpu_lik_graph_free(IqtreeGpuGraph* graph);

/* Selects the theta buffer used by every subsequent theta_*, branch_logl*, derivatives and
   download_theta / download_buffer_scale call. Slots are independent; nothing else in the
   engine is per-slot. Returns 0 on success. */
int iqtree_gpu_lik_set_theta_slot(IqtreeGpuLik* ctx, int slot);

/* Re-points the engine at a new rate-category count WITHOUT touching a single allocation.
   Recomputes block / partial_elems / scale_elems for `ncat` and fails (leaving the engine
   untouched) when ncat exceeds the capacity the arena was built for.

   THE CALLER OWNS TWO INVARIANTS AFTERWARDS:
     (1) every device partial and scale buffer still holds values computed at the OLD
         category count. Their meaning changes with ncat, so the caller must invalidate
         them -- in IQ-TREE terms, clearAllPartialLH() plus theta_computed = false -- exactly
         as it would after a full rebuild. The engine zeroes theta and buffer_scale itself so
         that a missed invalidation degrades to a non-finite likelihood rather than to a
         plausible wrong one, but it cannot reach the caller's slot table.
     (2) the rate/proportion vectors are per-category DATA, not just capacity. set_ncat drops
         them; iqtree_gpu_lik_set_rates() must be called again before any kernel that reads
         them, and the engine enforces that with a loud error rather than reading stale
         categories. */
int iqtree_gpu_lik_set_ncat(IqtreeGpuLik* ctx, int ncat);

/* ARENA LIFETIME vs TREE BINDING.  Everything iqtree_gpu_lik_create() allocates is a function
   of GEOMETRY alone -- npat, ncat_cap, nstates, safe_numeric, echild_slots, tip_slots and
   num_tip_states -- and none of that mentions which PhyloTree the numbers came from.  The
   binding to a particular tree lives entirely in the CALLER (the slot table derived from
   central_partial_lh) plus a handful of device buffers that hold values computed FOR that tree.
   Rebind detaches the second without touching the first:

     - d_rates / d_props are zeroed and marked STALE, so every kernel that reads them fails
       through the same rates_ready guard that set_ncat uses rather than reading the old
       model's categories;
     - d_theta and d_bufscale, which describe one branch of the OLD tree, are zeroed, so a
       caller that forgets to clear its theta_computed flag gets log(0) = -inf instead of a
       plausible number;
     - d_echild and the collapsed d_leaf tables, which hold exp(eval*rate*len) for the OLD
       tree's branch lengths, are zeroed (branch_precompute rewrites them per branch anyway);
     - the lazy scratch (d_u, d_sitelh, d_lhcat) is zeroed for the same reason.

   WHAT REBIND DOES NOT TOUCH, AND WHY THE CALLER OWNS IT:

     (1) the pooled partial / scale buffers handed out by iqtree_gpu_lik_alloc_partial().  They
         are the multi-GB part, and zeroing them would cost exactly what a rebuild costs.  They
         still hold the OLD tree's partials, so the caller must invalidate its own "this slot is
         current" bookkeeping just as thoroughly as it would after a destroy+create -- in
         IQ-TREE terms clearAllPartialLH() plus theta_computed = false, which pushes every
         neighbour into traversal_info and rewrites every device slot before it is read.

     (2) d_tip_states, the per-leaf state columns.  They are a property of the ALIGNMENT, not of
         the tree, and re-uploading them is O(tip_slots * npat) -- the single most expensive part
         of a rebuild.  The engine keeps them, which is only sound if the new tree is on the SAME
         alignment with the SAME leaf-slot numbering.  The engine cannot see either, so the
         caller MUST verify it (identical Alignment object, identical leaf id -> sequence
         mapping) and use a full destroy+create otherwise.  Two trees with equal npat and equal
         leaf counts but different alignments would otherwise alias silently, which is the worst
         failure this file can produce.

     (3) the G matrix.  Its rows are indexed by BRANCH ID of one particular tree, so it is not
         rebindable at all; the caller must rebuild instead of rebinding whenever the MCMCTree
         export is active.  Rebind reports an error and changes nothing in that case.

   After a successful rebind the caller must re-upload the model exactly as after a create:
   set_eigen, set_rates, set_tip_partials, set_pattern_data. */
int iqtree_gpu_lik_rebind(IqtreeGpuLik* ctx);

/* Geometry helpers (all derived from the config; handy for the integration layer). */
int    iqtree_gpu_lik_pattern_stride(const IqtreeGpuLik* ctx); /* npat rounded to 32   */
int    iqtree_gpu_lik_block(const IqtreeGpuLik* ctx);          /* ncat * nstates       */
int    iqtree_gpu_lik_nstates(const IqtreeGpuLik* ctx);        /* 20 or 4              */
int    iqtree_gpu_lik_ncat(const IqtreeGpuLik* ctx);           /* current ncat         */
int    iqtree_gpu_lik_ncat_cap(const IqtreeGpuLik* ctx);       /* ncat the arena fits  */
size_t iqtree_gpu_lik_partial_elems(const IqtreeGpuLik* ctx);  /* block*pattern_stride */
size_t iqtree_gpu_lik_partial_bytes(const IqtreeGpuLik* ctx);
size_t iqtree_gpu_lik_scale_elems(const IqtreeGpuLik* ctx);    /* stride*(safe?ncat:1) */
size_t iqtree_gpu_lik_scale_bytes(const IqtreeGpuLik* ctx);
/* Same two, but for the ALLOCATED capacity -- what a pooled slot actually costs in VRAM. */
size_t iqtree_gpu_lik_partial_bytes_cap(const IqtreeGpuLik* ctx);
size_t iqtree_gpu_lik_scale_bytes_cap(const IqtreeGpuLik* ctx);

/* Bytes one pooled slot (partial + scale counters) would occupy for a hypothetical geometry,
   without creating anything. Exposed so the integration layer can size the ncat capacity
   against device memory without duplicating the pattern-stride rule.

   nstates is NOT optional and has no default: a slot is ncat*nstates*stride doubles, so
   omitting it (as this function did while the engine was amino-acid-only) over-estimates a
   DNA arena by exactly 5x and silently shrinks the category capacity for no reason. That is
   the one place in this API where forgetting the state count is invisible at runtime. */
size_t iqtree_gpu_lik_slot_bytes(int npat, int ncat, int nstates, int safe_numeric);

/* ---- one-time / per-model-update uploads ---- */

/* eval[N], evec[N*N] (evec[x*N+i]), inv_evec[N*N] (inv_evec[k*N+x]), N == cfg.nstates. */
int iqtree_gpu_lik_set_eigen(IqtreeGpuLik* ctx, const double* eval,
                             const double* evec, const double* inv_evec);

/* rates[ncat] = site_rate->getRate(c); props[ncat] = site_rate->getProp(c).
   props ALREADY includes the (1 - p_invar) factor -- do not apply it again. */
int iqtree_gpu_lik_set_rates(IqtreeGpuLik* ctx, const double* rates, const double* props);

/* tip_partial_lh[state*nstates + i], num_tip_states rows. */
int iqtree_gpu_lik_set_tip_partials(IqtreeGpuLik* ctx, const double* tip_partial_lh);

/* ptn_freq[npat] and ptn_invar[npat]. Both are zero-extended to pattern_stride here,
   which is what makes the padded lanes contribute nothing to any reduction. */
int iqtree_gpu_lik_set_pattern_data(IqtreeGpuLik* ctx, const double* ptn_freq,
                                    const double* ptn_invar);

/* States of one leaf, npat entries. Padding up to pattern_stride is filled with
   state_unknown, matching how IQ-TREE pads its SIMD lanes. */
int iqtree_gpu_lik_set_tip_states(IqtreeGpuLik* ctx, int tip_slot, const int* states);

/* ---- device buffers for node partials / scale counters ----
   Sized for ncat_cap, not for the current ncat, so that a set_ncat within the capacity never
   invalidates a buffer the caller is already holding. */
double*        iqtree_gpu_lik_alloc_partial(IqtreeGpuLik* ctx);
unsigned char* iqtree_gpu_lik_alloc_scale(IqtreeGpuLik* ctx);
void           iqtree_gpu_lik_free(IqtreeGpuLik* ctx, void* dptr);

/* Copies for the differential harness and the unit test. */
int iqtree_gpu_lik_download_partial(IqtreeGpuLik* ctx, double* h_dst, const double* d_src);
int iqtree_gpu_lik_upload_partial  (IqtreeGpuLik* ctx, double* d_dst, const double* h_src);
int iqtree_gpu_lik_download_scale  (IqtreeGpuLik* ctx, unsigned char* h_dst,
                                    const unsigned char* d_src);
int iqtree_gpu_lik_upload_scale    (IqtreeGpuLik* ctx, unsigned char* d_dst,
                                    const unsigned char* h_src);
int iqtree_gpu_lik_download_theta  (IqtreeGpuLik* ctx, double* h_dst);
/* buffer_scale[p] = min_scale_p * log(2^-256), pattern_stride entries. */
int iqtree_gpu_lik_download_buffer_scale(IqtreeGpuLik* ctx, double* h_dst);

/* ---- per-branch precompute ----
   Fills echild slot `slot` for branch length `len`, and (when want_leaf) the collapsed
   partial_lh_leaf table for a tip child hanging off that branch. Must be called before
   any partial kernel that references the slot. */
int iqtree_gpu_lik_branch_precompute(IqtreeGpuLik* ctx, int slot, double len, int want_leaf);

/* ---- device-len precompute for CUDA-graph replay (STEP 2) ----
   As iqtree_gpu_lik_branch_precompute, but the length is read on-device from
   d_branch_len[branch_id] instead of passed by value, so a recorded launch is frozen only to
   the topological branch_id and replays correctly after new lengths are uploaded. Use ONLY
   while recording a full-traversal graph. iqtree_gpu_lik_set_branch_lengths must have been
   called (with a buffer sized to cover branch_id) before the graph is replayed. */
int iqtree_gpu_lik_branch_precompute_dev(IqtreeGpuLik* ctx, int slot, int branch_id, int want_leaf);

/* Upload the tree's current branch lengths, indexed by IQ-TREE branch id, into the device
   buffer k_echild_dev reads. Call once per evaluation, just before replaying a full-traversal
   graph (a blocking H2D on the default stream, so NEVER between capture_begin and capture_end).
   Lazily (re)allocates when nbranch grows. */
int iqtree_gpu_lik_set_branch_lengths(IqtreeGpuLik* ctx, const double* lens, int nbranch);

/* ---- Newton-loop replay path (STEP 2) ----
   The per-iteration derivative (and logL-only) evaluations of optimizeOneBranch are a fixed
   3-kernel sequence over engine-global buffers, whose only per-iteration input is the candidate
   branch length. set_eval_len uploads that length to a device scalar; *_dev_issue issues the
   sequence reading it (this is what a captured graph records/replays); *_readback copies the
   scalar result(s) back. No G row is produced (callers wanting one use derivatives_g). */
int iqtree_gpu_lik_set_eval_len(IqtreeGpuLik* ctx, double len);
int iqtree_gpu_lik_derivatives_dev_issue(IqtreeGpuLik* ctx);
int iqtree_gpu_lik_derivatives_readback(IqtreeGpuLik* ctx, double* out_logl,
                                        double* out_df, double* out_ddf);
int iqtree_gpu_lik_branch_logl_dev_issue(IqtreeGpuLik* ctx, int save_log_value);
int iqtree_gpu_lik_branch_logl_readback(IqtreeGpuLik* ctx, double* out_logl);

/* ---- On-device Newton loop (STEP 3) ----
   Runs an entire per-branch Newton-Raphson optimisation on the GPU inside one conditional-WHILE
   graph, so only the final optimised length crosses the bus. build() (re)instantiates the graph
   for the current geometry/theta slot -- call it after selecting GPU_THETA_NEWTON and whenever
   ncat/geometry change; optimize() runs it from xguess over [x1,x2] with tolerance xacc and the
   maxNRStep cap, mirroring Optimization::minimizeNewton exactly, and returns the converged length
   (out_error != 0 means a non-finite derivative forced a stop -- caller should use the host
   optimiser). Requires CUDA >= 12.4 conditional graph nodes (built with 12.8). newton_count() is
   the number of on-device loops run (each is one graph launch, i.e. one branch fully optimised). */
int  iqtree_gpu_lik_newton_build(IqtreeGpuLik* ctx);
int  iqtree_gpu_lik_newton_optimize(IqtreeGpuLik* ctx, double x1, double x2, double xacc,
                                    int maxNRStep, double xguess, double* out_optx, int* out_error);
void iqtree_gpu_lik_newton_free(IqtreeGpuLik* ctx);
long long iqtree_gpu_lik_newton_count(void);

/* ---- partial-likelihood kernels: the four cases of computePartialLikelihood ----
   Each writes d_out (partial_elems doubles) and d_scale_out (scale_elems bytes) with
   ONE launch per node covering ALL patterns. */

/* Both children are leaves. No rescaling is performed and the counters are zeroed,
   exactly as in IQ-TREE's cherry case. */
int iqtree_gpu_lik_partial_tip_tip(IqtreeGpuLik* ctx,
                                   double* d_out, unsigned char* d_scale_out,
                                   int slot_left,  int tip_left,
                                   int slot_right, int tip_right);

/* Left child is a leaf, right child internal. Counters start as a copy of the right
   child's, then get bumped where the pattern underflows. */
int iqtree_gpu_lik_partial_tip_internal(IqtreeGpuLik* ctx,
                                        double* d_out, unsigned char* d_scale_out,
                                        int slot_left, int tip_left,
                                        int slot_right, const double* d_right,
                                        const unsigned char* d_scale_right);

/* Both children internal. Counters start as left+right. */
int iqtree_gpu_lik_partial_internal_internal(IqtreeGpuLik* ctx,
                                             double* d_out, unsigned char* d_scale_out,
                                             int slot_left,  const double* d_left,
                                             const unsigned char* d_scale_left,
                                             int slot_right, const double* d_right,
                                             const unsigned char* d_scale_right);

/* Degree > 3. Children are folded in one at a time and the underflow test runs after
   EVERY child, on the running product in the x (non-eigen) basis -- that is IQ-TREE's
   ordering, and it is observable through the scale counters. The inv_evec rotation is
   applied once at the end and is itself never rescaled. */
int iqtree_gpu_lik_partial_multifurcating(IqtreeGpuLik* ctx,
                                          double* d_out, unsigned char* d_scale_out,
                                          int nchild, const IqtreeGpuChild* children);

/* ---- theta (cached per branch) ----
   theta[i] = Ltilde_node[i] * Ltilde_dad[i] over the whole block, plus the scale
   renormalisation described at the top of this header. */
int iqtree_gpu_lik_theta_internal(IqtreeGpuLik* ctx,
                                  const double* d_node, const unsigned char* d_scale_node,
                                  const double* d_dad,  const unsigned char* d_scale_dad);

/* dad is a leaf: the same nstates-element tip vector is reused for every category, and only
   the node-side scale counters exist. */
int iqtree_gpu_lik_theta_tip(IqtreeGpuLik* ctx, int tip_slot,
                             const double* d_dad, const unsigned char* d_scale_dad);

/* ---- branch quantities from theta ----
   Both consume the theta buffer left behind by the calls above. */
int iqtree_gpu_lik_branch_logl(IqtreeGpuLik* ctx, double len, double* out_logl);

/* Same computation, but it also hands back the PER-PATTERN quantities that IQ-TREE stores
   in _pattern_lh / _pattern_scaling / _pattern_lh_cat. They are not a debugging nicety:
   SH-aLRT and UFBoot resample _pattern_lh, and the +R / +I EM optimisers read
   _pattern_lh_cat, so a branch kernel that returns only the scalar cannot reproduce the
   support values. The values already exist per thread before the reduction, so producing
   them costs one extra store each.
     h_ptn_lh      [npat]      -- log(L_p) + min_scale_p*log(2^-256) when save_log_value,
                                  otherwise the raw L_p = |SUM val0*theta| + ptn_invar[p]
     h_ptn_scaling [npat]      -- min_scale_p*log(2^-256)
     h_ptn_lh_cat  [ncat*npat] -- CATEGORY-MAJOR, h_ptn_lh_cat[c*npat + p], the per-category
                                  likelihood BEFORE |.| and before adding ptn_invar
   Any of the three may be NULL; passing all three NULL makes this identical to
   iqtree_gpu_lik_branch_logl.

   h_ptn_lh == NULL does NOT mean the per-pattern values are not produced: they are always
   computed and left in device memory, and only the host copy is skipped. Pass NULL in the
   Newton branch-length loop, where nothing reads _pattern_lh, and call
   iqtree_gpu_lik_download_ptn_lh() when a consumer finally needs them. */
int iqtree_gpu_lik_branch_logl_full(IqtreeGpuLik* ctx, double len, double* out_logl,
                                    double* h_ptn_lh, double* h_ptn_scaling,
                                    double* h_ptn_lh_cat, int save_log_value);

/* Copies the per-pattern log-likelihoods of the MOST RECENT branch evaluation (either
   entry point above) into h_ptn_lh[npat]. Semantically identical to having passed h_ptn_lh
   to that call, because the device buffer is written by every branch evaluation and by
   nothing else -- the same invariant IQ-TREE's host _pattern_lh has. Fails if no branch
   evaluation has run yet on this context. */
int iqtree_gpu_lik_download_ptn_lh(IqtreeGpuLik* ctx, double* h_ptn_lh);

/* Analytic first and second derivative w.r.t. the branch length. out_logl may be NULL;
   it is produced for free from the same pass. */
int iqtree_gpu_lik_derivatives(IqtreeGpuLik* ctx, double len,
                               double* out_logl, double* out_df, double* out_ddf);

/* ------------------------------------------------------------------------------------
 * G matrix and the MCMCTree Hessian
 *
 * IQ-TREE's --dating mcmctree export needs more than the two scalars above. For every
 * branch i it wants the PER-PATTERN first derivative
 *
 *     G[i][p] = (dL_p/dt_i) / L_p
 *
 * (phylokernelnew.h:2532, `df_ptn.store_a(&G_matrix[g_index + ptn])`), from which
 * main/phyloanalysis.cpp:2853-2855 forms
 *
 *     H = G * diag(ptn_freq) * G^T          (then negated, then its diagonal is replaced
 *                                            by hessian_diagonal[])
 *
 * That is a symmetric rank-k update, so it is one cuBLAS DSYRK on the device rather than a
 * branchNum x npat host transpose plus an Eigen product. The quantity G[i][p] already sits
 * in a register inside k_derivatives (`dfr = df * invL`), so producing it costs one store.
 *
 * The row pitch is the CALLER's, because the host G_matrix has to be readable by
 * printMCMCFileFormat(), which indexes it as G[branch_id*mem_size + ptn] with mem_size the
 * PADDED pattern count (phylotree.cpp:910-921). The engine writes only p < npat and the
 * whole buffer is zeroed at enable, so the padding lanes are exactly 0.0 forever. That is
 * provably equivalent to what the CPU produces, because ptn_freq -- and hence the diagonal
 * of the rank-k update -- is itself exactly 0.0 there (computePtnFreq(),
 * phylotreesse.cpp:614-616), while leaving the padding uninitialised would multiply
 * whatever garbage was in it by that zero and can yield NaN.
 * ---------------------------------------------------------------------------------- */

/* Allocates (and zeroes) the device G matrix: nbranch rows of row_stride doubles.
   Idempotent for the same geometry; re-enabling with a different geometry reallocates.
   Until this is called, the derivative kernel writes no G row at all. */
int iqtree_gpu_lik_gmatrix_enable(IqtreeGpuLik* ctx, int nbranch, int row_stride);

/* As iqtree_gpu_lik_derivatives, but additionally writes row g_branch_id of G. A negative
   g_branch_id, or a context with no G matrix, behaves exactly like the plain call. */
int iqtree_gpu_lik_derivatives_g(IqtreeGpuLik* ctx, double len, int g_branch_id,
                                 double* out_logl, double* out_df, double* out_ddf);

/* Copies one row (row_stride doubles, padding included) back to the host, which is what
   the partitioned / super-tree paths of printMCMCFileFormat() memmove out of. */
int iqtree_gpu_lik_gmatrix_download_row(IqtreeGpuLik* ctx, int branch_id, double* h_row);

/* H = G * diag(ptn_freq) * G^T, computed on the device: G's columns are scaled by
   sqrt(ptn_freq) once and cuBLAS DSYRK does the rest. h_hessian receives nbranch*nbranch
   doubles; the result is symmetric, so row-major and column-major agree and the caller may
   read it either way. NOTE THE SIGN: this is the UNNEGATED product, matching the
   intermediate at phyloanalysis.cpp:2855 before `hessian = (-1) * hessian`. */
int iqtree_gpu_lik_gmatrix_hessian(IqtreeGpuLik* ctx, double* h_hessian);

/* Blocks until every queued kernel has retired. */
int iqtree_gpu_lik_sync(IqtreeGpuLik* ctx);

/* ------------------------------------------------------------------------------------
 * TEST HOOK -- not used by the integration layer.
 *
 * At nstates == 4 the partial-likelihood and multifurcating-child kernels run a dedicated
 * 4-state shape (one thread owns all four states of a (pattern, category); no shared-memory
 * staging of partials; no __syncthreads). The generic templated shape -- the same code the
 * 20-state path runs, instantiated at 4 -- is also compiled, and this switch selects it.
 *
 * The point is differential: the unit test runs BOTH shapes against the SAME scalar CPU
 * reference, so a disagreement between them localises a bug to the specialisation
 * immediately instead of leaving "the DNA kernel is wrong somewhere". Selecting the generic
 * shape changes nothing else -- the host wrappers, the scaling semantics and the scale
 * counters are shared -- and it is a no-op at nstates == 20, which has only one shape.
 *
 * Returns 0 on success. Default is 0 (use the 4-state specialisation).
 * ---------------------------------------------------------------------------------- */
int iqtree_gpu_lik_force_generic_shape(IqtreeGpuLik* ctx, int use_generic);


#ifdef __cplusplus
} /* extern "C" */

// C++ side (phylotreegpu.cpp):
class PhyloTree;
class Params;
int iqtree_gpu_report();   // report visible CUDA devices; returns count

// OUTPUT GUARD (IQ-TREE3-GPU). gpu_eligible() validates the INPUTS of a run -- data type,
// model, tree flags. Correctness also depends on which OUTPUT BUFFERS the run goes on to
// consume, and nothing used to check those, which is how --dating mcmctree ran for months
// against a G_matrix the GPU never filled. This is the missing half: given the run's Params,
// it returns NULL when every output the run will ask for is one the GPU maintains, and
// otherwise a human-readable sentence naming the flag. setLikelihoodKernel() calls it before
// installing the GPU function pointers, so an unsupported output degrades to the CPU kernel
// with a message instead of to a plausible-looking wrong number. See the OUTPUT CONTRACT
// table at the top of phylotreegpu.cpp for the buffer-by-buffer status this is derived from.
const char* iqtree_gpu_unsupported_output(const Params *params);
#endif

#endif /* IQTREE_GPU_KERNELS_H */
