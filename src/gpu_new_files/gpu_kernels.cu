/*
 * gpu_kernels.cu -- CUDA backend for the IQ-TREE 3 GPU likelihood engine (IQ-TREE3-GPU).
 *
 * DELIBERATELY self-contained: this file includes NO IQ-TREE headers, so nvcc never sees
 * vectorclass/AVX intrinsics or Eigen. It exposes the plain C API declared in
 * gpu_kernels.h, which the C++ side (phylotreegpu.cpp, compiled by g++) calls.
 *
 * The algorithm mirrors phylokernelnew.h: partials in the EIGEN basis, power-of-two
 * (ldexp) rescaling at 2^-256, analytic branch-length derivatives. See gpu_kernels.h for
 * the full formulation and the scaling semantics; the comments below explain the GPU
 * choices rather than repeating the maths.
 *
 * WHY THE LAYOUT IS PATTERN-FASTEST.  The CPU stores partial[(pat*ncat+cat)*N+s], i.e.
 * state fastest.  On a GPU that is the worst possible choice: consecutive threads
 * handling consecutive patterns would read addresses ncat*N*8 bytes apart (800 B for
 * protein, 160 B for DNA), so a warp touches 32 separate cache lines.  Here every buffer is
 * buf[(cat*N+state)*pattern_stride + pattern], so a warp reads 32 consecutive doubles --
 * one or two 128-byte transactions.  Nothing else about the kernel matters as much.
 *
 * WHY THE STATE COUNT IS A TEMPLATE PARAMETER AND NOT A RUNTIME ARGUMENT.  The engine
 * supports nstates 20 (amino acids) and 4 (DNA).  Both are compile-time constants of the
 * kernels, instantiated twice, and dispatched on cfg.nstates by the host wrappers.  A
 * runtime nstates would kill the unrolling of the dot products, force the compiler to spill
 * the per-state accumulators to local memory (it cannot keep Ltilde[0..N-1] in registers
 * when the trip count is unknown), and make every index a multiply instead of a shift --
 * all of which hurt most in the DNA case, which is the bandwidth-bound one.  IQ-TREE's own
 * CPU kernels are organised the same way: phylokernelfma.cpp:47-66 dispatches
 * `switch (aln->num_states) { case 4: ...; case 20: ...; default: generic }` over
 * KERNEL_FIX_STATES instantiations.
 *
 * WHY DNA GETS A DIFFERENT KERNEL SHAPE, NOT JUST A DIFFERENT CONSTANT.  Per (pattern,
 * category) an internal-internal node does 2420 FLOP against 480 B at N=20, and 100 FLOP
 * against 96 B at N=4: arithmetic intensity falls from ~5.0 to ~1.0 FLOP/B.  That is below
 * the fp64 balance point of every datacenter card (A100 ~4.8, V100 ~8.7) and below the L40S
 * ~1.6, so protein is compute-bound in fp64 while DNA is MEMORY-bound.  The amino-acid
 * shape -- one thread per (pattern, category, state), cooperating through a shared NxPTILE
 * staging tile behind five __syncthreads -- earns its keep at N=20 because it makes every
 * child element cross the bus once instead of 20 times.  At N=4 the same trade inverts: the
 * tile saves 4 redundant loads that were perfectly coalesced and L2-resident, and pays for
 * them with a shared store, a barrier and four shared loads, in a block of only 32x4 = 128
 * threads.  So the 4-state path uses one thread per (pattern, category) holding all four
 * states in registers, PTILE4 = 128 patterns per block, no shared staging and no barrier at
 * all; the 4x4 matrices are warp-uniform and go through the read-only (__ldg) path.
 * The reduction ORDER is identical in both shapes (acc = row[0]*x[0], then fma), because the
 * unit test compares both against one scalar reference at a tolerance that will not be
 * loosened.
 *
 * WHY DERIVATIVES ARE ANALYTIC.  Because the partials live in the eigen basis, the branch
 * length enters only through the diagonal factor exp(eval[k]*rate_c*t) that multiplies
 * theta.  Differentiating it is exact and costs one extra multiply per term.  The
 * previous probability-space kernel had to use finite differences, whose second
 * derivative error was 1e5-1e9 at realistic branch lengths and produced IQ-TREE's
 * "NUMERICAL ERROR: Wrong computeFuncDerv".
 *
 * FMA POLICY.  The build uses -fmad=false so that the compiler never silently contracts
 * a*b+c.  Every place where a fused multiply-add is wanted calls fma() explicitly, which
 * matches the CPU kernel's mul_add() and keeps rounding behaviour predictable.
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <atomic>
#include <cuda_runtime.h>
/* cuBLAS is used for exactly one thing: the symmetric rank-k update that turns the
   per-pattern derivative matrix G into the MCMCTree Hessian. A hand-written DSYRK would be
   both slower and one more piece of numerical code to justify, and this one is not on the
   hot path -- it runs once per --dating mcmctree export. */
#include <cublas_v2.h>

#include "gpu_kernels.h"

/* ------------------------------------------------------------------------------------
 * Compile-time constants
 * ---------------------------------------------------------------------------------- */

/* The two supported state counts. Every NST-dependent kernel below is a template on the
   state count and is instantiated for exactly these two; the host wrappers dispatch on
   cfg.nstates. Fixing the count at compile time is what lets every dot product be fully
   unrolled and (at 20) every matrix be staged in shared memory. */
#define NST_AA  20
#define NST_DNA 4

/* Patterns per thread-block tile for the GENERIC (state-per-thread) kernels. One warp wide,
   so the pattern index is exactly the lane index and every global access is perfectly
   coalesced. The block is dim3(PTILE, NST), i.e. 640 threads at NST=20. */
#define PTILE 32

/* Patterns per thread-block tile for the 4-state specialisations, whose block is
   dim3(PTILE4) -- one thread per (pattern, category), all four states in registers. 128 is
   four warps of patterns, which restores a sensible block size after the state dimension
   leaves the block shape (dim3(PTILE, 4) would be a mere 128 threads with a 4-wide y).
   It must stay a multiple of 32 so that the twelve global accesses per thread stay one
   128-byte transaction per warp each.

   The PATTERN STRIDE is deliberately NOT raised to a multiple of PTILE4: it stays npat
   rounded to PTILE, which is a documented property of the public API and what the padding
   arguments elsewhere in this file are written against. The 4-state kernels therefore
   launch ceil(stride/PTILE4) blocks and carry a bounds test instead of an exact division.
   That is free here precisely because they have no cooperative phase -- there is no
   __syncthreads that an early-returning thread could deadlock. */
#define PTILE4 128

/* IQ-TREE's scaling constants (phylotree.h). SCALING_THRESHOLD is 2^-256 and the rescale
   multiplies by 2^256 via ldexp, so the mantissa is preserved bit-for-bit. */
#define SCALING_THRESHOLD_EXP 256
__host__ __device__ static inline double gpu_scaling_threshold() { return 0x1p-256; }
/* log(2^-256) == -256*ln(2). */
static const double LOG_SCALING_THRESHOLD = -256.0 * 0.69314718055994530942;

/* Fixed reduction geometry. Fixing the block count (rather than deriving it from npat)
   makes the two-stage sum bit-reproducible across runs for a given problem size. */
#define RED_THREADS 256
#define RED_BLOCKS  256

/* ------------------------------------------------------------------------------------
 * Error plumbing
 * ---------------------------------------------------------------------------------- */

static char g_lik_err[512] = "";

/* ------------------------------------------------------------------------------------
 * Launch accounting (STEP 1 of the DNA launch-overhead work)
 * ----------------------------------------------------------------------------------
 * DIAGNOSIS TO CONFIRM. The DNA (4-state) likelihood path is memory-bound with a tiny
 * work-per-launch (a per-node kernel does ~3 us of compute but pays ~7 us of launch
 * latency), and the traversal issues MANY tiny launches with the host driving one node at
 * a time. To turn that hypothesis into a number -- and to MEASURE any fix -- we count every
 * real CUDA kernel launch, and, separately, every replay of a captured CUDA graph. If the
 * hypothesis holds the DNA run shows millions of kernel_launches; a graph-based fix must
 * drop kernel_launches sharply and raise graph_replays.
 *
 * WHERE IT INCREMENTS. Every device launch in this file is immediately followed by a
 * KERNEL_CHECK(...) (the 1:1 launch-error probe), so counting inside KERNEL_CHECK counts
 * launches exactly, with no risk of a launch site being missed or double-counted, and no new
 * code at 39 call sites. The one launch NOT followed by KERNEL_CHECK is the one-time
 * k_fill_int in create(), a negligible constant.
 *
 * WHY RELAXED ATOMIC. The device-launch path is single-threaded (packet 0 owns the node),
 * so a plain counter would do; the atomic is defensive and, being a relaxed fetch_add with
 * no fences, is unmeasurable against a launch of hundreds of microseconds. */
static std::atomic<long long> g_kernel_launches{0};
static std::atomic<long long> g_graph_replays{0};
/* On-device Newton loops run to completion in a single conditional-graph launch (STEP 3). */
static std::atomic<long long> g_newton_loops{0};

/* The stream every kernel launch and every device-to-device copy in this file is issued on.
 * It defaults to 0 (the legacy default stream), which makes a launch on stream 0 byte-
 * identical to a launch with no stream argument -- so threading it through the launches
 * changes NOTHING on the
 * normal path. It is temporarily pointed at an explicit capture stream ONLY inside
 * iqtree_gpu_lik_graph_capture_begin()/..._end(), which is the sole way to record the
 * launch DAG into a cudaGraph. See the CUDA-graph facility near the end of this file. */
static cudaStream_t g_stream = 0;

static int lik_fail(const char *what, cudaError_t e) {
    snprintf(g_lik_err, sizeof(g_lik_err), "%s: %s", what,
             (e == cudaSuccess) ? "invalid argument" : cudaGetErrorString(e));
    return -1;
}
static int lik_fail_msg(const char *what) {
    snprintf(g_lik_err, sizeof(g_lik_err), "%s", what);
    return -1;
}

#define CUDA_TRY(call, what)                                       \
    do {                                                           \
        cudaError_t e_ = (call);                                    \
        if (e_ != cudaSuccess) return lik_fail((what), e_);          \
    } while (0)

#define KERNEL_CHECK(what)                                         \
    do {                                                           \
        g_kernel_launches.fetch_add(1, std::memory_order_relaxed);  \
        cudaError_t e_ = cudaGetLastError();                        \
        if (e_ != cudaSuccess) return lik_fail((what), e_);          \
    } while (0)

/* ------------------------------------------------------------------------------------
 * On-device Newton loop state (STEP 3)
 * ----------------------------------------------------------------------------------
 * A device-resident mirror of the local variables of Optimization::minimizeNewton
 * (utils/optimization.cpp:422-499). k_newton_step advances one iteration per body pass of a
 * conditional WHILE graph, so the whole per-branch Newton loop runs on the GPU with the host
 * reading back only the final length. Field names mirror minimizeNewton's locals exactly so the
 * port can be checked line by line. `result` is the value minimizeNewton would return; `error`
 * latches the !isfinite(f)/!isfinite(df) case that minimizeNewton handles with nrerror(). */
struct NewtonState {
    double x1, x2, xacc;      /* bracket ends and tolerance (== min_branch_length)         */
    double xl, xh;            /* current bracket                                            */
    double rts, rts_old;      /* current and previous root estimate                        */
    double dx, dxold;         /* current and previous step                                 */
    double d2l;               /* the 2nd-derivative value minimizeNewton returns in &d2l    */
    double result;            /* the length minimizeNewton would return                     */
    int    j;                 /* Newton step counter (1..maxNRStep)                         */
    int    maxNRStep;
    int    initialized;       /* 0 until the pre-loop eval (iteration 0) has been processed  */
    int    converged;         /* 1 when the loop is done                                     */
    int    error;             /* 1 if a non-finite derivative was seen                       */
};

/* ------------------------------------------------------------------------------------
 * Engine state
 * ---------------------------------------------------------------------------------- */

struct IqtreeGpuLik {
    IqtreeGpuLikConfig cfg;

    int    stride;        /* npat rounded up to a multiple of PTILE                  */

    /* Test hook (iqtree_gpu_lik_force_generic_shape): run the generic templated partial /
       multifurcating-child kernels at nstates == 4 instead of the 4-state specialisations,
       so the unit test can hold both GPU shapes against one CPU reference. Never set by the
       integration layer; no effect at nstates == 20. */
    int    force_generic;

    /* CURRENT geometry -- what the kernels use for launch shapes and indexing. These move
       with cfg.ncat on every iqtree_gpu_lik_set_ncat(). */
    int    block;         /* ncat * cfg.nstates                                      */
    size_t partial_elems; /* block * stride                                          */
    size_t scale_elems;   /* stride * (safe_numeric ? ncat : 1)                       */

    /* ALLOCATED capacity -- what every cudaMalloc below was sized from. Never changes for
       the life of the engine. The kernels never see these: widening their notion of `block`
       would make them compute categories that do not exist. */
    int    ncat_cap;
    int    block_cap;         /* ncat_cap * cfg.nstates                              */
    size_t partial_elems_cap; /* block_cap * stride                                  */
    size_t scale_elems_cap;   /* stride * (safe_numeric ? ncat_cap : 1)               */

    /* ncat at the last successful set_rates. set_ncat() drops the rate/prop vectors (they
       are per-category DATA, not capacity), and every kernel that reads them refuses to run
       until they have been re-uploaded for the current ncat. */
    int    rates_ncat;

    /* model, uploaded once per model update. N below is cfg.nstates. */
    double *d_eval;       /* [N]                                                     */
    double *d_evec;       /* [N*N], evec[x*N+i]                                      */
    double *d_inv_evec;   /* [N*N], inv_evec[k*N+x]                                  */
    double *d_rates;      /* [ncat_cap], first ncat entries live                     */
    double *d_props;      /* [ncat_cap], already carries (1 - p_invar)               */
    double *d_tip_lh;     /* [num_tip_states*N]                                      */

    /* alignment, uploaded once */
    double *d_ptn_freq;   /* [stride], zero-padded                                   */
    double *d_ptn_invar;  /* [stride], zero-padded                                   */
    int    *d_tip_states; /* [tip_slots*stride], padded with state_unknown           */

    /* per-branch precompute slots. Slot pitch is the CAPACITY pitch, so slot bases never
       move when ncat changes; each slot then uses only its first ncat*... elements. */
    double *d_echild;     /* [echild_slots * ncat_cap * N*N]                          */
    double *d_leaf;       /* [echild_slots * num_tip_states * block_cap]              */

    /* per-branch working set. theta and bufscale are per SLOT: the derivative path and the
       branch-likelihood path each own one, so that a computeLikelihoodBranch cannot
       invalidate the theta that IQ-TREE's theta_computed flag still vouches for. */
    int     theta_slots;  /* >= 1                                                      */
    int     theta_slot;   /* currently selected slot                                   */
    double *d_theta;      /* [theta_slots * partial_elems_cap]                         */
    double *d_bufscale;   /* [theta_slots * stride], min_scale * log(2^-256)           */
    double *d_val;        /* [3*block_cap], val0 | val1 | val2                         */

    /* G matrix (--dating mcmctree only; NULL otherwise, and then no G row is ever written).
       Row pitch is the CALLER's mem_size, not our pattern_stride, because the host consumer
       indexes G[branch_id*mem_size + ptn]. Zeroed at enable and only p < npat is ever
       written, so the padding lanes stay exactly 0.0. */
    double *d_gmat;       /* [g_nbranch * g_stride]                                    */
    int     g_nbranch;
    int     g_stride;
    double *d_hess;       /* [g_nbranch * g_nbranch], DSYRK output                     */
    cublasHandle_t cublas;
    int     cublas_ready;

    /* Device-resident branch lengths for the CUDA-graph replay path (STEP 2). Indexed by
       IQ-TREE branch id; k_echild_dev reads d_branch_len[branch_id]. Lazily allocated by
       iqtree_gpu_lik_set_branch_lengths and grown if the branch count ever rises. NULL until
       the first full-traversal capture asks for it. */
    double *d_branch_len; /* [branch_len_cap]                                          */
    int     branch_len_cap;

    /* Device scalar holding the candidate branch length for the Newton-loop replay path
       (k_val0_dev / k_val012_dev read d_len[0]). Allocated at create; refreshed by
       iqtree_gpu_lik_set_eval_len before each derivative/logl graph replay, and updated ON
       DEVICE by k_newton_step during the on-device Newton loop. */
    double *d_len;        /* [1]                                                       */

    /* On-device Newton loop (STEP 3). d_newton holds the loop state; newton_exec is the
       instantiated conditional-WHILE graph [val012_dev -> derivatives -> reduce -> newton_step].
       Built lazily and rebuilt when ncat / theta slot / geometry change (the host tags it). */
    NewtonState    *d_newton;    /* [1] */
    cudaGraphExec_t newton_exec; /* NULL until built */

    /* scratch */
    double *d_u;          /* [partial_elems_cap], multifurcating accumulator (lazy)    */
    double *d_sitelh;     /* [stride], per-pattern likelihood (lazy)                   */
    double *d_lhcat;      /* [ncat_cap*stride], per-pattern per-category lh (lazy)     */
    double *d_redblocks;  /* [RED_BLOCKS*3]                                            */
    double *d_redout;     /* [3]                                                       */
};

/* Base of the currently selected theta / buffer-scale slot. The theta pitch is the CAPACITY
   pitch, not the current one, so that slot 1 does not slide over slot 0's tail when ncat
   shrinks -- the slots must stay independent across a set_ncat, that is their whole point. */
static inline double* theta_of(IqtreeGpuLik *c) {
    return c->d_theta + (size_t)c->theta_slot * c->partial_elems_cap;
}
static inline double* bufscale_of(IqtreeGpuLik *c) {
    return c->d_bufscale + (size_t)c->theta_slot * c->stride;
}

/* ------------------------------------------------------------------------------------
 * Device helpers
 * ---------------------------------------------------------------------------------- */

__device__ __forceinline__ double warp_sum(double v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* Stage-1 half of the hand-rolled reduction (no CUB dependency by design): NACC
   accumulators are reduced within the block and one partial per block is written out.
   Stage 2 then folds the RED_BLOCKS partials with a single block. */
template <int NACC>
__device__ __forceinline__ void block_sum_store(double *v, double *out_blocks) {
    __shared__ double sh[NACC][32];
    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
#pragma unroll
    for (int a = 0; a < NACC; a++) {
        double s = warp_sum(v[a]);
        if (lane == 0) sh[a][warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int a = 0; a < NACC; a++) {
            double s = (lane < nwarps) ? sh[a][lane] : 0.0;
            s = warp_sum(s);
            if (lane == 0) out_blocks[(size_t)blockIdx.x * NACC + a] = s;
        }
    }
}

template <int NACC>
__global__ void k_reduce_stage2(const double * __restrict__ blocks, int n, double *out) {
    double v[NACC];
#pragma unroll
    for (int a = 0; a < NACC; a++) v[a] = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
#pragma unroll
        for (int a = 0; a < NACC; a++) v[a] += blocks[(size_t)i * NACC + a];
    }
    block_sum_store<NACC>(v, out);
}

/* ------------------------------------------------------------------------------------
 * Small utility kernels
 * ---------------------------------------------------------------------------------- */

__global__ void k_fill(double *p, double v, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

__global__ void k_fill_int(int *p, int v, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

/* out = a + b, with either operand optionally absent (treated as zero). Used to seed a
   node's scale counters from its children before the underflow test bumps them. */
__global__ void k_scale_init(unsigned char *out, const unsigned char * __restrict__ a,
                             const unsigned char * __restrict__ b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int v = 0;
    if (a) v += a[i];
    if (b) v += b[i];
    out[i] = (unsigned char)v;
}

__global__ void k_scale_add(unsigned char *out, const unsigned char * __restrict__ a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (unsigned char)(out[i] + a[i]);
}

/* ------------------------------------------------------------------------------------
 * Per-branch precompute
 * ---------------------------------------------------------------------------------- */

/* echild[(c*N + x)*N + i] = evec[x*N+i] * exp(eval[i] * rate_c * len).
   The parenthesisation exp(eval[i] * (rate_c*len)) is IQ-TREE's (it forms len_child =
   rate*len first); it differs from exp((eval*rate)*len) only in the last ulp. */
template <int NST>
__global__ void k_echild(double *echild, const double * __restrict__ evec,
                         const double * __restrict__ eval, const double * __restrict__ rates,
                         double len, int ncat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncat * NST * NST) return;
    int c = idx / (NST * NST);
    int r = idx - c * (NST * NST);
    int x = r / NST;
    int i = r - x * NST;
    echild[idx] = evec[x * NST + i] * exp(eval[i] * (rates[c] * len));
}

/* Device-len twin of k_echild (STEP 2 wiring). Identical arithmetic, but the
   branch length is READ FROM DEVICE MEMORY -- d_branch_len[branch_id] -- instead of arriving as
   a by-value kernel argument. This is the single enabling change for CUDA-graph replay of a
   full traversal: a captured k_echild_dev launch freezes only the branch_id (a topological
   constant of the fixed tree), so the SAME graph replays correctly after the host uploads a new
   branch-length vector for the next evaluation. The by-value k_echild above is retained
   unchanged for the immediate (non-captured) path and the amino-acid path, so nothing on the
   validated hot path moves. exp(eval[i]*(rates[c]*len)) is byte-for-byte the same expression. */
template <int NST>
__global__ void k_echild_dev(double *echild, const double * __restrict__ evec,
                             const double * __restrict__ eval, const double * __restrict__ rates,
                             const double * __restrict__ d_branch_len, int branch_id, int ncat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncat * NST * NST) return;
    int c = idx / (NST * NST);
    int r = idx - c * (NST * NST);
    int x = r / NST;
    int i = r - x * NST;
    double len = d_branch_len[branch_id];
    echild[idx] = evec[x * NST + i] * exp(eval[i] * (rates[c] * len));
}

/* Collapse a tip child: leaf[state*block + c*N + x] = SUM_i echild[(c*N+x)*N+i]*tip[state][i].
   The STATE_UNKNOWN row is forced to exactly 1.0 over the whole block -- IQ-TREE writes
   the literal 1.0 rather than computing it, and the difference is observable
   (phylokernelnew.h:997-1001). Only the row INDEX moves between data types (23 -> 18); it is
   a config field either way, and every ambiguity row below it arrives pre-collapsed in
   tip_partial_lh, so this kernel never interprets an IUPAC code. */
template <int NST>
__global__ void k_leaf_partials(double *leaf, const double * __restrict__ echild,
                                const double * __restrict__ tip,
                                int nrows, int block, int state_unknown) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrows * block) return;
    int st = idx / block;
    int b  = idx - st * block;
    if (st == state_unknown) { leaf[idx] = 1.0; return; }
    const double *e = echild + (size_t)b * NST;   /* b == c*NST + x already */
    const double *t = tip + (size_t)st * NST;
    double acc = e[0] * t[0];
#pragma unroll
    for (int i = 1; i < NST; i++) acc = fma(e[i], t[i], acc);
    leaf[idx] = acc;
}

/* ------------------------------------------------------------------------------------
 * Partial-likelihood kernel (bifurcating: tip-tip, tip-internal, internal-internal)
 * ---------------------------------------------------------------------------------- */

struct PartialArgs {
    double       *out;
    const double *eL;        /* echild slot for the left child,  ncat*N*N (internal only) */
    const double *eR;        /* echild slot for the right child, ncat*N*N (internal only) */
    const double *leafL;     /* collapsed leaf table (tip only)                           */
    const double *leafR;
    const int    *stateL;    /* per-pattern tip state (tip only)                          */
    const int    *stateR;
    const double *inL;       /* child partial in eigen basis (internal only)              */
    const double *inR;
    const double *inv_evec;
    int           P;         /* pattern stride                                            */
    int           block;     /* ncat*N                                                    */
};

/* GENERIC SHAPE -- the amino-acid path, and also compiled at NST=4 so that the unit test can
   hold the 4-state specialisation below against a second GPU shape.

   One thread per (pattern, category, state); grid.y indexes the category so that grid.x
   and threadIdx.x carry the pattern.  Each thread produces one output state through an
   NST-element dot product, which at NST=20 is the arithmetic-intensity sweet spot claimed by
   the design (~2400 FLOP per pattern-category against ~480 B of traffic).
 *
 * The block cooperates twice through shared memory:
 *   phase 1 -- thread (p,x) forms u_x = (echild_l . Ltilde_l) * (echild_r . Ltilde_r)
 *   phase 2 -- thread (p,k) forms Ltilde[k] = SUM_x inv_evec[k][x] * u_x
 * The same NSTxPTILE shared tile is reused to stage each child's partial before the dot
 * product, so every element of each child's partial is read from global memory exactly
 * once instead of NST times.  At NST=20 that saves 19 redundant loads for the price of one
 * shared store, one barrier and 20 shared loads and is clearly worth it; at NST=4 it saves
 * three and is not, which is what k_partial4 exists for.
 */
template <int NST, bool LTIP, bool RTIP>
__global__ void k_partial(PartialArgs a) {
    extern __shared__ double smem[];

    const int c   = blockIdx.y;
    const int tx  = threadIdx.x;          /* pattern within the tile   */
    const int y   = threadIdx.y;          /* state index, 0..19        */
    const int p   = blockIdx.x * PTILE + tx;
    const int tid = y * PTILE + tx;
    const bool act = (p < a.P);

    double *s_el = smem;
    double *s_er = s_el + (LTIP ? 0 : NST * NST);
    double *s_iv = s_er + (RTIP ? 0 : NST * NST);
    double *s_u  = s_iv + NST * NST;

    const int nthreads = PTILE * NST;
    if (!LTIP)
        for (int i = tid; i < NST * NST; i += nthreads) s_el[i] = a.eL[(size_t)c * NST * NST + i];
    if (!RTIP)
        for (int i = tid; i < NST * NST; i += nthreads) s_er[i] = a.eR[(size_t)c * NST * NST + i];
    for (int i = tid; i < NST * NST; i += nthreads) s_iv[i] = a.inv_evec[i];
    __syncthreads();

    double vL, vR;

    if (LTIP) {
        /* Pre-collapsed tip vector: one lookup, no dot product. The gather over
           stateL[p] is scattered within the warp, but the table is only
           num_tip_states*block doubles and stays hot in L1. */
        vL = act ? a.leafL[(size_t)a.stateL[p] * a.block + c * NST + y] : 0.0;
    } else {
        s_u[y * PTILE + tx] = act ? a.inL[((size_t)(c * NST + y)) * a.P + p] : 0.0;
        __syncthreads();
        const double *row = s_el + y * NST;
        double acc = row[0] * s_u[0 * PTILE + tx];
#pragma unroll
        for (int i = 1; i < NST; i++) acc = fma(row[i], s_u[i * PTILE + tx], acc);
        vL = acc;
        __syncthreads();
    }

    if (RTIP) {
        vR = act ? a.leafR[(size_t)a.stateR[p] * a.block + c * NST + y] : 0.0;
    } else {
        s_u[y * PTILE + tx] = act ? a.inR[((size_t)(c * NST + y)) * a.P + p] : 0.0;
        __syncthreads();
        const double *row = s_er + y * NST;
        double acc = row[0] * s_u[0 * PTILE + tx];
#pragma unroll
        for (int i = 1; i < NST; i++) acc = fma(row[i], s_u[i * PTILE + tx], acc);
        vR = acc;
        __syncthreads();
    }

    s_u[y * PTILE + tx] = vL * vR;
    __syncthreads();

    if (act) {
        const double *row = s_iv + y * NST;   /* y now plays the role of the output k */
        double acc = row[0] * s_u[0 * PTILE + tx];
#pragma unroll
        for (int x = 1; x < NST; x++) acc = fma(row[x], s_u[x * PTILE + tx], acc);
        a.out[((size_t)(c * NST + y)) * a.P + p] = acc;
    }
}

/* 4-STATE SHAPE.  Same maths, same reduction order, same outputs as k_partial<4,...>; a
 * different thread mapping, because DNA is memory-bound and the generic shape's shared-memory
 * staging is a net cost at four states (see the file header).
 *
 * One thread owns ALL FOUR states of one (pattern, category):
 *
 *   grid  = (ceil(P / PTILE4), ncat),  block = (PTILE4)
 *   p = blockIdx.x*PTILE4 + threadIdx.x,  c = blockIdx.y
 *
 * Consequences, each of which is the reason for the change rather than a side effect:
 *
 *   - Global traffic is IDENTICAL to the generic shape: 8 doubles in, 4 out per
 *     (pattern, category).  The staging tile never reduced traffic, it reduced the number
 *     of TIMES each element was fetched; holding the four values in registers achieves the
 *     same thing without the round-trip.
 *   - Zero __syncthreads.  All five barriers of the generic kernel exist to protect the
 *     shared tile.  With no tile there is nothing to protect, which also makes the bounds
 *     test a plain early return -- no thread can be waiting at a barrier the returned ones
 *     will never reach.
 *   - Coalescing is preserved: threadIdx.x is still the pattern, so each of the twelve
 *     accesses is one 128-byte transaction per warp.
 *   - The three 4x4 matrices are only 128 bytes each and are WARP-UNIFORM -- every thread in
 *     the warp reads the same 48 doubles -- so they go through the read-only (__ldg) path,
 *     which broadcasts and keeps them resident, rather than through shared memory.
 *   - Reduction order is preserved exactly (acc = row[0]*x[0], then fma for i >= 1), matching
 *     the generic kernel, the CPU kernel and the unit test's scalar reference.  Changing it
 *     would move results by ulps and the test's tolerance is not going to be loosened.
 */
template <bool LTIP, bool RTIP>
__global__ void k_partial4(PartialArgs a) {
    const int c = blockIdx.y;
    const int p = blockIdx.x * PTILE4 + threadIdx.x;
    if (p >= a.P) return;                      /* padding lanes past the stride */

    const int cb = c * NST_DNA;                /* first state plane of this category */
    double vL[NST_DNA], vR[NST_DNA];

    if (LTIP) {
        /* Pre-collapsed tip vector: four consecutive doubles, no dot product. The gather
           over stateL[p] is scattered within the warp, but the whole leaf table is
           num_tip_states*block doubles (19*ncat*4, a few kB) and stays hot in L1. */
        const double *lf = a.leafL + (size_t)a.stateL[p] * a.block + cb;
#pragma unroll
        for (int x = 0; x < NST_DNA; x++) vL[x] = lf[x];
    } else {
        double Ll[NST_DNA];
#pragma unroll
        for (int i = 0; i < NST_DNA; i++) Ll[i] = a.inL[((size_t)(cb + i)) * a.P + p];
        const double *e = a.eL + (size_t)c * (NST_DNA * NST_DNA);
#pragma unroll
        for (int x = 0; x < NST_DNA; x++) {
            const double *row = e + x * NST_DNA;
            double acc = __ldg(row) * Ll[0];
#pragma unroll
            for (int i = 1; i < NST_DNA; i++) acc = fma(__ldg(row + i), Ll[i], acc);
            vL[x] = acc;
        }
    }

    if (RTIP) {
        const double *lf = a.leafR + (size_t)a.stateR[p] * a.block + cb;
#pragma unroll
        for (int x = 0; x < NST_DNA; x++) vR[x] = lf[x];
    } else {
        double Lr[NST_DNA];
#pragma unroll
        for (int i = 0; i < NST_DNA; i++) Lr[i] = a.inR[((size_t)(cb + i)) * a.P + p];
        const double *e = a.eR + (size_t)c * (NST_DNA * NST_DNA);
#pragma unroll
        for (int x = 0; x < NST_DNA; x++) {
            const double *row = e + x * NST_DNA;
            double acc = __ldg(row) * Lr[0];
#pragma unroll
            for (int i = 1; i < NST_DNA; i++) acc = fma(__ldg(row + i), Lr[i], acc);
            vR[x] = acc;
        }
    }

    double u[NST_DNA];
#pragma unroll
    for (int x = 0; x < NST_DNA; x++) u[x] = vL[x] * vR[x];

#pragma unroll
    for (int k = 0; k < NST_DNA; k++) {
        const double *row = a.inv_evec + k * NST_DNA;
        double acc = __ldg(row) * u[0];
#pragma unroll
        for (int x = 1; x < NST_DNA; x++) acc = fma(__ldg(row + x), u[x], acc);
        a.out[((size_t)(cb + k)) * a.P + p] = acc;
    }
}

/* ------------------------------------------------------------------------------------
 * Rescaling
 * ---------------------------------------------------------------------------------- */

/* NORM_LH (safe_numeric == 0): ONE counter per pattern and ONE test per pattern, taken on
   the maximum over the WHOLE block, i.e. across every rate category at once.  That is
   what IQ-TREE does when lh_max is accumulated by productVecMat across the category loop
   without being reset.
 *
 * Note the deliberate absence of a "max > 0" guard: IQ-TREE rescales a block whose
 * maximum is exactly zero as well (a no-op on the values) and still increments the
 * counter.  Adding the guard would silently change the scale bookkeeping. */
__global__ void k_rescale_norm(double *L, unsigned char *scale,
                               const double * __restrict__ invar, int P, int block) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    double m = 0.0;
    for (int b = 0; b < block; b++) {
        double v = fabs(L[(size_t)b * P + p]);
        m = fmax(m, v);
    }
    if (m < gpu_scaling_threshold() && invar[p] == 0.0) {
        for (int b = 0; b < block; b++)
            L[(size_t)b * P + p] = ldexp(L[(size_t)b * P + p], SCALING_THRESHOLD_EXP);
        scale[p] = (unsigned char)(scale[p] + 1);
    }
}

/* SAFE_LH (safe_numeric == 1): one counter per (pattern, category), test inside the
   category loop. Counter layout is scale[c*P + p] so that it stays pattern-fastest.
   NST enters only as the width of one category's state run; the threshold, the ldexp step
   and the deliberate absence of a "max > 0" guard are pure floating-point properties and are
   the same for 4 states as for 20. */
template <int NST>
__global__ void k_rescale_safe(double *L, unsigned char *scale,
                               const double * __restrict__ invar, int P, int ncat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P * ncat) return;
    int c = idx / P;
    int p = idx - c * P;
    double m = 0.0;
    for (int x = 0; x < NST; x++) {
        double v = fabs(L[((size_t)(c * NST + x)) * P + p]);
        m = fmax(m, v);
    }
    if (m < gpu_scaling_threshold() && invar[p] == 0.0) {
        for (int x = 0; x < NST; x++) {
            size_t o = ((size_t)(c * NST + x)) * P + p;
            L[o] = ldexp(L[o], SCALING_THRESHOLD_EXP);
        }
        scale[idx] = (unsigned char)(scale[idx] + 1);
    }
}

/* ------------------------------------------------------------------------------------
 * Multifurcating node (degree > 3)
 * ---------------------------------------------------------------------------------- */

/* u *= this child's contribution, in the x (non-eigen) basis.  Kept as a separate kernel
   from k_partial because IQ-TREE folds multifurcating children in one at a time and runs
   the underflow test after EVERY child, on the running product -- an ordering that is
   observable through the scale counters, so it cannot be collapsed into a single pass.
   Generic shape, as k_partial; k_mf_child4 below is its 4-state counterpart. */
template <int NST, bool TIP>
__global__ void k_mf_child(double *u, const double * __restrict__ echild,
                           const double * __restrict__ leaf, const int * __restrict__ state,
                           const double * __restrict__ in, int P, int block) {
    extern __shared__ double smem[];
    const int c   = blockIdx.y;
    const int tx  = threadIdx.x;
    const int y   = threadIdx.y;
    const int p   = blockIdx.x * PTILE + tx;
    const int tid = y * PTILE + tx;
    const bool act = (p < P);

    double *s_e     = smem;                       /* NST*NST, internal children only */
    double *s_stage = s_e + (TIP ? 0 : NST * NST);/* NST*PTILE                        */

    const int nthreads = PTILE * NST;
    if (!TIP)
        for (int i = tid; i < NST * NST; i += nthreads) s_e[i] = echild[(size_t)c * NST * NST + i];

    double v;
    if (TIP) {
        __syncthreads();
        v = act ? leaf[(size_t)state[p] * block + c * NST + y] : 0.0;
    } else {
        s_stage[y * PTILE + tx] = act ? in[((size_t)(c * NST + y)) * P + p] : 0.0;
        __syncthreads();
        const double *row = s_e + y * NST;
        double acc = row[0] * s_stage[0 * PTILE + tx];
#pragma unroll
        for (int i = 1; i < NST; i++) acc = fma(row[i], s_stage[i * PTILE + tx], acc);
        v = acc;
    }
    if (act) u[((size_t)(c * NST + y)) * P + p] *= v;
}

/* 4-state counterpart of k_mf_child: one thread per (pattern, category), all four states in
   registers, no shared memory and no barrier. Same argument as k_partial4. */
template <bool TIP>
__global__ void k_mf_child4(double *u, const double * __restrict__ echild,
                            const double * __restrict__ leaf, const int * __restrict__ state,
                            const double * __restrict__ in, int P, int block) {
    const int c = blockIdx.y;
    const int p = blockIdx.x * PTILE4 + threadIdx.x;
    if (p >= P) return;

    const int cb = c * NST_DNA;
    double v[NST_DNA];

    if (TIP) {
        const double *lf = leaf + (size_t)state[p] * block + cb;
#pragma unroll
        for (int x = 0; x < NST_DNA; x++) v[x] = lf[x];
    } else {
        double Ll[NST_DNA];
#pragma unroll
        for (int i = 0; i < NST_DNA; i++) Ll[i] = in[((size_t)(cb + i)) * P + p];
        const double *e = echild + (size_t)c * (NST_DNA * NST_DNA);
#pragma unroll
        for (int x = 0; x < NST_DNA; x++) {
            const double *row = e + x * NST_DNA;
            double acc = __ldg(row) * Ll[0];
#pragma unroll
            for (int i = 1; i < NST_DNA; i++) acc = fma(__ldg(row + i), Ll[i], acc);
            v[x] = acc;
        }
    }
#pragma unroll
    for (int x = 0; x < NST_DNA; x++) u[((size_t)(cb + x)) * P + p] *= v[x];
}

/* Final rotation into the eigen basis. Never rescaled: in the multifurcating case
   IQ-TREE has already done all its scaling on the x-basis accumulator.
   Left in the generic shape at both state counts: it runs once per multifurcating node
   rather than once per child, and multifurcating nodes are rare enough that a second
   4-state shape would be more code than it is worth. */
template <int NST>
__global__ void k_mf_transform(double *out, const double * __restrict__ u,
                               const double * __restrict__ inv_evec, int P) {
    extern __shared__ double smem[];
    const int c   = blockIdx.y;
    const int tx  = threadIdx.x;
    const int y   = threadIdx.y;
    const int p   = blockIdx.x * PTILE + tx;
    const int tid = y * PTILE + tx;
    const bool act = (p < P);

    double *s_iv = smem;                 /* NST*NST  */
    double *s_u  = s_iv + NST * NST;     /* NST*PTILE */

    const int nthreads = PTILE * NST;
    for (int i = tid; i < NST * NST; i += nthreads) s_iv[i] = inv_evec[i];
    s_u[y * PTILE + tx] = act ? u[((size_t)(c * NST + y)) * P + p] : 0.0;
    __syncthreads();

    if (act) {
        const double *row = s_iv + y * NST;
        double acc = row[0] * s_u[0 * PTILE + tx];
#pragma unroll
        for (int x = 1; x < NST; x++) acc = fma(row[x], s_u[x * PTILE + tx], acc);
        out[((size_t)(c * NST + y)) * P + p] = acc;
    }
}

/* ------------------------------------------------------------------------------------
 * Theta
 * ---------------------------------------------------------------------------------- */

/* theta[i] = Ltilde_node[i] * Ltilde_dad[i], elementwise over the whole block. */
__global__ void k_theta_internal(double *theta, const double * __restrict__ node,
                                 const double * __restrict__ dad, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) theta[i] = node[i] * dad[i];
}

/* dad is a leaf: the SAME nstates-element tip vector is reused for every rate category,
   because a tip partial has no category dimension. That is what the CPU does too --
   phylokernelnew.h:2133-2141 re-bases the tip vector per category with no category
   dimension of its own -- and it is independent of the state count. */
template <int NST>
__global__ void k_theta_tip(double *theta, const double * __restrict__ tip,
                            const int * __restrict__ state,
                            const double * __restrict__ dad, int P) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int b = blockIdx.y;
    const int i = b - (b / NST) * NST;
    theta[(size_t)b * P + p] = tip[(size_t)state[p] * NST + i] * dad[(size_t)b * P + p];
}

/* NORM_LH scale combination: nothing to renormalise, the counter is already per pattern. */
__global__ void k_theta_scale_norm(double *bufscale, const unsigned char * __restrict__ sdad,
                                   const unsigned char * __restrict__ snode,
                                   double logthr, int P) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int s = sdad[p];
    if (snode) s += snode[p];
    bufscale[p] = (double)s * logthr;
}

/* SAFE_LH scale combination.  Categories are renormalised to the MINIMUM scale count:
   a category exactly one step above the minimum is multiplied by 2^-256, and a category
   MORE than one step above is HARD-ZEROED.  The hard zero is a genuine numerical
   behaviour of IQ-TREE (it discards categories whose contribution is more than 2^-512
   below the dominant one), not a guard, so it is replicated exactly. NST is only the width
   of the category being renormalised; the decision itself is over scale counters. */
template <int NST>
__global__ void k_theta_scale_safe(double *theta, double *bufscale,
                                   const unsigned char * __restrict__ sdad,
                                   const unsigned char * __restrict__ snode,
                                   double logthr, int P, int ncat) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    int mn = 1 << 30;
    for (int c = 0; c < ncat; c++) {
        int s = sdad[(size_t)c * P + p];
        if (snode) s += snode[(size_t)c * P + p];
        if (s < mn) mn = s;
    }
    for (int c = 0; c < ncat; c++) {
        int s = sdad[(size_t)c * P + p];
        if (snode) s += snode[(size_t)c * P + p];
        if (s == mn + 1) {
            for (int x = 0; x < NST; x++)
                theta[((size_t)(c * NST + x)) * P + p] *= gpu_scaling_threshold();
        } else if (s > mn + 1) {
            for (int x = 0; x < NST; x++)
                theta[((size_t)(c * NST + x)) * P + p] = 0.0;
        }
    }
    bufscale[p] = (double)mn * logthr;
}

/* ------------------------------------------------------------------------------------
 * val0 / val1 / val2
 * ---------------------------------------------------------------------------------- */

/* Branch log-likelihood form: val0[c*N+i] = exp(eval[i] * (rate_c*len)) * prop_c.
   IQ-TREE forms the product rate*len first here and (eval*rate) first in the derivative
   path; the results differ only in rounding, which is far below the 1e-4 gate.
   NST is used only to split the flat block index into (category, eigenvalue). */
template <int NST>
__global__ void k_val0(double *val0, const double * __restrict__ eval,
                       const double * __restrict__ rates, const double * __restrict__ props,
                       double len, int block) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block) return;
    int c = idx / NST;
    int i = idx - c * NST;
    val0[idx] = exp(eval[i] * (rates[c] * len)) * props[c];
}

/* Analytic derivative coefficients: cof = eval[i]*rate_c, val0 = exp(cof*t)*prop_c,
   val1 = cof*val0, val2 = cof*val1.  prop_c already carries the (1 - p_invar) factor,
   so it must NOT be applied again anywhere downstream. */
template <int NST>
__global__ void k_val012(double *val, const double * __restrict__ eval,
                         const double * __restrict__ rates, const double * __restrict__ props,
                         double len, int block) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block) return;
    int c = idx / NST;
    int i = idx - c * NST;
    double cof   = eval[i] * rates[c];
    double lhval = exp(cof * len) * props[c];
    double dfval = cof * lhval;
    val[idx]             = lhval;
    val[block + idx]     = dfval;
    val[2 * block + idx] = cof * dfval;
}

/* Device-len twins of k_val0 / k_val012 (STEP 2 -- Newton-loop replay). Identical
   arithmetic; the candidate branch length is read from the device scalar d_len[0] instead of a
   by-value argument, so the fixed 3-kernel derivative/logl sequence can be captured ONCE and
   replayed for every Newton iteration -- the host only refreshes d_len before each replay. The
   by-value k_val0/k_val012 above are untouched and still used on every non-replay path. */
template <int NST>
__global__ void k_val0_dev(double *val0, const double * __restrict__ eval,
                           const double * __restrict__ rates, const double * __restrict__ props,
                           const double * __restrict__ d_len, int block) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block) return;
    int c = idx / NST;
    int i = idx - c * NST;
    double len = d_len[0];
    val0[idx] = exp(eval[i] * (rates[c] * len)) * props[c];
}

template <int NST>
__global__ void k_val012_dev(double *val, const double * __restrict__ eval,
                             const double * __restrict__ rates, const double * __restrict__ props,
                             const double * __restrict__ d_len, int block) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= block) return;
    int c = idx / NST;
    int i = idx - c * NST;
    double len   = d_len[0];
    double cof   = eval[i] * rates[c];
    double lhval = exp(cof * len) * props[c];
    double dfval = cof * lhval;
    val[idx]             = lhval;
    val[block + idx]     = dfval;
    val[2 * block + idx] = cof * dfval;
}

/* ------------------------------------------------------------------------------------
 * On-device Newton step (STEP 3): one iteration of minimizeNewton
 * ----------------------------------------------------------------------------------
 * A single-thread kernel that ports Optimization::minimizeNewton (utils/optimization.cpp:422-495)
 * one iteration at a time. It runs at the END of each conditional-WHILE body pass, AFTER the
 * derivative reduction has written d_redout = {logL, dlogL/dt, d2logL/dt2}. It consumes those,
 * updates the NewtonState, writes the next candidate length to d_len[0], and sets the loop
 * condition (cudaGraphSetConditional handle = keep-going ? 1 : 0). The body's next pass then
 * re-evaluates the derivative at the new length. Because d_redout is produced by the SAME
 * reduction kernels the host path uses, and every scalar operation below is the SAME IEEE double
 * arithmetic in the SAME order as minimizeNewton, the sequence of candidate lengths -- and hence
 * the converged length -- matches the host optimiser (to its tolerance, and in the common case
 * bit for bit). The cited lines are the exact statements ported.
 *
 * NEGATION: computeFuncDerv negates both derivatives (phylotree.cpp:2606-2607), so minimizeNewton
 * sees f = -dlogL/dt and df = -d2logL/dt2. We apply the same negation to d_redout[1]/[2]. */
__global__ void k_newton_step(NewtonState *st, const double * __restrict__ redout,
                              double *d_len, cudaGraphConditionalHandle handle) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;   /* single thread owns the scalar loop */

    /* f, df as minimizeNewton names them (f = 1st deriv, df = 2nd deriv), post-negation. */
    double f  = -redout[1];
    double df = -redout[2];

    /* minimizeNewton aborts via nrerror on a non-finite derivative (:435-437, :481). We cannot
       abort on-device; latch an error, stop, and let the host fall back to the host optimiser. */
    if (!isfinite(f) || !isfinite(df)) {
        st->error = 1; st->result = st->rts; st->converged = 1;
        cudaGraphSetConditional(handle, 0);
        return;
    }

    if (!st->initialized) {
        /* Iteration 0: the eval at the (clamped) initial guess. minimizeNewton:433-447. */
        st->d2l = df;                                                  /* :434 */
        if (df >= 0.0 && fabs(f) < st->xacc) {                          /* :438 */
            st->result = st->rts; st->converged = 1;
            cudaGraphSetConditional(handle, 0);
            return;
        }
        if (f < 0.0) { st->xl = st->rts; st->xh = st->x2; }            /* :439-441 */
        else         { st->xh = st->rts; st->xl = st->x1; }            /* :442-445 */
        st->dx = st->dxold = fabs(st->xh - st->xl);                    /* :447 */
        st->initialized = 1;
        /* fall through into the first loop body (j == 1) */
    } else {
        /* Bottom of a loop iteration: the eval at the new rts. minimizeNewton:482-494. */
        if (df > 0.0 && fabs(f) < st->xacc) {                          /* :482 */
            st->d2l = df; st->result = st->rts; st->converged = 1;
            cudaGraphSetConditional(handle, 0);
            return;
        }
        if (f < 0.0) st->xl = st->rts;                                 /* :491-492 */
        else if (f > 0.0) st->xh = st->rts;                           /* :493-494 */
    }

    /* One Newton/bisection step. minimizeNewton:448-468. */
    st->j += 1;                                                        /* for (j=1..maxNRStep) */
    st->rts_old = st->rts;                                             /* :449 */
    if (df <= 0.0 ||                                                   /* :451 concave */
        (((st->rts - st->xh) * df - f) * ((st->rts - st->xl) * df - f) >= 0.0)) { /* :453 out of bound */
        st->dxold = st->dx;                                            /* :456 */
        st->dx = 0.5 * (st->xh - st->xl);                             /* :457 */
        st->rts = st->xl + st->dx;                                    /* :458 */
        st->d2l = df;                                                 /* :459 */
        if (st->xl == st->rts) {                                      /* :460 return rts */
            st->result = st->rts; st->converged = 1;
            cudaGraphSetConditional(handle, 0);
            return;
        }
    } else {
        st->dxold = st->dx;                                            /* :462 */
        st->dx = f / df;                                             /* :463 */
        double temp = st->rts;                                        /* :464 */
        st->rts -= st->dx;                                            /* :465 */
        st->d2l = df;                                                 /* :466 */
        if (temp == st->rts) {                                        /* :467 return rts */
            st->result = st->rts; st->converged = 1;
            cudaGraphSetConditional(handle, 0);
            return;
        }
    }
    if (fabs(st->dx) < st->xacc || st->j == st->maxNRStep) {          /* :469 */
        st->result = st->rts_old; st->converged = 1;                  /* :475 return rts_old */
        cudaGraphSetConditional(handle, 0);
        return;
    }
    /* Not converged: evaluate the derivative at the new rts on the next body pass. */
    d_len[0] = st->rts;
    st->converged = 0;
    cudaGraphSetConditional(handle, 1);
}

/* ------------------------------------------------------------------------------------
 * Branch log-likelihood and derivatives (from theta)
 * ---------------------------------------------------------------------------------- */

/* logL_p = log(|SUM_(c,k) val0*theta| + ptn_invar[p]) + min_scale_p*log(2^-256)
   tree_lh = SUM_p ptn_freq[p] * logL_p
   Only p < npat participates: the padded lanes have zero frequency, and a log of the
   padded (zero) likelihood would be -inf, whose product with a zero frequency is NaN.
 *
 * out_site and out_cat mirror IQ-TREE's _pattern_lh and _pattern_lh_cat. The per-category
 * partial sum lh_cat is already in a register on the way to lh, so writing it out costs one
 * store; recomputing it later would cost a second full pass over theta. out_cat is
 * category-major (out_cat[c*P + p]) to keep the store coalesced, exactly like every other
 * buffer here.
 *
 * NST appears here only to split `block` into categories so that out_cat can be written; the
 * arithmetic is a flat sum over `block`, which is why the shape is right at 4 states as it
 * stands (its thread mapping is over patterns, not over the state dimension). */
template <int NST>
__global__ void k_branch_logl(const double * __restrict__ theta, const double * __restrict__ val0,
                              const double * __restrict__ invar, const double * __restrict__ freq,
                              const double * __restrict__ bufscale,
                              int P, int npat, int ncat,
                              double *out_site, double *out_cat, int save_log_value,
                              double *blocks) {
    extern __shared__ double s_val[];
    const int block = ncat * NST;
    for (int i = threadIdx.x; i < block; i += blockDim.x) s_val[i] = val0[i];
    __syncthreads();

    double acc = 0.0;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < npat; p += gridDim.x * blockDim.x) {
        double lh = 0.0;
        /* Accumulate one category at a time, which is the order the CPU kernel uses
           (lh_ptn += lh_cat[c]) and which is also what makes lh_cat observable. */
        for (int c = 0; c < ncat; c++) {
            double lh_cat = 0.0;
            const int b0 = c * NST;
            for (int k = 0; k < NST; k++)
                lh_cat = fma(s_val[b0 + k], theta[(size_t)(b0 + k) * P + p], lh_cat);
            if (out_cat) out_cat[(size_t)c * P + p] = lh_cat;
            lh += lh_cat;
        }
        double L    = fabs(lh) + invar[p];
        double logl = log(L) + bufscale[p];
        if (out_site) out_site[p] = save_log_value ? logl : L;
        acc += freq[p] * logl;
    }
    double v[1] = { acc };
    block_sum_store<1>(v, blocks);
}

/* One pass produces lh, df and ddf; finite differences would need three full likelihood
   evaluations and could not deliver a usable second derivative at all.
 *
 * NOT templated on the state count, and deliberately so: it never mentions NST at all. Its
 * inner loop is over `block` (== ncat*nstates, 16 to 800 elements), taken as a runtime
 * argument, so the one kernel is correct and well-shaped for both data types. The general
 * objection to a runtime state count -- that a 4-trip loop with an unknown bound loses its
 * unrolling and spills the per-state registers -- does not apply to a loop whose trip count
 * is `block` and whose body holds three scalars.
 *
 * grow, when non-NULL, is row branch_id of the G matrix: G[i][p] = (dL_p/dt_i)/L_p, which is
 * IQ-TREE's `df_ptn *= inv_lh_ptn` at phylokernelnew.h:2526-2532. It is already in the
 * register `dfr` on the way to the reduction, so emitting it costs one coalesced store --
 * the same argument that made _pattern_lh free in k_branch_logl. Only p < npat is written;
 * the padding beyond it was zeroed at enable and must stay zero. */
__global__ void k_derivatives(const double * __restrict__ theta, const double * __restrict__ val,
                              const double * __restrict__ invar, const double * __restrict__ freq,
                              const double * __restrict__ bufscale,
                              int P, int npat, int block, double *grow, double *blocks) {
    extern __shared__ double s_val[];
    for (int i = threadIdx.x; i < 3 * block; i += blockDim.x) s_val[i] = val[i];
    __syncthreads();
    const double *v0 = s_val;
    const double *v1 = s_val + block;
    const double *v2 = s_val + 2 * block;

    double a_lh = 0.0, a_df = 0.0, a_ddf = 0.0;
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < npat; p += gridDim.x * blockDim.x) {
        double lh = 0.0, df = 0.0, ddf = 0.0;
        for (int b = 0; b < block; b++) {
            double t = theta[(size_t)b * P + p];
            lh  = fma(v0[b], t, lh);
            df  = fma(v1[b], t, df);
            ddf = fma(v2[b], t, ddf);
        }
        double L = fabs(lh) + invar[p];
        double f = freq[p];
        double invL = 1.0 / L;
        double dfr  = df * invL;
        double ddfr = ddf * invL;
        if (grow) grow[p] = dfr;
        a_lh  += f * (log(L) + bufscale[p]);
        a_df  += f * dfr;
        a_ddf += f * (ddfr - dfr * dfr);
    }
    double v[3] = { a_lh, a_df, a_ddf };
    block_sum_store<3>(v, blocks);
}

/* ------------------------------------------------------------------------------------
 * G matrix -> Hessian
 * ---------------------------------------------------------------------------------- */

/* Gs[i][p] = G[i][p] * sqrt(ptn_freq[p]), so that Gs*Gs^T == G*diag(ptn_freq)*G^T and the
   whole thing becomes one symmetric rank-k update. ptn_freq is a site count, hence never
   negative, so the square root is always real. Gs is written PACKED (pitch npat) because
   that is what DSYRK wants as its leading dimension; the padding columns of G are zero and
   contribute nothing, so they are simply not copied. */
__global__ void k_gmat_scale(double *gs, const double * __restrict__ g,
                             const double * __restrict__ freq,
                             int nbranch, int npat, int gstride) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= npat) return;
    double s = sqrt(freq[p]);
    for (int b = 0; b < nbranch; b++)
        gs[(size_t)b * npat + p] = g[(size_t)b * gstride + p] * s;
}

/* DSYRK fills one triangle only. The host reads the full nbranch^2 block, so the other
   triangle is mirrored here rather than left holding whatever cudaMalloc returned.
   Which triangle is which: cuBLAS is column-major, so CUBLAS_FILL_MODE_LOWER writes the
   offsets i + j*n with i >= j, and the same offsets read row-major (row = j, col = i) are
   the UPPER triangle. Hence the copy below goes upper -> lower in row-major terms. The
   result is symmetric either way, which is exactly why the caller may read it in whichever
   order it likes. */
__global__ void k_symmetrize(double *h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n || j <= i) return;
    h[(size_t)j * n + i] = h[(size_t)i * n + j];
}

/* ------------------------------------------------------------------------------------
 * Host helpers
 * ---------------------------------------------------------------------------------- */

static inline int div_up(int a, int b) { return (a + b - 1) / b; }
static inline size_t div_up_sz(size_t a, size_t b) { return (a + b - 1) / b; }

static int alloc_zero(void **p, size_t bytes, const char *what) {
    cudaError_t e = cudaMalloc(p, bytes);
    if (e != cudaSuccess) return lik_fail(what, e);
    e = cudaMemset(*p, 0, bytes);
    if (e != cudaSuccess) return lik_fail(what, e);
    return 0;
}

/* Shared-memory footprint of the GENERIC k_partial for a given tip/internal combination.
   14.7 kB at 20 states (occupancy-relevant), 1.4 kB at 4 (irrelevant -- and the 4-state
   specialisation uses none at all). */
static size_t partial_smem(int nstates, bool ltip, bool rtip) {
    const size_t n = (size_t)nstates;
    size_t d = (ltip ? 0 : n * n) + (rtip ? 0 : n * n) + n * n + n * PTILE;
    return d * sizeof(double);
}

/* THE ONE PLACE THAT KNOWS THERE ARE TWO KERNEL SHAPES.
 *
 * Every bifurcating case (tip-tip, tip-internal, internal-internal) launches through here, so
 * the choice of state count and of shape is made once rather than at each of the three call
 * sites -- and, more importantly, so that the surrounding wrapper code (scale-counter seeding,
 * the cherry case's "no rescale, zero the counters" rule, run_rescale) has exactly one form.
 *
 * Note the two grids. The generic shape divides the stride exactly, because the stride is
 * rounded to PTILE; the 4-state shape rounds UP to PTILE4 and relies on the kernel's early
 * return, which is sound only because that kernel has no __syncthreads. Both cover every
 * lane p < stride, including the padding beyond npat -- the padding must be computed, not
 * skipped, because its tip state is STATE_UNKNOWN and the rescale that follows reads the
 * whole stride.
 *
 * These are function templates, so they must live outside the extern "C" block below.
 */
template <bool LTIP, bool RTIP>
static int launch_partial(IqtreeGpuLik *c, const PartialArgs &a, const char *what) {
    if (c->cfg.nstates == NST_DNA && !c->force_generic) {
        dim3 grid(div_up(c->stride, PTILE4), c->cfg.ncat);
        k_partial4<LTIP, RTIP><<<grid, PTILE4, 0, g_stream>>>(a);
    } else if (c->cfg.nstates == NST_DNA) {
        dim3 grid(c->stride / PTILE, c->cfg.ncat), blk(PTILE, NST_DNA);
        k_partial<NST_DNA, LTIP, RTIP><<<grid, blk, partial_smem(NST_DNA, LTIP, RTIP), g_stream>>>(a);
    } else {
        dim3 grid(c->stride / PTILE, c->cfg.ncat), blk(PTILE, NST_AA);
        k_partial<NST_AA, LTIP, RTIP><<<grid, blk, partial_smem(NST_AA, LTIP, RTIP), g_stream>>>(a);
    }
    KERNEL_CHECK(what);
    return 0;
}

/* Same two-shape choice for one multifurcating child. */
template <bool TIP>
static int launch_mf_child(IqtreeGpuLik *c, double *u, const double *ech, const double *leaf,
                           const int *state, const double *in, const char *what) {
    if (c->cfg.nstates == NST_DNA && !c->force_generic) {
        dim3 grid(div_up(c->stride, PTILE4), c->cfg.ncat);
        k_mf_child4<TIP><<<grid, PTILE4, 0, g_stream>>>(u, ech, leaf, state, in, c->stride, c->block);
    } else if (c->cfg.nstates == NST_DNA) {
        dim3 grid(c->stride / PTILE, c->cfg.ncat), blk(PTILE, NST_DNA);
        const size_t smem = (size_t)((TIP ? 0 : NST_DNA * NST_DNA) + NST_DNA * PTILE)
                          * sizeof(double);
        k_mf_child<NST_DNA, TIP><<<grid, blk, smem, g_stream>>>(u, ech, leaf, state, in,
                                                      c->stride, c->block);
    } else {
        dim3 grid(c->stride / PTILE, c->cfg.ncat), blk(PTILE, NST_AA);
        const size_t smem = (size_t)((TIP ? 0 : NST_AA * NST_AA) + NST_AA * PTILE)
                          * sizeof(double);
        k_mf_child<NST_AA, TIP><<<grid, blk, smem, g_stream>>>(u, ech, leaf, state, in,
                                                     c->stride, c->block);
    }
    KERNEL_CHECK(what);
    return 0;
}

extern "C" {

const char* iqtree_gpu_lik_last_error(void) { return g_lik_err; }

/* ------------------------------------------------------------------------------------
 * Device introspection
 * ---------------------------------------------------------------------------------- */

int iqtree_gpu_device_count(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

const char* iqtree_gpu_device_name(void) {
    static char name[320] = "none";   /* cudaDeviceProp::name is char[256] plus the suffix */
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        snprintf(name, sizeof(name), "%s (sm_%d%d)", p.name, p.major, p.minor);
    }
    return name;
}

int iqtree_gpu_device_mem(size_t *free_bytes, size_t *total_bytes) {
    size_t f = 0, t = 0;
    cudaError_t e = cudaMemGetInfo(&f, &t);
    if (e != cudaSuccess) return lik_fail("device_mem/cudaMemGetInfo", e);
    if (free_bytes)  *free_bytes  = f;
    if (total_bytes) *total_bytes = t;
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------------------- */

int iqtree_gpu_lik_create(IqtreeGpuLik **out, const IqtreeGpuLikConfig *cfg) {
    if (!out || !cfg) return lik_fail_msg("create: null argument");
    *out = NULL;
    if (cfg->nstates != NST_AA && cfg->nstates != NST_DNA)
        return lik_fail_msg("create: only nstates=20 (amino acids) and nstates=4 (DNA) are "
                            "supported; codon/morphological/mixture data must fall back to "
                            "the CPU kernel");
    if (cfg->ncat <= 0 || cfg->npat <= 0)
        return lik_fail_msg("create: ncat and npat must be positive");
    if (cfg->num_tip_states <= 0 || cfg->state_unknown < 0 ||
        cfg->state_unknown >= cfg->num_tip_states)
        return lik_fail_msg("create: state_unknown must index a row of tip_partial_lh");
    if (cfg->echild_slots <= 0 || cfg->tip_slots <= 0)
        return lik_fail_msg("create: echild_slots and tip_slots must be positive");
    /* Capacity of the arena in rate categories. Anything below the current ncat is
       meaningless, so it is read as "no headroom". */
    int ncat_cap = (cfg->ncat_cap > cfg->ncat) ? cfg->ncat_cap : cfg->ncat;

    /* The derivative kernel stages val0|val1|val2 in dynamic shared memory, so ncat is
       bounded by the 48 KB default shared-memory allocation. ncat is a handful in
       practice (+G4, +R10); this only exists so an absurd value fails loudly at setup
       rather than as a launch error deep inside a tree search. The bound is applied to the
       CAPACITY, because set_ncat() may later run the engine at any ncat up to it, and it is
       computed from the RUNTIME state count: 102 categories at 20 states, 512 at 4. */
    if (3u * (unsigned)ncat_cap * (unsigned)cfg->nstates * sizeof(double) > 48u * 1024u)
        return lik_fail_msg("create: ncat too large for the shared-memory staging of val0/1/2");

    if (cfg->device >= 0) CUDA_TRY(cudaSetDevice(cfg->device), "create/cudaSetDevice");

    IqtreeGpuLik *c = (IqtreeGpuLik*)calloc(1, sizeof(IqtreeGpuLik));
    if (!c) return lik_fail_msg("create: host allocation failed");
    c->cfg = *cfg;

    /* Round patterns up to a warp so that every tile is full and no kernel needs a
       partially-populated first warp. */
    c->stride        = div_up(cfg->npat, PTILE) * PTILE;
    c->block         = cfg->ncat * cfg->nstates;
    c->partial_elems = (size_t)c->block * (size_t)c->stride;
    c->scale_elems   = (size_t)c->stride * (size_t)(cfg->safe_numeric ? cfg->ncat : 1);
    c->ncat_cap          = ncat_cap;
    c->block_cap         = ncat_cap * cfg->nstates;
    c->partial_elems_cap = (size_t)c->block_cap * (size_t)c->stride;
    c->scale_elems_cap   = (size_t)c->stride * (size_t)(cfg->safe_numeric ? ncat_cap : 1);
    c->cfg.ncat_cap      = ncat_cap;
    c->rates_ncat        = 0;   /* nothing uploaded yet */
    /* 0 means "one slot", so that callers written against the single-theta API still work. */
    c->theta_slots   = (cfg->theta_slots > 0) ? cfg->theta_slots : 1;
    c->theta_slot    = 0;

    /* Everything below that carries a category dimension is sized from the CAPACITY. */
    const size_t ns  = (size_t)cfg->nstates;
    const size_t ns2 = ns * ns;
    int rc = 0;
    rc |= alloc_zero((void**)&c->d_eval,       ns * sizeof(double),                  "create/eval");
    rc |= alloc_zero((void**)&c->d_evec,       ns2 * sizeof(double),                 "create/evec");
    rc |= alloc_zero((void**)&c->d_inv_evec,   ns2 * sizeof(double),           "create/inv_evec");
    rc |= alloc_zero((void**)&c->d_rates,      (size_t)ncat_cap * sizeof(double),    "create/rates");
    rc |= alloc_zero((void**)&c->d_props,      (size_t)ncat_cap * sizeof(double),    "create/props");
    /* num_tip_states rows: 24 for protein, 19 for DNA. The DNA table stays DENSE at 19 even
       though rows 4, 5, 7 and 11 are numerically duplicates of A, C, G, T that the alignment
       parser can never emit -- the row index IS the state value, so it cannot be compacted. */
    rc |= alloc_zero((void**)&c->d_tip_lh,     (size_t)cfg->num_tip_states * ns * sizeof(double),
                     "create/tip_partial_lh");
    rc |= alloc_zero((void**)&c->d_ptn_freq,   (size_t)c->stride * sizeof(double),   "create/ptn_freq");
    rc |= alloc_zero((void**)&c->d_ptn_invar,  (size_t)c->stride * sizeof(double),   "create/ptn_invar");
    rc |= alloc_zero((void**)&c->d_tip_states,
                     (size_t)cfg->tip_slots * c->stride * sizeof(int),               "create/tip_states");
    rc |= alloc_zero((void**)&c->d_echild,
                     (size_t)cfg->echild_slots * ncat_cap * ns2 * sizeof(double),
                     "create/echild");
    rc |= alloc_zero((void**)&c->d_leaf,
                     (size_t)cfg->echild_slots * cfg->num_tip_states * c->block_cap * sizeof(double),
                     "create/partial_lh_leaf");
    rc |= alloc_zero((void**)&c->d_theta,
                     (size_t)c->theta_slots * c->partial_elems_cap * sizeof(double), "create/theta");
    rc |= alloc_zero((void**)&c->d_bufscale,
                     (size_t)c->theta_slots * c->stride * sizeof(double),            "create/bufscale");
    rc |= alloc_zero((void**)&c->d_val,    3 * (size_t)c->block_cap * sizeof(double),"create/val");
    rc |= alloc_zero((void**)&c->d_redblocks,  RED_BLOCKS * 3 * sizeof(double),      "create/redblocks");
    rc |= alloc_zero((void**)&c->d_redout,     3 * sizeof(double),                   "create/redout");
    /* Eager (1 double): the Newton-loop replay graphs bake in a read of d_len, and no cudaMalloc
       may happen inside a capture, so the buffer must exist before the first capture. */
    rc |= alloc_zero((void**)&c->d_len,        sizeof(double),                       "create/len");
    rc |= alloc_zero((void**)&c->d_newton,     sizeof(NewtonState),                  "create/newton");
    if (rc) { iqtree_gpu_lik_destroy(c); return -1; }

    /* Tip state columns default to STATE_UNKNOWN, which is also what the padded lanes
       must contain: IQ-TREE treats every pattern beyond orig_nptn as unknown. */
    {
        size_t n = (size_t)cfg->tip_slots * c->stride;
        k_fill_int<<<div_up_sz(n, 256), 256, 0, g_stream>>>(c->d_tip_states, cfg->state_unknown, n);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { iqtree_gpu_lik_destroy(c); return lik_fail("create/fill states", e); }
    }

    *out = c;
    g_lik_err[0] = '\0';
    return 0;
}

void iqtree_gpu_lik_destroy(IqtreeGpuLik *c) {
    if (!c) return;
    cudaFree(c->d_eval);       cudaFree(c->d_evec);      cudaFree(c->d_inv_evec);
    cudaFree(c->d_rates);      cudaFree(c->d_props);     cudaFree(c->d_tip_lh);
    cudaFree(c->d_ptn_freq);   cudaFree(c->d_ptn_invar); cudaFree(c->d_tip_states);
    cudaFree(c->d_echild);     cudaFree(c->d_leaf);
    cudaFree(c->d_theta);      cudaFree(c->d_bufscale);  cudaFree(c->d_val);
    cudaFree(c->d_u);          cudaFree(c->d_sitelh);    cudaFree(c->d_lhcat);
    cudaFree(c->d_gmat);       cudaFree(c->d_hess);
    cudaFree(c->d_redblocks);  cudaFree(c->d_redout);
    cudaFree(c->d_branch_len); cudaFree(c->d_len); cudaFree(c->d_newton);
    if (c->newton_exec) cudaGraphExecDestroy(c->newton_exec);
    if (c->cublas_ready) cublasDestroy(c->cublas);
    free(c);
}

/* ------------------------------------------------------------------------------------
 * Launch accounting -- public read-out (STEP 1)
 * ---------------------------------------------------------------------------------- */
long long iqtree_gpu_lik_launch_count(void) {
    return g_kernel_launches.load(std::memory_order_relaxed);
}
long long iqtree_gpu_lik_graph_replay_count(void) {
    return g_graph_replays.load(std::memory_order_relaxed);
}
void iqtree_gpu_lik_reset_launch_count(void) {
    g_kernel_launches.store(0, std::memory_order_relaxed);
    g_graph_replays.store(0, std::memory_order_relaxed);
}

/* ------------------------------------------------------------------------------------
 * CUDA-graph capture / replay facility (STEP 2)
 * ----------------------------------------------------------------------------------
 * WHAT THIS IS. A thin, engine-owned wrapper around CUDA stream capture. Between
 * capture_begin() and capture_end() every kernel launch this file issues is recorded --
 * unchanged, same kernels, same order, same arguments, same device pointers -- into a
 * cudaGraph, which capture_end() instantiates into a replayable cudaGraphExec. graph_launch()
 * then replays that whole DAG with ONE driver call instead of re-issuing every launch. For a
 * DNA traversal, whose per-node kernels do ~3 us of work but pay ~7 us of launch latency, that
 * is the difference between paying the latency ~2n*6 times per evaluation and paying it once.
 *
 * WHY IT IS EQUIVALENT BY CONSTRUCTION. Capture does not change WHAT is computed, only HOW the
 * launches are issued. The recorded nodes are the identical __global__ functions with the
 * identical grid/block/shared-memory/argument bytes they would have had un-captured; a replay
 * runs them in the identical stream order. No kernel math, tolerance, scaling, or reduction
 * order changes. The unit test (gpu_kernels_test.cu) asserts this directly: it runs an
 * evaluation un-captured, then captured-and-replayed, and requires byte-identical device
 * output. That test is the gate; it needs a GPU to run.
 *
 * THE CAPTURE STREAM. Stream capture is illegal on the legacy default stream, so capture_begin
 * creates an explicit non-blocking stream, points the file-static g_stream at it, and every
 * launch (which already reads g_stream, defaulting to 0) is thereby recorded. capture_end
 * restores g_stream to 0. NO synchronous CUDA call may occur inside the captured region:
 * the traversal wrappers issue only launches plus device-to-device copies, and the three
 * blocking cudaMemset/cudaMemcpy that used to sit on the partial path were converted to the
 * *Async forms above so they record cleanly. The blocking result copies (branch_logl /
 * derivatives download the scalar logL/df/ddf) are NOT part of a traversal and must stay
 * OUTSIDE any captured region -- capture the partial + theta sweep, replay it, THEN read back.
 *
 * VALIDITY / INVALIDATION -- the caller's responsibility, spelled out for the integration in
 * phylotreegpu.cpp (see the DESIGN NOTE below):
 *   A captured graph bakes in (a) the exact kernel SEQUENCE -- which nodes, which of the four
 *   partial cases, which echild/tip/output SLOTS -- and (b) the device POINTERS of those
 *   slots. It stays valid only while BOTH hold:
 *     1. the slot pointers are unchanged: guaranteed while gpuContextMatches() holds and no
 *        rebind/rebuild/set_ncat has run (the generation counter g_gpu.gen is the hook -- a
 *        change of gen must free the graph and force re-capture);
 *     2. the traversed node set is unchanged: the SAME (node,dad) root AND the SAME set of
 *        dirty partials. This is the subtle one -- see the DESIGN NOTE.
 *   Data that flows through DEVICE BUFFERS (model eigen matrices d_evec/d_eval/d_props/d_rates,
 *   pattern data, and -- once plumbed -- branch lengths) may change freely between replays: the
 *   graph reads whatever those buffers hold at replay time. Data that flows through kernel
 *   ARGUMENTS BY VALUE is frozen at capture; today that is the branch length `len` of k_echild,
 *   k_val0 and k_val012, which is why a replayed traversal needs branch lengths moved into a
 *   device buffer first (DESIGN NOTE, step "device-resident len").
 *
 * A graph is an opaque handle owned by the caller; free it with graph_free(). */
struct IqtreeGpuGraph {
    cudaGraphExec_t exec;
};

int iqtree_gpu_lik_graph_capture_begin(IqtreeGpuLik *c) {
    if (!c) return lik_fail_msg("graph_capture_begin: null context");
    if (g_stream != 0)
        return lik_fail_msg("graph_capture_begin: a capture is already in progress");
    /* Non-blocking so the capture stream does not implicitly serialise with the legacy
       default stream during capture. */
    CUDA_TRY(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking),
             "graph_capture_begin/stream");
    cudaError_t e = cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        cudaStreamDestroy(g_stream);
        g_stream = 0;
        return lik_fail("graph_capture_begin/beginCapture", e);
    }
    return 0;
}

int iqtree_gpu_lik_graph_capture_end(IqtreeGpuLik *c, IqtreeGpuGraph **out) {
    if (!c || !out) return lik_fail_msg("graph_capture_end: null argument");
    *out = NULL;
    if (g_stream == 0)
        return lik_fail_msg("graph_capture_end: no capture in progress");
    cudaStream_t s = g_stream;
    cudaGraph_t graph = NULL;
    cudaError_t e = cudaStreamEndCapture(s, &graph);
    /* Whatever happens, stop steering launches onto the capture stream. */
    g_stream = 0;
    if (e != cudaSuccess) { cudaStreamDestroy(s); return lik_fail("graph_capture_end/endCapture", e); }

    IqtreeGpuGraph *g = (IqtreeGpuGraph*)calloc(1, sizeof(IqtreeGpuGraph));
    if (!g) { cudaGraphDestroy(graph); cudaStreamDestroy(s); return lik_fail_msg("graph_capture_end: host alloc"); }
    e = cudaGraphInstantiate(&g->exec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    if (e != cudaSuccess) { free(g); return lik_fail("graph_capture_end/instantiate", e); }
    *out = g;
    return 0;
}

int iqtree_gpu_lik_graph_launch(IqtreeGpuLik *c, IqtreeGpuGraph *g) {
    if (!c || !g) return lik_fail_msg("graph_launch: null argument");
    /* Replay on the legacy default stream (g_stream == 0 here), so the blocking result
       copies that follow a traversal continue to order correctly against it. */
    CUDA_TRY(cudaGraphLaunch(g->exec, 0), "graph_launch");
    g_graph_replays.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

void iqtree_gpu_lik_graph_free(IqtreeGpuGraph *g) {
    if (!g) return;
    if (g->exec) cudaGraphExecDestroy(g->exec);
    free(g);
}

/* ==================================================================================
 * DESIGN NOTE -- why the launch storm is real, and how to spend the graph facility
 * ==================================================================================
 * STEP 1 (done): kernel_launches in the GPUSTATS line counts every device launch. Run any DNA
 * dataset with `-gpu` and read it back: a full tree search issues, per full-tree likelihood
 * evaluation, ~2n internal nodes * (2 echild + up to 2 leaf + 1 partial + 1 scale-init +
 * 1 rescale) ~= 6-8 launches per node, and ModelFinder + NNI + UFBoot drive thousands of such
 * evaluations, so the count is in the millions. Amino acid pays the SAME launch count but
 * hides it: each launch does ~25x more arithmetic (20x20 vs 4x4), so latency is ~7% of a
 * traversal instead of ~45%. Confirm this with the counter before/after; it is also how any
 * fix is measured (kernel_launches must fall, graph_replays must rise).
 *
 * STEP 2 (this facility + the wiring below): replace the per-node launch storm of a traversal
 * with a single graph replay. Two things must be handled by the phylotreegpu.cpp caller, and
 * neither can be validated without a GPU, which is why the wiring is staged behind this note:
 *
 *  (i) DEVICE-RESIDENT len. k_echild/k_val0/k_val012 take the branch length `len` as a by-value
 *      argument, frozen at capture. A replayed traversal must instead read len from a device
 *      buffer refreshed (one async H2D of the branch-length vector, indexed by the STABLE
 *      IQ-TREE branch id) before each cudaGraphLaunch. Only k_echild appears in a traversal
 *      (val0/val012 live in the branch_logl/derivatives endpoints, which are NOT captured), so
 *      only k_echild needs a device-len variant; add it as a sibling of the existing kernel so
 *      the amino-acid path and the unit test keep the by-value form unchanged.
 *
 *  (ii) DIRTY-SET VALIDITY -- the real subtlety, and the reason a single cached graph is NOT a
 *      drop-in for every traversal. IQ-TREE recomputes only DIRTY partials: computeTraversalInfo
 *      returns just the nodes whose partial is stale, so the node set -- and therefore the exact
 *      captured kernel sequence -- varies call to call during branch-length optimisation, even
 *      though the TOPOLOGY is fixed. Capturing one traversal and blindly replaying it for the
 *      next would compute the wrong nodes. The safe, high-value target is the FULL traversal
 *      (every internal node dirty), which recurs identically many times: notably the first
 *      evaluation of each of ModelFinder's ~1200 DNA candidate models on the fixed topology
 *      (a model change dirties ALL partials), and each fresh full likelihood after a topology
 *      accept. So: capture when traversal_info covers the full internal-node set, key the
 *      cached graph on (g_gpu.gen, full-set signature), replay it whenever the identical full
 *      set recurs under the same generation, and fall back to per-node launches for the partial
 *      (incremental) sweeps. Model parameters differ per candidate but flow through device
 *      buffers (d_evec/d_eval/d_props/d_rates, re-uploaded by set_eigen/set_rates), so the same
 *      graph is correct across models once (i) is in place.
 *
 * Everything the graph bakes in beyond (i) is already stable for a fixed (gen, full node set):
 * the slot pointers, the case selection, the grid/block shapes. gpuContextMatches()/g_gpu.gen
 * already exist as the invalidation hook; a gen change must iqtree_gpu_lik_graph_free() the
 * cached graph. This note is the contract the wiring must honour; the unit test proves the
 * facility itself is faithful. */

int iqtree_gpu_lik_set_theta_slot(IqtreeGpuLik *c, int slot) {
    if (!c) return lik_fail_msg("set_theta_slot: null context");
    if (slot < 0 || slot >= c->theta_slots)
        return lik_fail_msg("set_theta_slot: slot out of range");
    c->theta_slot = slot;
    return 0;
}

/* Re-point the engine at a new rate-category count. No allocation, no free: see the layout
   argument in gpu_kernels.h -- the ncat < ncat_cap arrays are an exact PREFIX of the
   ncat_cap ones, so only the geometry the kernels launch with has to move.
 *
 * What is DATA rather than capacity, and therefore has to be dealt with here:
 *
 *   d_rates / d_props   Per-category values. Growing ncat would expose entries left over
 *                       from an earlier, longer run. They are zeroed and marked stale; every
 *                       kernel that reads them refuses to run until set_rates() has been
 *                       called again for this ncat. Zeroing alone would be silent (prop 0
 *                       gives a plausible-looking likelihood of 0), which is why the stale
 *                       flag exists as well.
 *   d_theta/d_bufscale  Written in full before every read, and the caller is required to
 *                       clear its theta_computed flag. Zeroed anyway so that a caller which
 *                       forgets gets log(0) = -inf -- a loud non-finite likelihood -- rather
 *                       than a wrong number computed from the old category layout.
 *   partials/counters   Owned by the caller (iqtree_gpu_lik_alloc_partial). The engine cannot
 *                       reach them; invalidating them is invariant (1) in the header.
 *   d_echild / d_leaf   Rewritten by branch_precompute before every use, per branch.
 *   d_val / d_lhcat     Rewritten by the kernel that reads them, in the same call. */
int iqtree_gpu_lik_set_ncat(IqtreeGpuLik *c, int ncat) {
    if (!c) return lik_fail_msg("set_ncat: null context");
    if (ncat <= 0) return lik_fail_msg("set_ncat: ncat must be positive");
    if (ncat > c->ncat_cap)
        return lik_fail_msg("set_ncat: ncat exceeds the capacity the arena was built for");
    if (ncat == c->cfg.ncat) return 0;

    c->cfg.ncat      = ncat;
    c->block         = ncat * c->cfg.nstates;
    c->partial_elems = (size_t)c->block * (size_t)c->stride;
    c->scale_elems   = (size_t)c->stride * (size_t)(c->cfg.safe_numeric ? ncat : 1);

    c->rates_ncat = 0;
    CUDA_TRY(cudaMemset(c->d_rates, 0, (size_t)c->ncat_cap * sizeof(double)), "set_ncat/rates");
    CUDA_TRY(cudaMemset(c->d_props, 0, (size_t)c->ncat_cap * sizeof(double)), "set_ncat/props");
    CUDA_TRY(cudaMemset(c->d_theta, 0,
                        (size_t)c->theta_slots * c->partial_elems_cap * sizeof(double)),
             "set_ncat/theta");
    CUDA_TRY(cudaMemset(c->d_bufscale, 0,
                        (size_t)c->theta_slots * c->stride * sizeof(double)),
             "set_ncat/bufscale");
    return 0;
}

/* Detach the engine from the tree it was serving, keeping every allocation. See the long
   contract next to the declaration in gpu_kernels.h; the short version is that ALL of the
   cudaMalloc'd sizes here are functions of geometry (npat, ncat_cap, safe_numeric,
   echild_slots, tip_slots, num_tip_states) and none of them mention the tree, so a tree with
   the same geometry can reuse them as they stand.
 *
 * The buffers below are the ones holding values that were computed FOR the old tree. Each is
 * either zeroed (so a caller that skips its own invalidation gets a non-finite likelihood
 * rather than a plausible wrong one) or marked stale through the existing rates_ready guard
 * (so it fails at the call that would have read it, naming the reason). The two things that
 * are deliberately NOT touched -- the pooled partial/scale slots and d_tip_states -- are the
 * caller's obligations (1) and (2) in the header, because zeroing them would cost exactly the
 * rebuild this entry point exists to avoid. */
int iqtree_gpu_lik_rebind(IqtreeGpuLik *c) {
    if (!c) return lik_fail_msg("rebind: null context");
    /* G rows are indexed by the branch id of ONE tree. There is no meaningful way to carry
       them across, and silently keeping them would corrupt the exported Hessian, so refuse. */
    if (c->d_gmat)
        return lik_fail_msg("rebind: a G matrix is enabled (--dating mcmctree); its rows are "
                            "indexed by the branch id of one tree and cannot be rebound");

    /* Per-category DATA, not capacity. Same treatment as set_ncat: zero AND mark stale, because
       zeroing alone is silent (prop 0 yields a plausible-looking likelihood of 0). */
    c->rates_ncat = 0;
    CUDA_TRY(cudaMemset(c->d_rates, 0, (size_t)c->ncat_cap * sizeof(double)), "rebind/rates");
    CUDA_TRY(cudaMemset(c->d_props, 0, (size_t)c->ncat_cap * sizeof(double)), "rebind/props");

    /* theta / bufscale describe one BRANCH of the old tree. */
    CUDA_TRY(cudaMemset(c->d_theta, 0,
                        (size_t)c->theta_slots * c->partial_elems_cap * sizeof(double)),
             "rebind/theta");
    CUDA_TRY(cudaMemset(c->d_bufscale, 0,
                        (size_t)c->theta_slots * c->stride * sizeof(double)),
             "rebind/bufscale");
    c->theta_slot = 0;

    /* echild and the collapsed leaf tables hold exp(eval*rate*len) for the OLD tree's branch
       lengths. branch_precompute rewrites them before every use, so this is belt and braces --
       but they are a few tens of kB, so the loud option is free. */
    CUDA_TRY(cudaMemset(c->d_echild, 0,
                        (size_t)c->cfg.echild_slots * c->ncat_cap
                            * (size_t)c->cfg.nstates * (size_t)c->cfg.nstates * sizeof(double)),
             "rebind/echild");
    CUDA_TRY(cudaMemset(c->d_leaf, 0,
                        (size_t)c->cfg.echild_slots * c->cfg.num_tip_states * c->block_cap
                            * sizeof(double)),
             "rebind/partial_lh_leaf");

    /* Lazy scratch. Rewritten by the kernel that reads it, in the same call; zeroed anyway so
       that nothing from the old tree can survive in device memory at all. */
    if (c->d_u)
        CUDA_TRY(cudaMemset(c->d_u, 0, c->partial_elems_cap * sizeof(double)), "rebind/u");
    if (c->d_sitelh)
        CUDA_TRY(cudaMemset(c->d_sitelh, 0, (size_t)c->stride * sizeof(double)), "rebind/sitelh");
    if (c->d_lhcat)
        CUDA_TRY(cudaMemset(c->d_lhcat, 0, (size_t)c->ncat_cap * c->stride * sizeof(double)),
                 "rebind/lhcat");
    return 0;
}

/* Guard for every kernel that reads d_rates / d_props. A host-side int compare, so it costs
   nothing on the hot path, and it turns "set_ncat without set_rates" from a silent wrong
   answer into an error at the call that would have read the stale categories. */
static inline int rates_ready(IqtreeGpuLik *c, const char *what) {
    if (c->rates_ncat == c->cfg.ncat) return 0;
    snprintf(g_lik_err, sizeof(g_lik_err),
             "%s: rate/proportion vectors are stale (uploaded for ncat=%d, engine is at "
             "ncat=%d); call iqtree_gpu_lik_set_rates after iqtree_gpu_lik_set_ncat",
             what, c->rates_ncat, c->cfg.ncat);
    return -1;
}

int    iqtree_gpu_lik_pattern_stride(const IqtreeGpuLik *c) { return c ? c->stride : 0; }
int    iqtree_gpu_lik_block         (const IqtreeGpuLik *c) { return c ? c->block  : 0; }
int    iqtree_gpu_lik_nstates       (const IqtreeGpuLik *c) { return c ? c->cfg.nstates : 0; }
int    iqtree_gpu_lik_ncat          (const IqtreeGpuLik *c) { return c ? c->cfg.ncat : 0; }
int    iqtree_gpu_lik_ncat_cap      (const IqtreeGpuLik *c) { return c ? c->ncat_cap : 0; }
size_t iqtree_gpu_lik_partial_elems (const IqtreeGpuLik *c) { return c ? c->partial_elems : 0; }
size_t iqtree_gpu_lik_partial_bytes (const IqtreeGpuLik *c) { return c ? c->partial_elems * sizeof(double) : 0; }
size_t iqtree_gpu_lik_scale_elems   (const IqtreeGpuLik *c) { return c ? c->scale_elems : 0; }
size_t iqtree_gpu_lik_scale_bytes   (const IqtreeGpuLik *c) { return c ? c->scale_elems : 0; }
size_t iqtree_gpu_lik_partial_bytes_cap(const IqtreeGpuLik *c) {
    return c ? c->partial_elems_cap * sizeof(double) : 0;
}
size_t iqtree_gpu_lik_scale_bytes_cap  (const IqtreeGpuLik *c) { return c ? c->scale_elems_cap : 0; }

size_t iqtree_gpu_lik_slot_bytes(int npat, int ncat, int nstates, int safe_numeric) {
    if (npat <= 0 || ncat <= 0 || nstates <= 0) return 0;
    size_t stride = (size_t)div_up(npat, PTILE) * PTILE;
    size_t partial = (size_t)ncat * (size_t)nstates * stride * sizeof(double);
    size_t scale   = stride * (size_t)(safe_numeric ? ncat : 1);
    return partial + scale;
}

int iqtree_gpu_lik_force_generic_shape(IqtreeGpuLik *c, int use_generic) {
    if (!c) return lik_fail_msg("force_generic_shape: null context");
    c->force_generic = use_generic ? 1 : 0;
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Uploads
 * ---------------------------------------------------------------------------------- */

int iqtree_gpu_lik_set_eigen(IqtreeGpuLik *c, const double *eval,
                             const double *evec, const double *inv_evec) {
    if (!c || !eval || !evec || !inv_evec) return lik_fail_msg("set_eigen: null argument");
    const size_t ns  = (size_t)c->cfg.nstates;
    const size_t ns2 = ns * ns;
    CUDA_TRY(cudaMemcpy(c->d_eval, eval, ns * sizeof(double), cudaMemcpyHostToDevice),
             "set_eigen/eval");
    CUDA_TRY(cudaMemcpy(c->d_evec, evec, ns2 * sizeof(double), cudaMemcpyHostToDevice),
             "set_eigen/evec");
    CUDA_TRY(cudaMemcpy(c->d_inv_evec, inv_evec, ns2 * sizeof(double), cudaMemcpyHostToDevice),
             "set_eigen/inv_evec");
    return 0;
}

int iqtree_gpu_lik_set_rates(IqtreeGpuLik *c, const double *rates, const double *props) {
    if (!c || !rates || !props) return lik_fail_msg("set_rates: null argument");
    /* Exactly ncat entries: the tail up to ncat_cap is capacity, not data, and must stay
       zero so that nothing can quietly read a category the current model does not have. */
    CUDA_TRY(cudaMemcpy(c->d_rates, rates, c->cfg.ncat * sizeof(double), cudaMemcpyHostToDevice),
             "set_rates/rates");
    CUDA_TRY(cudaMemcpy(c->d_props, props, c->cfg.ncat * sizeof(double), cudaMemcpyHostToDevice),
             "set_rates/props");
    c->rates_ncat = c->cfg.ncat;
    return 0;
}

int iqtree_gpu_lik_set_tip_partials(IqtreeGpuLik *c, const double *tip_partial_lh) {
    if (!c || !tip_partial_lh) return lik_fail_msg("set_tip_partials: null argument");
    CUDA_TRY(cudaMemcpy(c->d_tip_lh, tip_partial_lh,
                        (size_t)c->cfg.num_tip_states * (size_t)c->cfg.nstates * sizeof(double),
                        cudaMemcpyHostToDevice), "set_tip_partials");
    return 0;
}

int iqtree_gpu_lik_set_pattern_data(IqtreeGpuLik *c, const double *ptn_freq,
                                    const double *ptn_invar) {
    if (!c || !ptn_freq || !ptn_invar) return lik_fail_msg("set_pattern_data: null argument");
    size_t n = (size_t)c->cfg.npat * sizeof(double);
    CUDA_TRY(cudaMemset(c->d_ptn_freq, 0, (size_t)c->stride * sizeof(double)),
             "set_pattern_data/zero freq");
    CUDA_TRY(cudaMemset(c->d_ptn_invar, 0, (size_t)c->stride * sizeof(double)),
             "set_pattern_data/zero invar");
    CUDA_TRY(cudaMemcpy(c->d_ptn_freq, ptn_freq, n, cudaMemcpyHostToDevice),
             "set_pattern_data/freq");
    CUDA_TRY(cudaMemcpy(c->d_ptn_invar, ptn_invar, n, cudaMemcpyHostToDevice),
             "set_pattern_data/invar");
    return 0;
}

int iqtree_gpu_lik_set_tip_states(IqtreeGpuLik *c, int tip_slot, const int *states) {
    if (!c || !states) return lik_fail_msg("set_tip_states: null argument");
    if (tip_slot < 0 || tip_slot >= c->cfg.tip_slots)
        return lik_fail_msg("set_tip_states: slot out of range");
    int *dst = c->d_tip_states + (size_t)tip_slot * c->stride;
    /* Padding stays at STATE_UNKNOWN from create(); only the real patterns are written. */
    CUDA_TRY(cudaMemcpy(dst, states, (size_t)c->cfg.npat * sizeof(int), cudaMemcpyHostToDevice),
             "set_tip_states");
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Buffers
 * ---------------------------------------------------------------------------------- */

/* Sized for the CAPACITY, not for the current ncat: a slot handed out at ncat=4 must still
   be large enough after a set_ncat to 10, or the very reallocation this design removes would
   come back one buffer at a time. */
double* iqtree_gpu_lik_alloc_partial(IqtreeGpuLik *c) {
    if (!c) return NULL;
    void *p = NULL;
    if (alloc_zero(&p, c->partial_elems_cap * sizeof(double), "alloc_partial")) return NULL;
    return (double*)p;
}

unsigned char* iqtree_gpu_lik_alloc_scale(IqtreeGpuLik *c) {
    if (!c) return NULL;
    void *p = NULL;
    if (alloc_zero(&p, c->scale_elems_cap, "alloc_scale")) return NULL;
    return (unsigned char*)p;
}

void iqtree_gpu_lik_free(IqtreeGpuLik *c, void *dptr) { (void)c; if (dptr) cudaFree(dptr); }

int iqtree_gpu_lik_download_partial(IqtreeGpuLik *c, double *h_dst, const double *d_src) {
    if (!c || !h_dst || !d_src) return lik_fail_msg("download_partial: null argument");
    CUDA_TRY(cudaMemcpy(h_dst, d_src, c->partial_elems * sizeof(double), cudaMemcpyDeviceToHost),
             "download_partial");
    return 0;
}

int iqtree_gpu_lik_upload_partial(IqtreeGpuLik *c, double *d_dst, const double *h_src) {
    if (!c || !h_src || !d_dst) return lik_fail_msg("upload_partial: null argument");
    CUDA_TRY(cudaMemcpy(d_dst, h_src, c->partial_elems * sizeof(double), cudaMemcpyHostToDevice),
             "upload_partial");
    return 0;
}

int iqtree_gpu_lik_download_scale(IqtreeGpuLik *c, unsigned char *h_dst,
                                  const unsigned char *d_src) {
    if (!c || !h_dst || !d_src) return lik_fail_msg("download_scale: null argument");
    CUDA_TRY(cudaMemcpy(h_dst, d_src, c->scale_elems, cudaMemcpyDeviceToHost), "download_scale");
    return 0;
}

int iqtree_gpu_lik_upload_scale(IqtreeGpuLik *c, unsigned char *d_dst,
                                const unsigned char *h_src) {
    if (!c || !h_src || !d_dst) return lik_fail_msg("upload_scale: null argument");
    CUDA_TRY(cudaMemcpy(d_dst, h_src, c->scale_elems, cudaMemcpyHostToDevice), "upload_scale");
    return 0;
}

int iqtree_gpu_lik_download_theta(IqtreeGpuLik *c, double *h_dst) {
    if (!c || !h_dst) return lik_fail_msg("download_theta: null argument");
    CUDA_TRY(cudaMemcpy(h_dst, theta_of(c), c->partial_elems * sizeof(double),
                        cudaMemcpyDeviceToHost), "download_theta");
    return 0;
}

int iqtree_gpu_lik_download_buffer_scale(IqtreeGpuLik *c, double *h_dst) {
    if (!c || !h_dst) return lik_fail_msg("download_buffer_scale: null argument");
    CUDA_TRY(cudaMemcpy(h_dst, bufscale_of(c), (size_t)c->stride * sizeof(double),
                        cudaMemcpyDeviceToHost), "download_buffer_scale");
    return 0;
}

int iqtree_gpu_lik_sync(IqtreeGpuLik *c) {
    (void)c;
    CUDA_TRY(cudaDeviceSynchronize(), "sync");
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Per-branch precompute
 * ---------------------------------------------------------------------------------- */

/* Slot pitches use the CAPACITY, so a slot base never moves when ncat changes; each slot
   then writes and reads only its leading ncat*N*N (resp. num_tip_states*block) elements. */
static inline double* echild_of(IqtreeGpuLik *c, int slot) {
    return c->d_echild + (size_t)slot * c->ncat_cap
                       * (size_t)c->cfg.nstates * (size_t)c->cfg.nstates;
}
static inline double* leaf_of(IqtreeGpuLik *c, int slot) {
    return c->d_leaf + (size_t)slot * c->cfg.num_tip_states * c->block_cap;
}

/* ------------------------------------------------------------------------------------
 * STATE-COUNT DISPATCH
 *
 * Every host wrapper below that launches an NST-templated kernel picks the instantiation
 * with a plain two-way `if (c->cfg.nstates == NST_DNA)`, in the same spirit as IQ-TREE's own
 * `switch (aln->num_states) { case 4: ...; case 20: ... }` over its KERNEL_FIX_STATES CPU
 * kernels (phylokernelfma.cpp:47-66).
 *
 * The split is deliberately confined to the launch line. The SEMANTICS -- which of the four
 * partial cases exist, what the cherry case does to the counters, where the rescale runs,
 * what the scale counters mean, when theta is renormalised -- stay in the wrapper, in ONE
 * place, shared by both state counts and by both DNA kernel shapes. That is what makes this
 * a second set of __global__ functions behind one engine rather than a fork.
 * ---------------------------------------------------------------------------------- */

/* The leaf-collapse step, shared verbatim by the by-value and device-len echild wrappers: it
   reads the echild already written into `slot` and does not depend on how the length arrived. */
static int run_leaf_precompute(IqtreeGpuLik *c, int slot) {
    double *ech  = echild_of(c, slot);
    double *leaf = leaf_of(c, slot);
    const int ns = c->cfg.nstates;
    int m = c->cfg.num_tip_states * c->block;
    if (ns == NST_DNA)
        k_leaf_partials<NST_DNA><<<div_up(m, 256), 256, 0, g_stream>>>(
            leaf, ech, c->d_tip_lh, c->cfg.num_tip_states, c->block, c->cfg.state_unknown);
    else
        k_leaf_partials<NST_AA><<<div_up(m, 256), 256, 0, g_stream>>>(
            leaf, ech, c->d_tip_lh, c->cfg.num_tip_states, c->block, c->cfg.state_unknown);
    KERNEL_CHECK("branch_precompute/leaf");
    return 0;
}

int iqtree_gpu_lik_branch_precompute(IqtreeGpuLik *c, int slot, double len, int want_leaf) {
    if (!c) return lik_fail_msg("branch_precompute: null context");
    if (slot < 0 || slot >= c->cfg.echild_slots)
        return lik_fail_msg("branch_precompute: slot out of range");
    if (rates_ready(c, "branch_precompute")) return -1;

    double *ech = echild_of(c, slot);
    const int ns = c->cfg.nstates;
    int n = c->cfg.ncat * ns * ns;
    if (ns == NST_DNA)
        k_echild<NST_DNA><<<div_up(n, 256), 256, 0, g_stream>>>(ech, c->d_evec, c->d_eval, c->d_rates,
                                                   len, c->cfg.ncat);
    else
        k_echild<NST_AA><<<div_up(n, 256), 256, 0, g_stream>>>(ech, c->d_evec, c->d_eval, c->d_rates,
                                                  len, c->cfg.ncat);
    KERNEL_CHECK("branch_precompute/echild");

    if (want_leaf) return run_leaf_precompute(c, slot);
    return 0;
}

/* Device-len branch precompute (STEP 2). Same as branch_precompute, but the length comes from
   d_branch_len[branch_id] (see k_echild_dev). Used ONLY while recording a full-traversal graph,
   so that the recorded launch freezes the topological branch_id but not the (per-evaluation)
   length. iqtree_gpu_lik_set_branch_lengths() must have populated d_branch_len before any replay
   of the captured graph; at capture time the kernel records but does not execute, so the buffer
   need only be valid by the first replay. */
int iqtree_gpu_lik_branch_precompute_dev(IqtreeGpuLik *c, int slot, int branch_id, int want_leaf) {
    if (!c) return lik_fail_msg("branch_precompute_dev: null context");
    if (slot < 0 || slot >= c->cfg.echild_slots)
        return lik_fail_msg("branch_precompute_dev: slot out of range");
    if (!c->d_branch_len || branch_id < 0 || branch_id >= c->branch_len_cap)
        return lik_fail_msg("branch_precompute_dev: branch length buffer not set or id out of range");
    if (rates_ready(c, "branch_precompute_dev")) return -1;

    double *ech = echild_of(c, slot);
    const int ns = c->cfg.nstates;
    int n = c->cfg.ncat * ns * ns;
    if (ns == NST_DNA)
        k_echild_dev<NST_DNA><<<div_up(n, 256), 256, 0, g_stream>>>(
            ech, c->d_evec, c->d_eval, c->d_rates, c->d_branch_len, branch_id, c->cfg.ncat);
    else
        k_echild_dev<NST_AA><<<div_up(n, 256), 256, 0, g_stream>>>(
            ech, c->d_evec, c->d_eval, c->d_rates, c->d_branch_len, branch_id, c->cfg.ncat);
    KERNEL_CHECK("branch_precompute_dev/echild");

    if (want_leaf) return run_leaf_precompute(c, slot);
    return 0;
}

/* Upload the tree's current branch lengths, indexed by IQ-TREE branch id, into the device
   buffer that k_echild_dev reads. Lazily (re)allocates when nbranch grows. Blocking H2D on the
   default stream, so it MUST be called OUTSIDE a capture region -- the integration calls it just
   before a graph replay, where a blocking copy on stream 0 is correctly ordered before the
   replay. Tiny: nbranch doubles. */
int iqtree_gpu_lik_set_branch_lengths(IqtreeGpuLik *c, const double *lens, int nbranch) {
    if (!c || !lens) return lik_fail_msg("set_branch_lengths: null argument");
    if (nbranch <= 0) return lik_fail_msg("set_branch_lengths: nbranch must be positive");
    if (g_stream != 0)
        return lik_fail_msg("set_branch_lengths: refused during graph capture");
    if (nbranch > c->branch_len_cap) {
        if (c->d_branch_len) { cudaFree(c->d_branch_len); c->d_branch_len = NULL; }
        if (alloc_zero((void**)&c->d_branch_len, (size_t)nbranch * sizeof(double),
                       "set_branch_lengths/alloc")) return -1;
        c->branch_len_cap = nbranch;
    }
    CUDA_TRY(cudaMemcpy(c->d_branch_len, lens, (size_t)nbranch * sizeof(double),
                        cudaMemcpyHostToDevice), "set_branch_lengths/upload");
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Partial-likelihood dispatch
 * ---------------------------------------------------------------------------------- */

/* Runs the appropriate rescale kernel over an already-written buffer. Shared by the
   bifurcating cases (acting on the eigen-basis output) and the multifurcating case
   (acting on the x-basis accumulator). */
static int run_rescale(IqtreeGpuLik *c, double *buf, unsigned char *scale) {
    if (c->cfg.safe_numeric) {
        int n = c->stride * c->cfg.ncat;
        if (c->cfg.nstates == NST_DNA)
            k_rescale_safe<NST_DNA><<<div_up(n, 128), 128, 0, g_stream>>>(buf, scale, c->d_ptn_invar,
                                                             c->stride, c->cfg.ncat);
        else
            k_rescale_safe<NST_AA><<<div_up(n, 128), 128, 0, g_stream>>>(buf, scale, c->d_ptn_invar,
                                                            c->stride, c->cfg.ncat);
        KERNEL_CHECK("rescale/safe");
    } else {
        k_rescale_norm<<<div_up(c->stride, 128), 128, 0, g_stream>>>(buf, scale, c->d_ptn_invar,
                                                        c->stride, c->block);
        KERNEL_CHECK("rescale/norm");
    }
    return 0;
}

int iqtree_gpu_lik_partial_tip_tip(IqtreeGpuLik *c, double *d_out, unsigned char *d_scale_out,
                                   int slot_left, int tip_left, int slot_right, int tip_right) {
    if (!c || !d_out || !d_scale_out) return lik_fail_msg("partial_tip_tip: null argument");
    if (slot_left < 0 || slot_left >= c->cfg.echild_slots ||
        slot_right < 0 || slot_right >= c->cfg.echild_slots)
        return lik_fail_msg("partial_tip_tip: echild slot out of range");
    if (tip_left < 0 || tip_left >= c->cfg.tip_slots ||
        tip_right < 0 || tip_right >= c->cfg.tip_slots)
        return lik_fail_msg("partial_tip_tip: tip slot out of range");

    PartialArgs a;
    memset(&a, 0, sizeof(a));
    a.out      = d_out;
    a.leafL    = leaf_of(c, slot_left);
    a.leafR    = leaf_of(c, slot_right);
    a.stateL   = c->d_tip_states + (size_t)tip_left  * c->stride;
    a.stateR   = c->d_tip_states + (size_t)tip_right * c->stride;
    a.inv_evec = c->d_inv_evec;
    a.P        = c->stride;
    a.block    = c->block;

    if (launch_partial<true, true>(c, a, "partial_tip_tip")) return -1;

    /* IQ-TREE's cherry case performs NO rescaling and zeroes the counters. */
    CUDA_TRY(cudaMemsetAsync(d_scale_out, 0, c->scale_elems, g_stream), "partial_tip_tip/scale");
    return 0;
}

int iqtree_gpu_lik_partial_tip_internal(IqtreeGpuLik *c, double *d_out,
                                        unsigned char *d_scale_out,
                                        int slot_left, int tip_left,
                                        int slot_right, const double *d_right,
                                        const unsigned char *d_scale_right) {
    if (!c || !d_out || !d_scale_out || !d_right || !d_scale_right)
        return lik_fail_msg("partial_tip_internal: null argument");
    if (slot_left < 0 || slot_left >= c->cfg.echild_slots ||
        slot_right < 0 || slot_right >= c->cfg.echild_slots)
        return lik_fail_msg("partial_tip_internal: echild slot out of range");
    if (tip_left < 0 || tip_left >= c->cfg.tip_slots)
        return lik_fail_msg("partial_tip_internal: tip slot out of range");

    PartialArgs a;
    memset(&a, 0, sizeof(a));
    a.out      = d_out;
    a.leafL    = leaf_of(c, slot_left);
    a.stateL   = c->d_tip_states + (size_t)tip_left * c->stride;
    a.eR       = echild_of(c, slot_right);
    a.inR      = d_right;
    a.inv_evec = c->d_inv_evec;
    a.P        = c->stride;
    a.block    = c->block;

    if (launch_partial<true, false>(c, a, "partial_tip_internal")) return -1;

    /* Only the internal subtree carries scale counts into this node. */
    CUDA_TRY(cudaMemcpyAsync(d_scale_out, d_scale_right, c->scale_elems,
                             cudaMemcpyDeviceToDevice, g_stream),
             "partial_tip_internal/scale copy");
    return run_rescale(c, d_out, d_scale_out);
}

int iqtree_gpu_lik_partial_internal_internal(IqtreeGpuLik *c, double *d_out,
                                             unsigned char *d_scale_out,
                                             int slot_left, const double *d_left,
                                             const unsigned char *d_scale_left,
                                             int slot_right, const double *d_right,
                                             const unsigned char *d_scale_right) {
    if (!c || !d_out || !d_scale_out || !d_left || !d_right || !d_scale_left || !d_scale_right)
        return lik_fail_msg("partial_internal_internal: null argument");
    if (slot_left < 0 || slot_left >= c->cfg.echild_slots ||
        slot_right < 0 || slot_right >= c->cfg.echild_slots)
        return lik_fail_msg("partial_internal_internal: echild slot out of range");

    PartialArgs a;
    memset(&a, 0, sizeof(a));
    a.out      = d_out;
    a.eL       = echild_of(c, slot_left);
    a.eR       = echild_of(c, slot_right);
    a.inL      = d_left;
    a.inR      = d_right;
    a.inv_evec = c->d_inv_evec;
    a.P        = c->stride;
    a.block    = c->block;

    if (launch_partial<false, false>(c, a, "partial_internal_internal")) return -1;

    size_t n = c->scale_elems;
    k_scale_init<<<div_up_sz(n, 256), 256, 0, g_stream>>>(d_scale_out, d_scale_left, d_scale_right, n);
    KERNEL_CHECK("partial_internal_internal/scale init");
    return run_rescale(c, d_out, d_scale_out);
}

int iqtree_gpu_lik_partial_multifurcating(IqtreeGpuLik *c, double *d_out,
                                          unsigned char *d_scale_out,
                                          int nchild, const IqtreeGpuChild *children) {
    if (!c || !d_out || !d_scale_out || !children || nchild <= 0)
        return lik_fail_msg("partial_multifurcating: null or empty argument");

    if (!c->d_u) {
        /* Capacity-sized, like every other ncat-dependent buffer, so that a later set_ncat
           cannot outgrow it. */
        if (alloc_zero((void**)&c->d_u, c->partial_elems_cap * sizeof(double), "multifurcating/u"))
            return -1;
    }

    /* The running product lives in the x basis and starts at 1 for every element,
       exactly as IQ-TREE's partial_lh_all does. */
    k_fill<<<div_up_sz(c->partial_elems, 256), 256, 0, g_stream>>>(c->d_u, 1.0, c->partial_elems);
    KERNEL_CHECK("multifurcating/init");
    CUDA_TRY(cudaMemsetAsync(d_scale_out, 0, c->scale_elems, g_stream), "multifurcating/scale zero");

    for (int k = 0; k < nchild; k++) {
        const IqtreeGpuChild *ch = &children[k];
        if (ch->echild_slot < 0 || ch->echild_slot >= c->cfg.echild_slots)
            return lik_fail_msg("partial_multifurcating: echild slot out of range");
        const double *ech = echild_of(c, ch->echild_slot);

        if (ch->is_tip) {
            if (ch->tip_slot < 0 || ch->tip_slot >= c->cfg.tip_slots)
                return lik_fail_msg("partial_multifurcating: tip slot out of range");
            const double *leaf = leaf_of(c, ch->echild_slot);
            const int *st = c->d_tip_states + (size_t)ch->tip_slot * c->stride;
            if (launch_mf_child<true>(c, c->d_u, ech, leaf, st, NULL,
                                      "multifurcating/tip child")) return -1;
        } else {
            if (!ch->d_partial || !ch->d_scale)
                return lik_fail_msg("partial_multifurcating: internal child missing buffers");
            /* Scale counts are folded in BEFORE this child's underflow test, matching
               the order of IQ-TREE's neighbour loop. */
            size_t n = c->scale_elems;
            k_scale_add<<<div_up_sz(n, 256), 256, 0, g_stream>>>(d_scale_out, ch->d_scale, n);
            KERNEL_CHECK("multifurcating/scale add");
            if (launch_mf_child<false>(c, c->d_u, ech, NULL, NULL, ch->d_partial,
                                       "multifurcating/internal child")) return -1;
        }
        /* Underflow test after EVERY child, on the running x-basis product. */
        int rc = run_rescale(c, c->d_u, d_scale_out);
        if (rc) return rc;
    }

    /* The final rotation keeps the generic shape at both state counts (see k_mf_transform). */
    if (c->cfg.nstates == NST_DNA) {
        dim3 grid(c->stride / PTILE, c->cfg.ncat), blk(PTILE, NST_DNA);
        const size_t smem = (size_t)(NST_DNA * NST_DNA + NST_DNA * PTILE) * sizeof(double);
        k_mf_transform<NST_DNA><<<grid, blk, smem, g_stream>>>(d_out, c->d_u, c->d_inv_evec, c->stride);
    } else {
        dim3 grid(c->stride / PTILE, c->cfg.ncat), blk(PTILE, NST_AA);
        const size_t smem = (size_t)(NST_AA * NST_AA + NST_AA * PTILE) * sizeof(double);
        k_mf_transform<NST_AA><<<grid, blk, smem, g_stream>>>(d_out, c->d_u, c->d_inv_evec, c->stride);
    }
    KERNEL_CHECK("multifurcating/transform");
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Theta
 * ---------------------------------------------------------------------------------- */

int iqtree_gpu_lik_theta_internal(IqtreeGpuLik *c,
                                  const double *d_node, const unsigned char *d_scale_node,
                                  const double *d_dad,  const unsigned char *d_scale_dad) {
    if (!c || !d_node || !d_dad || !d_scale_node || !d_scale_dad)
        return lik_fail_msg("theta_internal: null argument");

    size_t n = c->partial_elems;
    k_theta_internal<<<div_up_sz(n, 256), 256, 0, g_stream>>>(theta_of(c), d_node, d_dad, n);
    KERNEL_CHECK("theta_internal");

    if (c->cfg.safe_numeric) {
        if (c->cfg.nstates == NST_DNA)
            k_theta_scale_safe<NST_DNA><<<div_up(c->stride, 128), 128, 0, g_stream>>>(
                theta_of(c), bufscale_of(c), d_scale_dad, d_scale_node,
                LOG_SCALING_THRESHOLD, c->stride, c->cfg.ncat);
        else
            k_theta_scale_safe<NST_AA><<<div_up(c->stride, 128), 128, 0, g_stream>>>(
                theta_of(c), bufscale_of(c), d_scale_dad, d_scale_node,
                LOG_SCALING_THRESHOLD, c->stride, c->cfg.ncat);
        KERNEL_CHECK("theta_internal/scale safe");
    } else {
        k_theta_scale_norm<<<div_up(c->stride, 128), 128, 0, g_stream>>>(
            bufscale_of(c), d_scale_dad, d_scale_node, LOG_SCALING_THRESHOLD, c->stride);
        KERNEL_CHECK("theta_internal/scale norm");
    }
    return 0;
}

int iqtree_gpu_lik_theta_tip(IqtreeGpuLik *c, int tip_slot,
                             const double *d_dad, const unsigned char *d_scale_dad) {
    if (!c || !d_dad || !d_scale_dad) return lik_fail_msg("theta_tip: null argument");
    if (tip_slot < 0 || tip_slot >= c->cfg.tip_slots)
        return lik_fail_msg("theta_tip: tip slot out of range");

    const int *st = c->d_tip_states + (size_t)tip_slot * c->stride;
    dim3 grid(div_up(c->stride, 128), c->block);
    if (c->cfg.nstates == NST_DNA)
        k_theta_tip<NST_DNA><<<grid, 128, 0, g_stream>>>(theta_of(c), c->d_tip_lh, st, d_dad, c->stride);
    else
        k_theta_tip<NST_AA><<<grid, 128, 0, g_stream>>>(theta_of(c), c->d_tip_lh, st, d_dad, c->stride);
    KERNEL_CHECK("theta_tip");

    /* A leaf carries no scale counters, so only the dad side contributes. */
    if (c->cfg.safe_numeric) {
        if (c->cfg.nstates == NST_DNA)
            k_theta_scale_safe<NST_DNA><<<div_up(c->stride, 128), 128, 0, g_stream>>>(
                theta_of(c), bufscale_of(c), d_scale_dad, NULL,
                LOG_SCALING_THRESHOLD, c->stride, c->cfg.ncat);
        else
            k_theta_scale_safe<NST_AA><<<div_up(c->stride, 128), 128, 0, g_stream>>>(
                theta_of(c), bufscale_of(c), d_scale_dad, NULL,
                LOG_SCALING_THRESHOLD, c->stride, c->cfg.ncat);
        KERNEL_CHECK("theta_tip/scale safe");
    } else {
        k_theta_scale_norm<<<div_up(c->stride, 128), 128, 0, g_stream>>>(
            bufscale_of(c), d_scale_dad, NULL, LOG_SCALING_THRESHOLD, c->stride);
        KERNEL_CHECK("theta_tip/scale norm");
    }
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Branch log-likelihood and analytic derivatives
 * ---------------------------------------------------------------------------------- */

int iqtree_gpu_lik_branch_logl_full(IqtreeGpuLik *c, double len, double *out_logl,
                                    double *h_ptn_lh, double *h_ptn_scaling,
                                    double *h_ptn_lh_cat, int save_log_value) {
    if (!c || !out_logl) return lik_fail_msg("branch_logl: null argument");
    if (rates_ready(c, "branch_logl")) return -1;

    /* d_sitelh is allocated on first use and, from then on, written by EVERY branch
       evaluation whether or not the caller wants a host copy this time. That is what makes
       iqtree_gpu_lik_download_ptn_lh() below sound: the device buffer then always holds the
       per-pattern values of the MOST RECENT branch evaluation, which is exactly the invariant
       IQ-TREE's host _pattern_lh has (every CPU computeLikelihoodBranch /
       computeLikelihoodFromBuffer overwrites it in full). Writing it costs one coalesced
       store of npat doubles inside a kernel that already moves two orders of magnitude more;
       the PCIe copy it lets the caller skip costs a synchronising 1.2 MB transfer per call.
       d_lhcat stays lazy -- it is ncat times larger and only computePatternLhCat asks for it. */
    if (!c->d_sitelh) {
        if (alloc_zero((void**)&c->d_sitelh, (size_t)c->stride * sizeof(double),
                       "branch_logl/sitelh")) return -1;
    }
    if (h_ptn_lh_cat && !c->d_lhcat) {
        if (alloc_zero((void**)&c->d_lhcat,
                       (size_t)c->ncat_cap * c->stride * sizeof(double),
                       "branch_logl/lhcat")) return -1;
    }

    if (c->cfg.nstates == NST_DNA)
        k_val0<NST_DNA><<<div_up(c->block, 128), 128, 0, g_stream>>>(c->d_val, c->d_eval, c->d_rates,
                                                        c->d_props, len, c->block);
    else
        k_val0<NST_AA><<<div_up(c->block, 128), 128, 0, g_stream>>>(c->d_val, c->d_eval, c->d_rates,
                                                       c->d_props, len, c->block);
    KERNEL_CHECK("branch_logl/val0");

    size_t smem = (size_t)c->block * sizeof(double);
    if (c->cfg.nstates == NST_DNA)
        k_branch_logl<NST_DNA><<<RED_BLOCKS, RED_THREADS, smem, g_stream>>>(
            theta_of(c), c->d_val, c->d_ptn_invar, c->d_ptn_freq, bufscale_of(c),
            c->stride, c->cfg.npat, c->cfg.ncat, c->d_sitelh,
            h_ptn_lh_cat ? c->d_lhcat : NULL, save_log_value, c->d_redblocks);
    else
        k_branch_logl<NST_AA><<<RED_BLOCKS, RED_THREADS, smem, g_stream>>>(
            theta_of(c), c->d_val, c->d_ptn_invar, c->d_ptn_freq, bufscale_of(c),
            c->stride, c->cfg.npat, c->cfg.ncat, c->d_sitelh,
            h_ptn_lh_cat ? c->d_lhcat : NULL, save_log_value, c->d_redblocks);
    KERNEL_CHECK("branch_logl/stage1");
    k_reduce_stage2<1><<<1, RED_THREADS, 0, g_stream>>>(c->d_redblocks, RED_BLOCKS, c->d_redout);
    KERNEL_CHECK("branch_logl/stage2");

    CUDA_TRY(cudaMemcpy(out_logl, c->d_redout, sizeof(double), cudaMemcpyDeviceToHost),
             "branch_logl/copy");
    if (h_ptn_lh)
        CUDA_TRY(cudaMemcpy(h_ptn_lh, c->d_sitelh, (size_t)c->cfg.npat * sizeof(double),
                            cudaMemcpyDeviceToHost), "branch_logl/copy sitelh");
    if (h_ptn_scaling)
        CUDA_TRY(cudaMemcpy(h_ptn_scaling, bufscale_of(c), (size_t)c->cfg.npat * sizeof(double),
                            cudaMemcpyDeviceToHost), "branch_logl/copy scaling");
    if (h_ptn_lh_cat) {
        /* Device layout is [c*stride + p]; the host wants [c*npat + p], so the padded tail
           of each category row is dropped one row at a time. */
        for (int cat = 0; cat < c->cfg.ncat; cat++)
            CUDA_TRY(cudaMemcpy(h_ptn_lh_cat + (size_t)cat * c->cfg.npat,
                                c->d_lhcat + (size_t)cat * c->stride,
                                (size_t)c->cfg.npat * sizeof(double), cudaMemcpyDeviceToHost),
                     "branch_logl/copy lhcat");
    }
    return 0;
}

int iqtree_gpu_lik_branch_logl(IqtreeGpuLik *c, double len, double *out_logl) {
    return iqtree_gpu_lik_branch_logl_full(c, len, out_logl, NULL, NULL, NULL, 1);
}

/* The deferred half of the copy branch_logl_full skips when h_ptn_lh is NULL. See the note
   next to the d_sitelh allocation above for why the buffer is always current. */
int iqtree_gpu_lik_download_ptn_lh(IqtreeGpuLik *c, double *h_ptn_lh) {
    if (!c || !h_ptn_lh) return lik_fail_msg("download_ptn_lh: null argument");
    if (!c->d_sitelh)
        return lik_fail_msg("download_ptn_lh: no branch log-likelihood has been computed yet");
    CUDA_TRY(cudaMemcpy(h_ptn_lh, c->d_sitelh, (size_t)c->cfg.npat * sizeof(double),
                        cudaMemcpyDeviceToHost), "download_ptn_lh");
    return 0;
}

int iqtree_gpu_lik_derivatives_g(IqtreeGpuLik *c, double len, int g_branch_id,
                                 double *out_logl, double *out_df, double *out_ddf) {
    if (!c || !out_df || !out_ddf) return lik_fail_msg("derivatives: null argument");
    if (rates_ready(c, "derivatives")) return -1;

    double *grow = NULL;
    if (g_branch_id >= 0) {
        if (!c->d_gmat)
            return lik_fail_msg("derivatives: a G row was requested but no G matrix is "
                                "enabled; call iqtree_gpu_lik_gmatrix_enable first");
        if (g_branch_id >= c->g_nbranch)
            return lik_fail_msg("derivatives: G row index out of range");
        grow = c->d_gmat + (size_t)g_branch_id * (size_t)c->g_stride;
    }

    if (c->cfg.nstates == NST_DNA)
        k_val012<NST_DNA><<<div_up(c->block, 128), 128, 0, g_stream>>>(c->d_val, c->d_eval, c->d_rates,
                                                          c->d_props, len, c->block);
    else
        k_val012<NST_AA><<<div_up(c->block, 128), 128, 0, g_stream>>>(c->d_val, c->d_eval, c->d_rates,
                                                         c->d_props, len, c->block);
    KERNEL_CHECK("derivatives/val012");

    size_t smem = 3 * (size_t)c->block * sizeof(double);
    k_derivatives<<<RED_BLOCKS, RED_THREADS, smem, g_stream>>>(theta_of(c), c->d_val, c->d_ptn_invar,
                                                     c->d_ptn_freq, bufscale_of(c),
                                                     c->stride, c->cfg.npat, c->block,
                                                     grow, c->d_redblocks);
    KERNEL_CHECK("derivatives/stage1");
    k_reduce_stage2<3><<<1, RED_THREADS, 0, g_stream>>>(c->d_redblocks, RED_BLOCKS, c->d_redout);
    KERNEL_CHECK("derivatives/stage2");

    double h[3];
    CUDA_TRY(cudaMemcpy(h, c->d_redout, 3 * sizeof(double), cudaMemcpyDeviceToHost),
             "derivatives/copy");
    if (out_logl) *out_logl = h[0];
    *out_df  = h[1];
    *out_ddf = h[2];
    return 0;
}

int iqtree_gpu_lik_derivatives(IqtreeGpuLik *c, double len,
                               double *out_logl, double *out_df, double *out_ddf) {
    return iqtree_gpu_lik_derivatives_g(c, len, -1, out_logl, out_df, out_ddf);
}

/* ------------------------------------------------------------------------------------
 * Newton-loop replay path (STEP 2). The three kernels of a derivative (or a
 * logL-only) evaluation form a FIXED sequence over engine-global buffers -- theta_of(c),
 * d_val, d_ptn_*, d_redout -- with the ONLY per-iteration input being the candidate branch
 * length, now read from the device scalar d_len. So the sequence is captured once and replayed
 * for every Newton iteration of every branch (the theta CONTENT differs per branch but flows
 * through the theta buffer, exactly like the model matrices). No G row is produced on this path
 * (Newton never wants one); callers that want G fall back to iqtree_gpu_lik_derivatives_g.
 * ---------------------------------------------------------------------------------- */

/* Upload the candidate branch length the replayed graphs read. Tiny blocking H2D on the default
   stream; refused during capture (the value is set between replays, never inside one). */
int iqtree_gpu_lik_set_eval_len(IqtreeGpuLik *c, double len) {
    if (!c) return lik_fail_msg("set_eval_len: null context");
    if (g_stream != 0) return lik_fail_msg("set_eval_len: refused during graph capture");
    CUDA_TRY(cudaMemcpy(c->d_len, &len, sizeof(double), cudaMemcpyHostToDevice), "set_eval_len");
    return 0;
}

/* Issue k_val012_dev + k_derivatives + k_reduce_stage2<3> into d_redout, reading len from d_len.
   No device-to-host copy: this is what gets captured, and what a replay re-runs. */
int iqtree_gpu_lik_derivatives_dev_issue(IqtreeGpuLik *c) {
    if (!c) return lik_fail_msg("derivatives_dev_issue: null context");
    if (rates_ready(c, "derivatives_dev_issue")) return -1;
    if (c->cfg.nstates == NST_DNA)
        k_val012_dev<NST_DNA><<<div_up(c->block, 128), 128, 0, g_stream>>>(
            c->d_val, c->d_eval, c->d_rates, c->d_props, c->d_len, c->block);
    else
        k_val012_dev<NST_AA><<<div_up(c->block, 128), 128, 0, g_stream>>>(
            c->d_val, c->d_eval, c->d_rates, c->d_props, c->d_len, c->block);
    KERNEL_CHECK("derivatives_dev_issue/val012");

    size_t smem = 3 * (size_t)c->block * sizeof(double);
    k_derivatives<<<RED_BLOCKS, RED_THREADS, smem, g_stream>>>(theta_of(c), c->d_val, c->d_ptn_invar,
                                                     c->d_ptn_freq, bufscale_of(c),
                                                     c->stride, c->cfg.npat, c->block,
                                                     NULL, c->d_redblocks);
    KERNEL_CHECK("derivatives_dev_issue/stage1");
    k_reduce_stage2<3><<<1, RED_THREADS, 0, g_stream>>>(c->d_redblocks, RED_BLOCKS, c->d_redout);
    KERNEL_CHECK("derivatives_dev_issue/stage2");
    return 0;
}

int iqtree_gpu_lik_derivatives_readback(IqtreeGpuLik *c, double *out_logl,
                                        double *out_df, double *out_ddf) {
    if (!c || !out_df || !out_ddf) return lik_fail_msg("derivatives_readback: null argument");
    double h[3];
    CUDA_TRY(cudaMemcpy(h, c->d_redout, 3 * sizeof(double), cudaMemcpyDeviceToHost),
             "derivatives_readback");
    if (out_logl) *out_logl = h[0];
    *out_df  = h[1];
    *out_ddf = h[2];
    return 0;
}

/* Issue k_val0_dev + k_branch_logl + k_reduce_stage2<1> into d_redout, reading len from d_len.
   Writes the per-pattern d_sitelh on-device as usual (the deferred host copy reads it later), but
   copies nothing to the host. d_sitelh must already exist: allocate it here only when NOT
   capturing, since a cudaMalloc inside a capture is illegal. */
int iqtree_gpu_lik_branch_logl_dev_issue(IqtreeGpuLik *c, int save_log_value) {
    if (!c) return lik_fail_msg("branch_logl_dev_issue: null context");
    if (rates_ready(c, "branch_logl_dev_issue")) return -1;
    if (!c->d_sitelh) {
        if (g_stream != 0)
            return lik_fail_msg("branch_logl_dev_issue: sitelh buffer must be allocated before capture");
        if (alloc_zero((void**)&c->d_sitelh, (size_t)c->stride * sizeof(double),
                       "branch_logl_dev_issue/sitelh")) return -1;
    }
    if (c->cfg.nstates == NST_DNA)
        k_val0_dev<NST_DNA><<<div_up(c->block, 128), 128, 0, g_stream>>>(
            c->d_val, c->d_eval, c->d_rates, c->d_props, c->d_len, c->block);
    else
        k_val0_dev<NST_AA><<<div_up(c->block, 128), 128, 0, g_stream>>>(
            c->d_val, c->d_eval, c->d_rates, c->d_props, c->d_len, c->block);
    KERNEL_CHECK("branch_logl_dev_issue/val0");

    size_t smem = (size_t)c->block * sizeof(double);
    if (c->cfg.nstates == NST_DNA)
        k_branch_logl<NST_DNA><<<RED_BLOCKS, RED_THREADS, smem, g_stream>>>(
            theta_of(c), c->d_val, c->d_ptn_invar, c->d_ptn_freq, bufscale_of(c),
            c->stride, c->cfg.npat, c->cfg.ncat, c->d_sitelh, NULL, save_log_value, c->d_redblocks);
    else
        k_branch_logl<NST_AA><<<RED_BLOCKS, RED_THREADS, smem, g_stream>>>(
            theta_of(c), c->d_val, c->d_ptn_invar, c->d_ptn_freq, bufscale_of(c),
            c->stride, c->cfg.npat, c->cfg.ncat, c->d_sitelh, NULL, save_log_value, c->d_redblocks);
    KERNEL_CHECK("branch_logl_dev_issue/stage1");
    k_reduce_stage2<1><<<1, RED_THREADS, 0, g_stream>>>(c->d_redblocks, RED_BLOCKS, c->d_redout);
    KERNEL_CHECK("branch_logl_dev_issue/stage2");
    return 0;
}

int iqtree_gpu_lik_branch_logl_readback(IqtreeGpuLik *c, double *out_logl) {
    if (!c || !out_logl) return lik_fail_msg("branch_logl_readback: null argument");
    CUDA_TRY(cudaMemcpy(out_logl, c->d_redout, sizeof(double), cudaMemcpyDeviceToHost),
             "branch_logl_readback");
    return 0;
}

/* ------------------------------------------------------------------------------------
 * On-device Newton loop (STEP 3)
 * ----------------------------------------------------------------------------------
 * The whole per-branch Newton-Raphson optimisation runs on the GPU inside ONE conditional-WHILE
 * graph, so the host reads back only the final optimised length -- one D2H per branch instead of
 * one per iteration. The WHILE body is [k_val012_dev -> k_derivatives -> k_reduce_stage2<3> ->
 * k_newton_step]; k_newton_step advances the ported minimizeNewton state and sets the loop
 * condition. Because the derivative kernels are the same ones the host path uses and k_newton_step
 * is a faithful scalar port, the converged length matches the host optimiser. Requires theta to be
 * already resident in the NEWTON slot (the host computes it once before the loop, exactly as
 * minimizeNewton's iteration 0 would). REQUIRES CUDA >= 12.4 conditional graph nodes. */
int iqtree_gpu_lik_newton_build(IqtreeGpuLik *c) {
    if (!c) return lik_fail_msg("newton_build: null context");
    if (rates_ready(c, "newton_build")) return -1;
    if (g_stream != 0) return lik_fail_msg("newton_build: refused during capture");
    if (c->newton_exec) { cudaGraphExecDestroy(c->newton_exec); c->newton_exec = NULL; }

    cudaGraph_t graph = NULL;
    CUDA_TRY(cudaGraphCreate(&graph, 0), "newton_build/create");

    cudaGraphConditionalHandle handle;
    cudaError_t e = cudaGraphConditionalHandleCreate(&handle, graph, /*default=*/1,
                                                     cudaGraphCondAssignDefault);
    if (e != cudaSuccess) { cudaGraphDestroy(graph); return lik_fail("newton_build/handle", e); }

    cudaGraphNodeParams np = {};   /* aggregate-init: the union deletes the default constructor */
    np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = handle;
    np.conditional.type   = cudaGraphCondTypeWhile;
    np.conditional.size   = 1;
    cudaGraphNode_t cnode;
    e = cudaGraphAddNode(&cnode, graph, NULL, 0, &np);
    if (e != cudaSuccess) { cudaGraphDestroy(graph); return lik_fail("newton_build/addnode", e); }
    cudaGraph_t body = np.conditional.phGraph_out[0];

    /* Populate the WHILE body by capturing the four-kernel sequence into it. The launches read
       g_stream, so point it at a temporary capture stream for the duration. */
    cudaStream_t s = NULL;
    e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (e != cudaSuccess) { cudaGraphDestroy(graph); return lik_fail("newton_build/stream", e); }
    e = cudaStreamBeginCaptureToGraph(s, body, NULL, NULL, 0, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) { cudaStreamDestroy(s); cudaGraphDestroy(graph); return lik_fail("newton_build/begin", e); }

    g_stream = s;
    if (c->cfg.nstates == NST_DNA)
        k_val012_dev<NST_DNA><<<div_up(c->block, 128), 128, 0, g_stream>>>(
            c->d_val, c->d_eval, c->d_rates, c->d_props, c->d_len, c->block);
    else
        k_val012_dev<NST_AA><<<div_up(c->block, 128), 128, 0, g_stream>>>(
            c->d_val, c->d_eval, c->d_rates, c->d_props, c->d_len, c->block);
    {
        size_t smem = 3 * (size_t)c->block * sizeof(double);
        k_derivatives<<<RED_BLOCKS, RED_THREADS, smem, g_stream>>>(
            theta_of(c), c->d_val, c->d_ptn_invar, c->d_ptn_freq, bufscale_of(c),
            c->stride, c->cfg.npat, c->block, NULL, c->d_redblocks);
    }
    k_reduce_stage2<3><<<1, RED_THREADS, 0, g_stream>>>(c->d_redblocks, RED_BLOCKS, c->d_redout);
    k_newton_step<<<1, 1, 0, g_stream>>>(c->d_newton, c->d_redout, c->d_len, handle);
    g_stream = 0;

    cudaGraph_t captured = NULL;
    e = cudaStreamEndCapture(s, &captured);
    cudaStreamDestroy(s);
    if (e != cudaSuccess) { cudaGraphDestroy(graph); return lik_fail("newton_build/end", e); }

    e = cudaGraphInstantiate(&c->newton_exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) { c->newton_exec = NULL; return lik_fail("newton_build/instantiate", e); }
    return 0;
}

int iqtree_gpu_lik_newton_optimize(IqtreeGpuLik *c, double x1, double x2, double xacc,
                                   int maxNRStep, double xguess, double *out_optx, int *out_error) {
    if (!c || !out_optx) return lik_fail_msg("newton_optimize: null argument");
    if (!c->newton_exec) return lik_fail_msg("newton_optimize: graph not built");
    if (g_stream != 0) return lik_fail_msg("newton_optimize: refused during capture");

    double rts = xguess;                                  /* clamp, minimizeNewton:429-430 */
    if (rts < x1) rts = x1;
    if (rts > x2) rts = x2;

    NewtonState h;
    memset(&h, 0, sizeof(h));
    h.x1 = x1; h.x2 = x2; h.xacc = xacc; h.maxNRStep = maxNRStep;
    h.xl = x1; h.xh = x2; h.rts = rts; h.rts_old = rts; h.result = rts;
    h.j = 0; h.initialized = 0; h.converged = 0; h.error = 0;
    CUDA_TRY(cudaMemcpy(c->d_newton, &h, sizeof(h), cudaMemcpyHostToDevice), "newton_optimize/state");
    CUDA_TRY(cudaMemcpy(c->d_len, &rts, sizeof(double), cudaMemcpyHostToDevice), "newton_optimize/len");

    CUDA_TRY(cudaGraphLaunch(c->newton_exec, 0), "newton_optimize/launch");
    g_newton_loops.fetch_add(1, std::memory_order_relaxed);

    CUDA_TRY(cudaMemcpy(&h, c->d_newton, sizeof(h), cudaMemcpyDeviceToHost), "newton_optimize/readback");
    *out_optx = h.result;
    if (out_error) *out_error = h.error;
    return 0;
}

void iqtree_gpu_lik_newton_free(IqtreeGpuLik *c) {
    if (c && c->newton_exec) { cudaGraphExecDestroy(c->newton_exec); c->newton_exec = NULL; }
}

long long iqtree_gpu_lik_newton_count(void) {
    return g_newton_loops.load(std::memory_order_relaxed);
}

/* ------------------------------------------------------------------------------------
 * G matrix and the MCMCTree Hessian
 * ---------------------------------------------------------------------------------- */

int iqtree_gpu_lik_gmatrix_enable(IqtreeGpuLik *c, int nbranch, int row_stride) {
    if (!c) return lik_fail_msg("gmatrix_enable: null context");
    if (nbranch <= 0) return lik_fail_msg("gmatrix_enable: nbranch must be positive");
    if (row_stride < c->cfg.npat)
        return lik_fail_msg("gmatrix_enable: row stride is narrower than the pattern count");
    /* Same geometry: keep the buffer, but re-zero it. The rows are keyed by branch id and a
       caller re-enabling has, by definition, no claim on what the previous owner left. */
    if (c->d_gmat && c->g_nbranch == nbranch && c->g_stride == row_stride) {
        CUDA_TRY(cudaMemset(c->d_gmat, 0,
                            (size_t)nbranch * (size_t)row_stride * sizeof(double)),
                 "gmatrix_enable/rezero");
        return 0;
    }
    cudaFree(c->d_gmat); c->d_gmat = NULL;
    cudaFree(c->d_hess); c->d_hess = NULL;
    c->g_nbranch = 0; c->g_stride = 0;

    /* Zeroed, not merely allocated: only p < npat is ever written, and the [npat, row_stride)
       padding has to be exactly 0.0 so that the rank-k update below -- and the host consumer
       that maps the same row as a length-mem_size vector -- see zeros rather than whatever
       was in device memory. ptn_freq is 0 there too, so this is equivalent to the CPU result
       and additionally removes a 0*NaN hazard. */
    if (alloc_zero((void**)&c->d_gmat,
                   (size_t)nbranch * (size_t)row_stride * sizeof(double), "gmatrix_enable/G"))
        return -1;
    if (alloc_zero((void**)&c->d_hess,
                   (size_t)nbranch * (size_t)nbranch * sizeof(double), "gmatrix_enable/H")) {
        cudaFree(c->d_gmat); c->d_gmat = NULL;
        return -1;
    }
    c->g_nbranch = nbranch;
    c->g_stride  = row_stride;
    return 0;
}

int iqtree_gpu_lik_gmatrix_download_row(IqtreeGpuLik *c, int branch_id, double *h_row) {
    if (!c || !h_row) return lik_fail_msg("gmatrix_download_row: null argument");
    if (!c->d_gmat) return lik_fail_msg("gmatrix_download_row: no G matrix is enabled");
    if (branch_id < 0 || branch_id >= c->g_nbranch)
        return lik_fail_msg("gmatrix_download_row: branch index out of range");
    CUDA_TRY(cudaMemcpy(h_row, c->d_gmat + (size_t)branch_id * (size_t)c->g_stride,
                        (size_t)c->g_stride * sizeof(double), cudaMemcpyDeviceToHost),
             "gmatrix_download_row/copy");
    return 0;
}

int iqtree_gpu_lik_gmatrix_hessian(IqtreeGpuLik *c, double *h_hessian) {
    if (!c || !h_hessian) return lik_fail_msg("gmatrix_hessian: null argument");
    if (!c->d_gmat || !c->d_hess) return lik_fail_msg("gmatrix_hessian: no G matrix is enabled");

    const int n = c->g_nbranch;
    const int k = c->cfg.npat;

    if (!c->cublas_ready) {
        if (cublasCreate(&c->cublas) != CUBLAS_STATUS_SUCCESS)
            return lik_fail_msg("gmatrix_hessian: cublasCreate failed");
        c->cublas_ready = 1;
    }

    /* Scratch for the column-scaled copy. Allocated here and freed before returning: it is as
       large as G itself (branchNum * npat doubles) and this runs once per export, so holding
       it for the life of the engine would take memory away from the pooled partial slots for
       no benefit. */
    double *d_gs = NULL;
    if (alloc_zero((void**)&d_gs, (size_t)n * (size_t)k * sizeof(double), "gmatrix_hessian/Gs"))
        return -1;

    k_gmat_scale<<<div_up(k, 256), 256, 0, g_stream>>>(d_gs, c->d_gmat, c->d_ptn_freq, n, k, c->g_stride);
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { cudaFree(d_gs); return lik_fail("gmatrix_hessian/scale", e); }
    }

    /* H = Gs * Gs^T.  d_gs is row-major n x k, which is exactly the column-major k x n matrix
       A with lda = k, so the product we want is A^T * A -- hence CUBLAS_OP_T. */
    const double alpha = 1.0, beta = 0.0;
    cublasStatus_t st = cublasDsyrk(c->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                    n, k, &alpha, d_gs, k, &beta, c->d_hess, n);
    cudaFree(d_gs);
    if (st != CUBLAS_STATUS_SUCCESS)
        return lik_fail_msg("gmatrix_hessian: cublasDsyrk failed");

    {
        dim3 blk(16, 16), grd(div_up(n, 16), div_up(n, 16));
        k_symmetrize<<<grd, blk, 0, g_stream>>>(c->d_hess, n);
        KERNEL_CHECK("gmatrix_hessian/symmetrize");
    }
    CUDA_TRY(cudaMemcpy(h_hessian, c->d_hess, (size_t)n * (size_t)n * sizeof(double),
                        cudaMemcpyDeviceToHost), "gmatrix_hessian/copy");
    return 0;
}

} /* extern "C" */

