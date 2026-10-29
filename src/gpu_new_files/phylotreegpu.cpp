// phylotreegpu.cpp -- IQ-TREE3-GPU likelihood backend, integration layer (g++ side).
//
// This translation unit is compiled by the normal C++ compiler WITH the IQ-TREE headers and
// calls the plain-C CUDA engine declared in gpu_kernels.h. It implements the four PhyloTree
// member functions that setLikelihoodKernel() swaps in when `-gpu` is active:
//
//     computePartialLikelihoodGPU   <- computePartialLikelihoodGenericSIMD
//     computeLikelihoodBranchGPU    <- computeLikelihoodBranchGenericSIMD
//     computeLikelihoodDervGPU      <- computeLikelihoodDervGenericSIMD
//     computeLikelihoodFromBufferGPU<- computeLikelihoodFromBufferGenericSIMD
//
// The four are a MUTUALLY CONSISTENT SET: between them they own the device partials, the
// device theta and the per-pattern outputs, and they never leave a host buffer half-written
// for a CPU kernel to pick up. See "CPU FALLBACK COHERENCE" below for why that is safe.
//
// ---------------------------------------------------------------------------------------
// SLOT POOLING (the single most important design point)
// ---------------------------------------------------------------------------------------
// IQ-TREE does not keep one partial-likelihood vector per branch. central_partial_lh is
// max_lh_slots * getPartialLhSize() doubles, and PhyloNeighbor::partial_lh points INTO that
// arena; MemSlotVector (LM_MEM_SAVE) and reorientPartialLh (LM_PER_NODE) recycle those slots
// along the traversal. Keying device buffers by PhyloNeighbor* -- which the previous version
// did -- therefore allocates 2*(nodeNum-1) buffers where IQ-TREE itself uses max_lh_slots:
// for the largest benchmark genus (65 taxa, ~156k patterns, 10 categories, 250 MB per slot)
// that is ~63 GB against a 40-48 GB card, i.e. a guaranteed OOM.
//
// Here every device buffer is keyed by the SLOT INDEX derived from the host pointer:
//
//     slot = (dad_branch->partial_lh - central_partial_lh) / getPartialLhSize()
//
// This is a 1:1 map onto IQ-TREE's own arena, so it inherits IQ-TREE's recycling for free and
// is bounded by max_lh_slots (+2 for the nni_partial_lh scratch pair). The correctness
// argument is short: the ONLY writer of device slot k is computePartialLikelihoodGPU for a
// neighbour currently owning host slot k, and whenever IQ-TREE hands slot k to a different
// neighbour it clears the previous owner's partial_lh_computed bit (MemSlotVector::allocate ->
// clearPartialLh, MemSlotVector::takeover). A cleared bit means the neighbour is pushed into
// traversal_info and recomputed, which rewrites device slot k. So "host slot k is valid for
// neighbour X" and "device slot k is valid for neighbour X" are always the same statement.
//
// ---------------------------------------------------------------------------------------
// ONE HOST THREAD DRIVES THE GPU
// ---------------------------------------------------------------------------------------
// IQ-TREE calls computePartialLikelihood once per (node, packet) with disjoint pattern
// ranges. The rewritten kernels compute ALL patterns of a node in a single launch, because
// splitting one large launch across host threads and streams was a main reason the previous
// kernel was slow (fragmented launches on top of an uncoalesced layout). So packet 0 does the
// whole node and every other packet returns immediately without touching the GPU. Everything
// runs on the default stream, in issue order, from one thread -- no mutex on the hot path,
// and no stream-ordering hazards to reason about.
//
// ---------------------------------------------------------------------------------------
// CPU FALLBACK COHERENCE (why a per-call fallback cannot corrupt anything)
// ---------------------------------------------------------------------------------------
// gpu_eligible() is a pure function of (alignment, model, model_factory, rate, tree flags).
// None of those can change while a likelihood evaluation is in flight, so within a single
// evaluation the GPU and CPU kernels never alternate: either every call is GPU or every call
// is CPU. The host buffers a CPU kernel would read (central_partial_lh, central_scale_num,
// theta_all, buffer_scale_all) are therefore only ever read in runs where the CPU also wrote
// them.
//
// Across evaluations, eligibility CAN flip -- ModelFinder swaps a mixture model in, and
// setLikelihoodKernel() runs again. To make the guarantee unconditional rather than
// probabilistic, both transitions force a full invalidation of the host partials:
//   * entering GPU mode (first context build for a tree) calls clearAllPartialLH(), so the
//     GPU never trusts partial_lh_computed bits set by the CPU for slots it never wrote;
//   * leaving GPU mode calls clearAllPartialLH(), so the CPU never trusts bits set by the
//     GPU for host memory that was never written.
// Both happen only in computeLikelihoodBranchGPU / computeLikelihoodDervGPU, i.e. at the top
// of an evaluation before any traversal has started -- never mid-traversal.
//
// ---------------------------------------------------------------------------------------
// OUTPUT CONTRACT (the other half of eligibility)
// ---------------------------------------------------------------------------------------
// gpu_eligible() below validates the INPUTS of a run: data type, model class, tree flags.
// That is only half of correctness. IQ-TREE's CPU kernels also fill a set of HOST BUFFERS as
// a side effect, and a feature that reads one of them is wrong on the GPU unless this file
// fills it too -- silently wrong, because the buffer exists and holds plausible numbers from
// whenever it was last written. `--dating mcmctree` is exactly that failure: it read
// G_matrix, nothing checked, and the exported Hessian was garbage from an uninitialised
// allocation. The table below is the fix for the CLASS of bug: every host buffer a CPU
// likelihood kernel writes, and what the GPU path does about it.
//
//   buffer             status under -gpu
//   ------------------ --------------------------------------------------------------------
//   _pattern_lh        FILLED, but LAZILY. Both branch entry points compute it on the device
//                      (k_branch_logl's out_site) on every call; the ~1.2 MB copy back is
//                      deferred to the first reader, of which there are exactly two outside
//                      the likelihood kernels -- PhyloTree::computeLikelihood()'s memmove and
//                      PhyloTree::computePatternLikelihood() -- and both call
//                      gpuSyncPatternLh() first. SH-aLRT and UFBoot reach it through the
//                      latter. See "DEFERRED _pattern_lh COPY" at gpu_flush_ptn_lh().
//   _pattern_scaling   FILLED on exactly the same condition as the CPU kernel, i.e. when
//                      save_log_value is false. Neither writes it otherwise.
//   _pattern_lh_cat    FILLED when gpu_ptn_lh_cat_wanted is set, which computePatternLhCat()
//                      does around its computeLikelihoodBranch call. Every in-tree consumer
//                      (+R EM, +I EM, empirical-Bayes site rates, -wslr) reaches it through
//                      computePatternLhCat. POISONED at context build so a future direct
//                      reader that forgets the flag produces NaN, not a stale answer.
//   _pattern_lh_cat_state
//                      NOT filled. Its only consumer is marginal ancestral reconstruction,
//                      which sets params->kernel_nonrev = true first
//                      (phylotreesse.cpp:1514-1521); that makes ModelSubst::useRevKernel()
//                      false, so gpu_eligible() rejects the tree and every call goes to the
//                      CPU. Excluded by an input test, but for an OUTPUT reason -- noted
//                      here so the next person does not "simplify" that check away.
//   _site_lh           NOT filled (it is per SITE, not per pattern; the engine has no site
//                      dimension at all). Consumers are -robust-phy / -robust-alrt, which
//                      iqtree_gpu_unsupported_output() refuses at kernel-selection time.
//   theta_all          NOT maintained; the device keeps theta instead. The only consumers are
//                      computeLikelihoodDerv / computeLikelihoodFromBuffer (both replaced
//                      here) and computeLikelihoodDervMixlen (mixlen trees are ineligible).
//                      POISONED at context build.
//   buffer_scale_all   NOT maintained; the device keeps the per-pattern scale sum. Consumers
//                      are the +ASC branches of the CPU kernels, and +ASC is ineligible.
//                      POISONED at context build.
//   G_matrix           FILLED when the run is `--dating mcmctree`, one row per branch, from
//                      the same register the derivative reduction uses. Otherwise POISONED,
//                      because nothing else reads it and a new reader must not get numbers.
//   gradient_vector    FILLED on every derivative call, as on the CPU.
//   hessian_diagonal   FILLED on every derivative call, as on the CPU.
//
// Two mechanisms enforce the table instead of merely describing it:
//   (1) POISONING. Everything marked POISONED is overwritten with signalling NaN in
//       gpuPoisonUnmaintainedBuffers(), called from gpuBuildContext(). An untraced consumer
//       then produces obviously broken output instead of plausible-but-wrong numbers. This is
//       the difference between a bug that is found in an afternoon and one that ships.
//   (2) THE PARAMS GUARD. iqtree_gpu_unsupported_output() maps run-constant Params flags to
//       the outputs they will demand and is consulted by setLikelihoodKernel() BEFORE the GPU
//       function pointers are installed. A run asking for something the table says is not
//       filled never reaches the GPU kernel in the first place, and is told why.

// ---------------------------------------------------------------------------------------
// PROVING THE GPU RAN (per-call accounting, and --gpu-strict)
// ---------------------------------------------------------------------------------------
// The "GPU acceleration ACTIVE" banner is printed ONCE, at kernel selection, and says nothing
// about what happened afterwards. All four entry points below re-test eligibility on EVERY
// call and silently delegate to the saved CPU pointers when it fails, so a run can print the
// banner, show non-zero nvidia-smi utilisation, and still evaluate an arbitrary fraction of
// its likelihoods on the CPU -- one ineligible ModelFinder candidate is enough. A benchmark in
// that state is partly CPU-vs-CPU and passes a banner-plus-utilisation gate unchanged. A silent
// CPU-vs-CPU result is easy to produce and hard to notice; the two mechanisms here exist to
// make that class of error impossible to miss rather than merely unlikely.
//
//   ACCOUNTING (always on). Every call is counted as executed-on-device or delegated-to-CPU,
//   per entry point, and every delegation records WHICH of gpu_eligible()'s ~12 conditions
//   failed. "3,412 calls delegated: model-is-mixture" is actionable; a bare count is not.
//   Reported once per process from the same static destructor as the context counters, with a
//   machine-readable GPUSTATS line a harness can assert on.
//
//   --gpu-strict. COMPUTATIONALLY IDENTICAL to plain -gpu -- same eligibility test, same
//   kernels, same launches, same tolerances, same results. What it changes is what happens on
//   a fallback: instead of delegating, the run aborts naming the failed condition and the call
//   site; a device error is fatal instead of routing through gpuAbortToCpu(); and the exit
//   accounting must show 100% GPU. It exists to PROVE that -gpu ran the whole likelihood, not
//   to make it faster.
//
//   --gpu-newton host|device. WHERE the per-branch Newton-Raphson loop runs. `host` (the
//   default) is IQ-TREE's own minimizeNewton on the CPU, calling the device for every
//   derivative; this is the configuration of every run in the published benchmark. `device`
//   runs the whole loop in one conditional CUDA graph (STEP 3 below) with a single host
//   round-trip per branch; it is a faithful port of the same routine and is unit-tested
//   against it, but it was NOT the benchmarked configuration, so it is opt-in. The choice is
//   reported in the banner and as `newton_mode=` in the GPUSTATS line.
//
// ---------------------------------------------------------------------------------------
// DEFERRED _pattern_lh DOWNLOAD
// ---------------------------------------------------------------------------------------
// The branch kernels used to copy _pattern_lh (npat doubles, ~1.2 MB at 156k patterns) back to
// the host on every call, because the CPU kernel always writes it. Most of those calls are
// inside the Newton branch-length loop and nothing reads the result. The values are now left
// on the device and pulled across only when a consumer asks; see gpu_flush_ptn_lh() for the
// argument that this is the same buffer content the CPU would have left behind.

#include "phylotree.h"
#include "gpu_kernels.h"
#include "model/modelfactory.h"
// Only for the explicit +E exclusion in gpu_ineligible_reason(): ModelDNAError is a reversible
// ModelDNA subclass that no other predicate there separates out.
#include "model/modeldnaerror.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>   // _exit(), for the --gpu-strict exit assertion

// Called from PhyloTree::setLikelihoodKernel() when -gpu is requested.
int iqtree_gpu_report() {
    int n = iqtree_gpu_device_count();
    if (n > 0) {
        printf("[IQ-TREE3-GPU] %d CUDA device(s) visible; device 0: %s\n",
               n, iqtree_gpu_device_name());
    } else {
        printf("[IQ-TREE3-GPU] no CUDA device visible - GPU likelihood unavailable, using CPU.\n");
    }
    fflush(stdout);
    return n;
}

// ------------------------------------------------------------------------------------------
// THE PARAMS GUARD -- outputs, not inputs. See the OUTPUT CONTRACT at the top of this file.
// ------------------------------------------------------------------------------------------
// Consulted by setLikelihoodKernel() before the GPU function pointers are installed. Returns
// NULL when every output this run will ask for is one the GPU maintains, otherwise a sentence
// naming the responsible flag. It reads only RUN-CONSTANT Params fields, which is what makes
// a single decision at kernel-selection time sound: none of them can change once the run is
// under way, so there is no window in which the answer flips mid-evaluation.
//
// Adding to this list is the required step when a new IQ-TREE feature reads a host buffer the
// engine does not fill. Deciding to accept a feature instead means filling the buffer AND
// moving its line in the table above out of the POISONED column -- not deleting the entry.
const char* iqtree_gpu_unsupported_output(const Params *params) {
    if (!params) return NULL;

    // -robust-phy / -robust-alrt redefine the tree log-likelihood as an order statistic over
    // _site_lh, i.e. over SITES. Every buffer in the engine is indexed by PATTERN, and the
    // pattern -> site expansion does not exist on the device at all, so there is nothing to
    // fill _site_lh from. gpu_eligible() also rejects these runs, which would make the search
    // fall back per call; refusing here instead means the user is told once, clearly, rather
    // than getting a silent CPU run that they paid for a GPU to avoid.
    if (params->robust_phy_keep < 1.0 || params->robust_median)
        return "-robust-phy / -robust-alrt: the tree log-likelihood is an order statistic over "
               "per-SITE likelihoods (_site_lh), which the pattern-indexed GPU engine does not "
               "produce";

    // --dating mcmctree is SUPPORTED and deliberately listed as such rather than omitted: it is
    // the feature that motivated this guard, and an empty-looking list invites the next reader
    // to conclude the mechanism is vestigial. The GPU fills G_matrix, gradient_vector and
    // hessian_diagonal, and computes H = G*diag(ptn_freq)*G^T itself; see
    // computeLikelihoodDervGPU and PhyloTree::gpuMCMCHessian.
    // (params->dating_method == "mcmctree" -> supported, no return)

    return NULL;
}

namespace {

// Theta slot assignment. The Newton slot is the analogue of IQ-TREE's theta_all and is
// governed by theta_computed; the branch slot exists because computeLikelihoodBranch must not
// disturb it (IQ-TREE's own computeLikelihoodBranch never touches theta_all).
const int GPU_THETA_NEWTON = 0;
const int GPU_THETA_BRANCH = 1;
const int GPU_THETA_SLOTS  = 2;

struct GpuSlot {
    double        *lh    = nullptr;
    unsigned char *scale = nullptr;
    // Generation in which this slot was last WRITTEN by computePartialLikelihoodGPU. See the
    // staleness note next to GpuContext::gen.
    long long      gen   = 0;
};

// ------------------------------------------------------------------------------------------
// RATE-CATEGORY CAPACITY (why the context is not rebuilt when ncat changes)
// ------------------------------------------------------------------------------------------
// ModelFinder (-m TESTNEW) enumerates +G{num_rate_cats} and +R{k} for k = min_rate_cats..
// max_rate_cats against every candidate matrix (CandidateModelSet::generate,
// main/phylotesting.cpp:1673-1676), so getRate()->getNRate() changes hundreds of times in a
// single run. Treating that as a shape change meant a full gpu_destroy() + create(), i.e. a
// cudaFree + cudaMalloc + cudaMemset of the entire multi-GB arena: measured 0.45 -> 1.12 GB
// for a small genus and 6.5 -> 16.3 GB for a large one, and it shows up as nothing but "the
// GPU is slow".
//
// The engine now sizes every category-dependent buffer for a CAPACITY and runs at any
// ncat <= capacity, because buf[(cat*nstates + state)*stride + pattern] makes the smaller
// layout an exact prefix of the larger one. All that is left on an ncat change is to
// recompute the launch geometry and invalidate what the category count gives meaning to --
// see gpuPrepare().
//
// ------------------------------------------------------------------------------------------
// ARENA LIFETIME vs TREE BINDING (why a new IQTree does not mean a new arena)
// ------------------------------------------------------------------------------------------
// The capacity above removed ncat as a rebuild trigger. It measurably did not help, because
// ModelFinder's rebuilds were never driven by ncat: inside ModelFinder each candidate is
// evaluated on a FRESHLY CONSTRUCTED IQTree (CandidateModel::evaluate,
// main/phylotesting.cpp:1938-1954, `new IQTree(in_aln)` ... `delete iqtree`), so `g_gpu.tree ==
// this`, `g_gpu.central_lh == central_partial_lh` and `g_gpu.nni_lh == nni_partial_lh` all fail
// on every candidate. A benchmark run reported 15761 full rebuilds and 0 ncat switches, and
// ModelFinder came in at 1.75x where tree search -- which never rebuilds -- reached 2.62x.
//
// The two lifetimes are genuinely independent:
//
//   THE ARENA is a function of GEOMETRY only -- npat, ncat_cap, nstates, safe_numeric, the slot
//   count and the host block/scale sizes. Nothing it allocates depends on WHICH PhyloTree the
//   numbers came from. A new tree with the same geometry can use the existing allocations as
//   they stand.
//
//   THE BINDING is the slot mapping, slot = (nei->partial_lh - central_partial_lh)/host_block,
//   plus the handful of device buffers holding values computed FOR the old tree.
//
// So when gpuGeometryMatches() holds but gpuContextMatches() does not, gpuRebind() re-points the
// context at the new tree -- new tree pointer, new central_lh / nni_lh / host_block / host_scale,
// engine-side invalidation via iqtree_gpu_lik_rebind() -- and NOTHING is freed or allocated.
//
// The invalidation on the rebind path is exactly as thorough as on the rebuild path, and it has
// to be: see the block comment in gpuPrepare(). The conditions that still force a full rebuild
// are enumerated in gpuGeometryMatches(); the two that matter most are a different Alignment
// object (which would alias the resident tip-state columns onto the wrong data) and an active
// --dating mcmctree G matrix (whose rows are indexed by one tree's branch ids).
//
// The capacity comes from IQ-TREE's own bound rather than a guessed constant: generate()
// returns max(num_rate_cats, max_rate_cats) as `max_cats` and IQ-TREE sizes its OWN memory
// estimate from it (getMemoryRequiredThreaded(max_cats), phylotesting.cpp:1485). Defaults are
// 4 and 10; -cmax raises max_rate_cats and this follows it. Mixtures would multiply that by
// getClassNum(), but gpu_eligible() rejects mixtures outright.
int gpu_ncat_capacity_wanted(const Params *params, int ncat_now) {
    int cap = ncat_now;
    if (params) {
        if (params->num_rate_cats > cap) cap = params->num_rate_cats;
        if (params->max_rate_cats > cap) cap = params->max_rate_cats;
    }
    return cap;
}

// Fraction of total device memory the pooled slots may occupy at full capacity. The slots are
// allocated lazily, so this is a ceiling on the worst case rather than on what is resident;
// the headroom left over covers the engine's own arrays plus the CUDA context.
const double GPU_ARENA_VRAM_FRACTION = 0.70;

// ------------------------------------------------------------------------------------------
// PER-CALL ACCOUNTING
// ------------------------------------------------------------------------------------------
// The four entry points, so a delegation can be attributed to the call site rather than to the
// backend as a whole. Which one delegates is diagnostic in itself: a run that is 100% GPU in
// computePartialLikelihood but delegates computeLikelihoodFromBuffer is a different bug from a
// run that delegates everything.
enum GpuEntryPoint {
    GPU_EP_PARTIAL = 0,
    GPU_EP_BRANCH,
    GPU_EP_DERV,
    GPU_EP_FROMBUFFER,
    GPU_EP_COUNT
};
const char *GPU_EP_NAME[GPU_EP_COUNT] = { "partial", "branch", "derv", "frombuffer" };

// Why a call was delegated. gpu_eligible() is a conjunction of about a dozen conditions, and
// knowing WHICH one failed is the difference between "the GPU did not run" and "the GPU did not
// run because ModelFinder was testing a +C60 profile mixture". The order of the enum is the
// order the conditions are tested in gpu_ineligible_reason(); a call is attributed to the FIRST
// condition that failed, which is exactly what the short-circuiting && chain used to do.
enum GpuDelegateReason {
    GPU_OK = 0,
    GPU_R_ENGINE_FAILED,      // sticky: a device error has already torn the context down
    GPU_R_NO_MODEL,           // model / rate / factory / alignment / params not attached yet
    GPU_R_UNSUPPORTED_DATA,   // not (20 states & SEQ_PROTEIN) and not (4 states & SEQ_DNA)
    GPU_R_MIXTURE,            // mixture or profile-mixture model
    GPU_R_NON_REVERSIBLE,     // !useRevKernel(): partials are not in the eigen basis
    GPU_R_SITE_SPECIFIC,      // site-specific (per-site frequency) model
    GPU_R_ASC,                // +ASC / unobserved patterns
    GPU_R_MIXLEN,             // mixlen tree: drives the un-swapped mixlen derivative
    GPU_R_SUPERTREE,          // partitioned / super tree
    GPU_R_TREEMIX,            // tree mixture
    GPU_R_ROOTED,             // rooted tree: the tip-state upload assumes an unrooted leaf loop
    GPU_R_NO_RATE_CATS,       // getNRate() < 1
    GPU_R_ROBUST_PHY,         // -robust-phy / -robust-alrt: tree_lh is an order statistic
    GPU_R_CONTEXT_NOT_READY,  // eligible, but no device context belongs to this tree yet
    GPU_R_PREPARE_FAILED,     // gpuPrepare() could not build or refresh the context
    GPU_R_DEVICE_ERROR,       // a CUDA error mid-evaluation; the result cannot be trusted
    GPU_R_COUNT
};

const char *gpu_reason_name(int r) {
    switch (r) {
        case GPU_R_ENGINE_FAILED:     return "engine-failed";
        case GPU_R_NO_MODEL:          return "model-not-attached";
        case GPU_R_UNSUPPORTED_DATA:  return "unsupported-data-type";
        case GPU_R_MIXTURE:           return "model-is-mixture";
        case GPU_R_NON_REVERSIBLE:    return "model-non-reversible";
        case GPU_R_SITE_SPECIFIC:     return "model-site-specific";
        case GPU_R_ASC:               return "ascertainment-bias-correction";
        case GPU_R_MIXLEN:            return "mixlen-tree";
        case GPU_R_SUPERTREE:         return "super-tree";
        case GPU_R_TREEMIX:           return "tree-mixture";
        case GPU_R_ROOTED:            return "rooted-tree";
        case GPU_R_NO_RATE_CATS:      return "no-rate-categories";
        case GPU_R_ROBUST_PHY:        return "robust-phylogeny";
        case GPU_R_CONTEXT_NOT_READY: return "device-context-not-ready";
        case GPU_R_PREPARE_FAILED:    return "context-build-failed";
        case GPU_R_DEVICE_ERROR:      return "device-error";
        default:                      return "unknown";
    }
}

// Instrumentation. The point of the capacity AND of the rebind path is to eliminate rebuild
// thrash, and the only honest way to know either worked is to count all three transitions and
// say so. Reported from a static destructor so the line appears once per process no matter how
// many trees, models and ModelFinder candidates went past. The per-call accounting rides in the
// same object for the same reason, and so that the two lines cannot be separated in a log.
//
// Reading the context line: full rebuilds should be a handful (one per genuinely new geometry),
// rebinds should absorb what used to be rebuilds (one per ModelFinder candidate), and ncat
// switches should appear only where a persistent tree changes its rate model in place.
//
// WHY ATOMIC. The GPU path proper is single-threaded -- one host thread, packet 0, one launch
// per node -- so a plain counter would do for it. The DELEGATED path is not: when eligibility
// fails, computeLikelihoodBranch runs the CPU SIMD kernel, whose traversal calls
// computePartialLikelihood from an `omp parallel for` over packets (phylokernelnew.h:1271-1277),
// i.e. computePartialLikelihoodGPU's delegation branch runs on num_threads threads at once.
// Relaxed atomics are a lock xadd with no fences and no ordering constraints; against a node
// launch measured in hundreds of microseconds, or against a CPU packet's own work, that is not
// measurable. Getting the count wrong, on the other hand, would defeat the entire point.
typedef std::atomic<long long> GpuCounter;

struct GpuContextStats {
    long long full_rebuilds  = 0;
    long long rebinds        = 0;
    long long ncat_switches  = 0;
    int       ncat_cap       = 0;

    // ---- STEP 2: launch-storm split accounting ----
    // These are touched only on the single-threaded GPU driver path (never from the delegated
    // multi-threaded CPU fallback), so a plain counter is exact here. full_captured/full_replayed
    // count full traversals turned into / served from a graph; subset_traversals counts the
    // dirty-subset sweeps that stay on the per-node path. launches_per_full is the per-full-
    // traversal launch count measured at the first capture, so the reader can estimate launches
    // eliminated == full_replayed * launches_per_full and compare it to kernel_launches.
    long long full_captured  = 0;
    long long full_replayed  = 0;
    long long subset_traversals = 0;
    long long launches_per_full = 0;
    // Newton-loop replay: derivative (df/ddf) and logL-only re-eval iterations served by a
    // replayed singleton graph instead of 3 fresh launches. These should dominate on DNA.
    long long derv_replayed  = 0;
    long long logl_replayed  = 0;
    // On-device Newton: whole per-branch loops run in one conditional-graph launch (STEP 3).
    long long newton_ondevice = 0;

    GpuCounter on_gpu[GPU_EP_COUNT];
    GpuCounter on_cpu[GPU_EP_COUNT];
    GpuCounter by_reason[GPU_R_COUNT];
    // Packets != 0 that returned without doing anything, because packet 0 already covered the
    // whole node in one launch. Not work, so not counted as either GPU or CPU; reported so that
    // "calls in minus calls accounted for" always balances.
    GpuCounter non_owning_packets;
    GpuCounter device_errors;
    // Latched from params->gpu_strict. The destructor has no PhyloTree to consult and the flag
    // is run-constant, so latching is exact rather than a guess.
    bool       strict = false;
    // Latched from params->gpu_newton_device, for the same reason and in the same place.
    bool       newton_device = false;

    GpuContextStats() {
        for (int i = 0; i < GPU_EP_COUNT; i++) { on_gpu[i].store(0); on_cpu[i].store(0); }
        for (int i = 0; i < GPU_R_COUNT; i++) by_reason[i].store(0);
        non_owning_packets.store(0);
        device_errors.store(0);
    }

    ~GpuContextStats() {
        long long gpu = 0, cpu = 0;
        for (int i = 0; i < GPU_EP_COUNT; i++) {
            gpu += on_gpu[i].load();
            cpu += on_cpu[i].load();
        }
        if (gpu == 0 && cpu == 0 && full_rebuilds == 0 && rebinds == 0 && ncat_switches == 0)
            return;

        const double pct = (gpu + cpu) ? 100.0 * (double)gpu / (double)(gpu + cpu) : 0.0;
        printf("[IQ-TREE3-GPU] likelihood calls: %lld on GPU, %lld delegated to CPU "
               "(%.1f%% GPU)\n", gpu, cpu, pct);
        printf("[IQ-TREE3-GPU]   per entry point (gpu/cpu):");
        for (int i = 0; i < GPU_EP_COUNT; i++)
            printf(" %s=%lld/%lld", GPU_EP_NAME[i], on_gpu[i].load(), on_cpu[i].load());
        printf("\n");
        if (cpu > 0) {
            printf("[IQ-TREE3-GPU]   delegated by reason:");
            for (int i = 1; i < GPU_R_COUNT; i++)
                if (by_reason[i].load())
                    printf(" %s=%lld", gpu_reason_name(i), by_reason[i].load());
            printf("\n");
        }
        if (full_rebuilds || rebinds || ncat_switches)
            printf("[IQ-TREE3-GPU] context: %lld full rebuilds, %lld rebinds, %lld ncat "
                   "switches (arena sized for ncat<=%d)\n",
                   full_rebuilds, rebinds, ncat_switches, ncat_cap);

        // MACHINE-READABLE. One line, stable key=value pairs, so a benchmark harness can assert
        // pct_gpu==100.00 instead of eyeballing a log. Keep the keys and their spelling stable:
        // scripts grep for them.
        printf("[IQ-TREE3-GPU] GPUSTATS gpu_calls=%lld cpu_calls=%lld pct_gpu=%.2f "
               "rebuilds=%lld rebinds=%lld ncat_switches=%lld packets_skipped=%lld "
               "device_errors=%lld kernel_launches=%lld graph_replays=%lld "
               "full_captured=%lld full_replayed=%lld subset_traversals=%lld "
               "launches_per_full=%lld derv_replayed=%lld logl_replayed=%lld "
               "newton_ondevice=%lld newton_loops=%lld strict=%d newton_mode=%s\n",
               gpu, cpu, pct, full_rebuilds, rebinds, ncat_switches,
               non_owning_packets.load(), device_errors.load(),
               iqtree_gpu_lik_launch_count(), iqtree_gpu_lik_graph_replay_count(),
               full_captured, full_replayed, subset_traversals, launches_per_full,
               derv_replayed, logl_replayed, newton_ondevice, iqtree_gpu_lik_newton_count(),
               strict ? 1 : 0, newton_device ? "device" : "host");
        fflush(stdout);

        // The --gpu-strict exit assertion. Every individual delegation is already fatal at the
        // point it happens, so reaching here with cpu > 0 means something got past those checks
        // -- exactly the failure mode this flag exists to catch, so it must not be a warning.
        // _exit() rather than exit(): we are already inside static destruction, the report above
        // has been flushed, and a non-zero exit status is the only signal a CI harness can act
        // on from here. gpu == 0 with cpu == 0 is NOT a failure: it means no likelihood was
        // evaluated at all (e.g. --help, or a run that stopped in alignment parsing).
        if (strict && cpu > 0) {
            fprintf(stderr, "[IQ-TREE3-GPU] --gpu-strict FAILED: %lld of %lld likelihood "
                            "calls ran on the CPU (%.2f%% GPU). The run's timings are not a "
                            "GPU measurement.\n", cpu, gpu + cpu, pct);
            cout.flush();
            cerr.flush();
            fflush(NULL);
            _exit(2);
        }
    }
};
GpuContextStats g_stats;

// ------------------------------------------------------------------------------------------
// --gpu-strict
// ------------------------------------------------------------------------------------------
// Latched on the first entry-point call from params->gpu_strict, which is parsed once from the
// command line and never rewritten. Latching is what lets gpu_error() and the static reporter
// -- neither of which has a PhyloTree in hand -- honour the flag. The write is idempotent
// (always the same value from the same run-constant source), so the delegation path's
// multi-threading does not make it a race with consequences.
bool g_gpu_strict = false;

inline bool gpu_note_strict(const PhyloTree *tree) {
    if (tree && tree->params && tree->params->gpu_strict) {
        g_gpu_strict = true;
        g_stats.strict = true;
    }
    // Same latch for the Newton placement, so the GPUSTATS line records which loop ran.
    if (tree && tree->params && tree->params->gpu_newton_device)
        g_stats.newton_device = true;
    return g_gpu_strict;
}

// Records a call. `reason` is GPU_OK for a call that ran on the device.
inline void gpu_count(int ep, int reason) {
    if (reason == GPU_OK) {
        g_stats.on_gpu[ep].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    g_stats.on_cpu[ep].fetch_add(1, std::memory_order_relaxed);
    g_stats.by_reason[reason].fetch_add(1, std::memory_order_relaxed);
}

// The strict-mode refusal. Named condition, named call site, and no fallback: a run that asked
// to be proved GPU-only does not get to quietly produce a partly-CPU answer instead.
void gpu_strict_refuse(int ep, int reason) {
    fflush(stdout);
    string msg = "IQ-TREE3-GPU --gpu-strict: ";
    msg += GPU_EP_NAME[ep];
    msg += " would have been delegated to the CPU likelihood kernel (condition: ";
    msg += gpu_reason_name(reason);
    msg += "). Plain -gpu delegates silently here, which is what makes a partly-CPU benchmark "
           "look like a GPU one; --gpu-strict refuses instead. ";
    if (reason == GPU_R_DEVICE_ERROR || reason == GPU_R_ENGINE_FAILED ||
        reason == GPU_R_PREPARE_FAILED || reason == GPU_R_CONTEXT_NOT_READY)
        msg += "This is a DEVICE-side failure, not an unsupported analysis: see the "
               "[IQ-TREE3-GPU] error above for what the CUDA runtime reported.";
    else
        msg += "Re-run with -gpu to allow the fallback, or restrict the analysis to what the "
               "engine supports: amino-acid or nucleotide data under a single reversible "
               "non-mixture matrix with +F/+G/+R/+I, on an unrooted, unpartitioned, "
               "single-length tree. Note that "
               "ModelFinder (-m TESTNEW / MFP) enumerates candidates the engine cannot run, so "
               "--gpu-strict normally requires an explicit -m.";
    outError(msg);
}

// Counts a delegation and, under --gpu-strict, refuses it. Returns so that the caller's
// fallback statement reads the same in both modes; under strict, outError() has already exited.
inline void gpu_delegate(int ep, int reason) {
    gpu_count(ep, reason);
    if (g_gpu_strict) gpu_strict_refuse(ep, reason);
}

// One tree drives the GPU at a time. A map keyed by PhyloTree* would multiply the VRAM
// footprint by the number of live trees, and the footprint is the binding constraint here.
struct GpuContext {
    IqtreeGpuLik *eng = nullptr;

    // ---- TREE BINDING. Changing any of these needs a gpuRebind(), nothing more. ----
    const PhyloTree *tree        = nullptr;
    const double    *central_lh  = nullptr;
    const double    *nni_lh      = nullptr;
    size_t           host_block  = 0;   // getPartialLhSize()
    size_t           host_scale  = 0;   // getScaleNumSize()

    // ---- ARENA GEOMETRY. Changing any of these needs a full destroy + create. ----
    // aln is here rather than among the bindings for one reason: the device tip-state columns
    // are uploaded once and survive a rebind, so a rebind onto a DIFFERENT alignment would read
    // one alignment's states while computing another's likelihood -- silently, since npat and
    // leafNum could easily match. Pointer identity is the check; see gpuGeometryMatches().
    const Alignment *aln         = nullptr;
    int              npat        = 0;
    // The state count is ARENA GEOMETRY, not a binding: every device buffer is
    // ncat*nstates*stride doubles and the engine is compiled per state count. A protein/DNA
    // switch always changes `aln` and `ntip_states` too, so the check added to
    // gpuGeometryMatches() is strictly redundant today -- it is there because relying on that
    // coincidence is exactly the kind of implicit invariant this file's comments warn about.
    int              nstates     = 0;
    int              ncat        = 0;   // CURRENT category count
    int              ncat_cap    = 0;   // categories the device arena was sized for
    int              safe_numeric = 0;
    int              nleaf       = 0;   // == cfg.tip_slots, rows of d_tip_states
    int              nodenum     = 0;
    int64_t          max_slots   = 0;   // PhyloTree::max_lh_slots
    int              ntip_states = 0;   // aln->STATE_UNKNOWN + 1, rows of d_tip_lh
    int              echild_slots = 0;  // per-branch precompute slots the arena holds

    int              host_slots  = 0;   // slots inside central_partial_lh
    int              nslots      = 0;   // host_slots + 2 (the nni_partial_lh pair)
    std::vector<GpuSlot> slots;

    // ---- SLOT STALENESS, and why a counter rather than a memset --------------------------
    // The pooled partial/scale buffers are the multi-GB part of the arena, and the rebind path
    // deliberately keeps them: zeroing them would cost most of what the rebuild it replaces
    // cost. They therefore still hold the PREVIOUS tree's (or the previous ncat's) partials
    // when a rebind or an ncat switch returns, and the argument that this is safe is the slot-
    // pooling argument at the top of this file -- clearAllPartialLH() clears every
    // partial_lh_computed bit, so every neighbour is pushed into traversal_info and its device
    // slot is rewritten before anything reads it.
    //
    // A rebuild used to make a violation of that argument LOUD for free, because the fresh
    // allocations were zeroed and a missed recomputation produced log(0) = -inf. Reusing the
    // buffers gives that up: a missed recomputation would read a plausible partial from another
    // tree. This counter buys the loudness back exactly, for one int compare per slot access:
    // it is bumped wherever the host partials are invalidated, stamped onto a slot when
    // computePartialLikelihoodGPU writes it, and checked on every READ. It tests something
    // strictly WEAKER than the pooling invariant already requires ("written at least once since
    // the last invalidation" vs "written for THIS neighbour"), so it cannot fire on a run the
    // existing design already handles correctly -- but it converts the one silent failure this
    // change could introduce into an immediate, named error and a CPU fallback.
    long long        gen         = 0;

    std::vector<double> h_lhcat;        // category-major staging for _pattern_lh_cat
    std::vector<int>    h_states;       // staging for one leaf's state column

    // --dating mcmctree only. gmat_stride is the HOST row pitch (mem_size), because the rows
    // are copied straight into PhyloTree::G_matrix, which printMCMCFileFormat() indexes as
    // G[branch_id*mem_size + ptn]. gmat_written tracks which rows have been produced SINCE the
    // buffer was created, so that the device-side Hessian is offered only when every row of
    // the current G is one this context wrote -- never a mixture of old and new.
    bool                gmat = false;
    int                 gmat_nbranch = 0;
    size_t              gmat_stride  = 0;
    std::vector<char>   gmat_written;
    int                 gmat_rows    = 0;

    bool                failed = false; // a device error was seen; never try the GPU again
    bool                error_seen = false; // a device error inside the current evaluation
    int                 nreported = 0;  // throttle for the message above

    // ---- DEFERRED _pattern_lh COPY -------------------------------------------------------
    // Set when a branch evaluation produced the per-pattern log-likelihoods on the device but
    // did not copy them to the host. ptn_lh_host is the destination the copy owes; it belongs
    // to the tree that was current at the time, which is not necessarily the tree asking now.
    bool                ptn_lh_pending = false;
    double             *ptn_lh_host    = nullptr;

    // ---- STEP 2: full-traversal CUDA-graph cache ----------------------------
    // `capturing` is set only while re-issuing a full traversal in stream-capture mode; it
    // tells computePartialLikelihoodGPU to precompute echild through the DEVICE-LEN path
    // (branch_precompute_dev) and to not double-count the pass in the GPU/CPU stats.
    //
    // full_graphs caches one executable graph per (traversal signature, gen, theta_slot, ncat).
    // The signature is the ordered list of dad_branch->id over traversal_info, which pins the
    // exact kernel sequence; gen/ncat pin the slot pointers and launch shapes; theta_slot pins
    // which theta buffer the captured theta step writes. A lookup that matches on all four is a
    // guarantee the recorded launches are still the right ones. Any gen bump (rebind), ncat
    // switch, or engine teardown frees the whole cache -- see gpuFreeFullGraphs().
    bool                capturing = false;
    struct FullGraph {
        std::vector<int> sig;
        long long        gen;
        int              theta_slot;
        int              ncat;
        IqtreeGpuGraph  *graph;
        long long        launches;   // launches this graph replaced, measured at capture
    };
    std::vector<FullGraph> full_graphs;

    // ---- Newton-loop replay singletons (STEP 2) ----
    // The per-iteration derivative and logL-only sequences read only engine-global buffers
    // (theta_of(NEWTON), d_val, d_ptn_*, d_len), so ONE captured graph each serves EVERY Newton
    // iteration of EVERY branch: the per-branch theta content flows through the theta buffer, the
    // candidate length through d_len. Validity is tagged by (gen, ncat, theta_slot); a mismatch
    // recaptures. Freed on gen bump / ncat change / teardown by gpuFreeGraphs().
    struct SmallGraph {
        IqtreeGpuGraph *graph = nullptr;
        long long       gen = -1;
        int             ncat = -1;
        int             theta_slot = -1;
    };
    SmallGraph derv_graph;   // k_val012_dev + k_derivatives + reduce<3>
    SmallGraph logl_graph;   // k_val0_dev   + k_branch_logl + reduce<1>

    // ---- On-device Newton loop (STEP 3) ----
    // The conditional-WHILE graph lives inside the engine (newton_exec); here we only track the
    // (gen, ncat) it was built for, to know when to rebuild. Built lazily on first use.
    long long newton_gen  = -1;
    int       newton_ncat = -1;
};

GpuContext g_gpu;

// Free and forget every cached full-traversal graph. Called whenever anything a cached graph
// bakes in could have moved: the slot generation (rebind), the category count (set_ncat), or the
// engine itself (teardown / full rebuild). Cheap and rare.
static void gpuFreeFullGraphs() {
    for (auto &e : g_gpu.full_graphs)
        if (e.graph) iqtree_gpu_lik_graph_free(e.graph);
    g_gpu.full_graphs.clear();
    if (g_gpu.derv_graph.graph) { iqtree_gpu_lik_graph_free(g_gpu.derv_graph.graph); g_gpu.derv_graph = GpuContext::SmallGraph(); }
    if (g_gpu.logl_graph.graph) { iqtree_gpu_lik_graph_free(g_gpu.logl_graph.graph); g_gpu.logl_graph = GpuContext::SmallGraph(); }
    if (g_gpu.eng) iqtree_gpu_lik_newton_free(g_gpu.eng);
    g_gpu.newton_gen = -1; g_gpu.newton_ncat = -1;
}

// Records a device failure. The flag matters more than the message: a kernel that did not
// run leaves its device slot holding the previous node's partial, which would silently
// produce a plausible-looking wrong likelihood. The enclosing entry point checks
// error_seen and abandons the whole GPU path rather than trusting a partial result.
//
// Under --gpu-strict there is no "rather than": a device error means the GPU did not compute
// what the run is about to report, and the whole promise of the flag is that such a run stops
// instead of finishing on the CPU with the timings of a GPU one.
void gpu_error(const char *what) {
    g_gpu.error_seen = true;
    g_stats.device_errors.fetch_add(1, std::memory_order_relaxed);
    if (g_gpu.nreported++ < 8) {
        fprintf(stderr, "[IQ-TREE3-GPU] %s: %s\n", what, iqtree_gpu_lik_last_error());
        fflush(stderr);
    }
    if (g_gpu_strict) {
        fflush(stdout);
        string msg = "IQ-TREE3-GPU --gpu-strict: device error in ";
        msg += what;
        msg += ": ";
        msg += iqtree_gpu_lik_last_error();
        msg += ". Plain -gpu would abandon the GPU path here and finish the run on the CPU "
               "kernel; --gpu-strict stops instead, because the resulting timings would not be "
               "a GPU measurement.";
        outError(msg);
    }
}

// ------------------------------------------------------------------------------------------
// DEFERRED _pattern_lh COPY -- correctness argument
// ------------------------------------------------------------------------------------------
// The device buffer behind this is written by EVERY branch evaluation and by nothing else (see
// the note next to the d_sitelh allocation in gpu_kernels.cu), which is precisely the invariant
// IQ-TREE's host _pattern_lh has: both hold the per-pattern values of the most recent
// computeLikelihoodBranch / computeLikelihoodFromBuffer. Copying at the moment of the read is
// therefore the same content as copying at the moment of the write. What has to be guaranteed
// is only that the copy happens BEFORE the device buffer's meaning changes, i.e. before:
//   * a reader reads _pattern_lh          -> PhyloTree::gpuSyncPatternLh(), called from the two
//                                            readers in phylotree.cpp;
//   * the in-kernel non-finite repair     -> flushed explicitly in both branch entry points;
//   * the context is destroyed or rebound -> gpu_destroy() and gpuPrepare() flush;
//   * the host buffer is freed            -> PhyloTree::gpuForgetPatternLh() DROPS it, since a
//                                            freed buffer has no reader and the destination
//                                            pointer would dangle.
// A CPU fallback needs nothing: the entry points that delegate call gpuLeave() first, which
// destroys the context and therefore flushes, and the CPU kernel then overwrites _pattern_lh in
// full anyway.
void gpu_flush_ptn_lh() {
    if (!g_gpu.ptn_lh_pending) return;
    g_gpu.ptn_lh_pending = false;          // cleared first: a failed copy must not be retried
    if (!g_gpu.eng || !g_gpu.ptn_lh_host) return;
    if (iqtree_gpu_lik_download_ptn_lh(g_gpu.eng, g_gpu.ptn_lh_host) != 0)
        gpu_error("download deferred _pattern_lh");
}

void gpu_destroy() {
    gpu_flush_ptn_lh();
    g_gpu.ptn_lh_host = nullptr;
    gpuFreeFullGraphs();   // graphs bake in this engine's device slot pointers
    if (g_gpu.eng) {
        for (GpuSlot &s : g_gpu.slots) {
            iqtree_gpu_lik_free(g_gpu.eng, s.lh);
            iqtree_gpu_lik_free(g_gpu.eng, s.scale);
        }
        iqtree_gpu_lik_destroy(g_gpu.eng);
    }
    g_gpu.eng = nullptr;
    g_gpu.tree = nullptr;
    g_gpu.aln = nullptr;  // no arena, so nothing is bound to an alignment either
    g_gpu.ncat_cap = 0;   // no arena, so no capacity to serve an ncat change from
    g_gpu.echild_slots = 0;
    g_gpu.host_slots = 0;
    g_gpu.nslots = 0;
    g_gpu.slots.clear();
    g_gpu.h_lhcat.clear();
    g_gpu.h_states.clear();
    g_gpu.gmat = false;
    g_gpu.gmat_nbranch = 0;
    g_gpu.gmat_stride  = 0;
    g_gpu.gmat_written.clear();
    g_gpu.gmat_rows = 0;
}

// A run whose exported Hessian comes from G_matrix. This is run-constant -- params->
// dating_method is parsed once from the command line and never rewritten -- which is exactly
// why the per-row device-to-host copy can be gated on it: it is the precise condition under
// which printHessian() fires, so no Newton inner loop outside that mode pays the ~1.2 MB
// transfer for a buffer only dating reads.
bool gpu_wants_gmatrix(const PhyloTree *tree) {
    return tree->params && tree->params->dating_method == "mcmctree";
}

// The row pitch of PhyloTree::G_matrix, copied verbatim from the two places that define it:
// the allocation (phylotree.cpp:910-921) and the consumer (phyloanalysis.cpp:2877-2879). It
// must be THEIR formula and not the CPU kernel's g_matrix_nptn (phylokernelnew.h:2277), which
// rounds with VectorClass::size() instead of get_safe_upper_limit(); the two agree in every
// real build, and gpuBuildContext() checks that they do rather than assuming it.
size_t gpu_gmatrix_stride(PhyloTree *tree) {
    return get_safe_upper_limit(tree->getAlnNPattern())
         + std::max(get_safe_upper_limit(tree->getModel()->num_states),
                    get_safe_upper_limit(tree->getModelFactory()->unobserved_ptns.size()));
}

// Echild precompute slots the tree needs: one per child of its widest node, at least two.
// Walked in a single traversal rather than via findNodeID(), which is O(n) per lookup and would
// make this O(n^2) on every ModelFinder candidate.
int gpu_max_children(PhyloTree *tree) {
    int max_children = 2;
    NodeVector inodes;
    tree->getInternalNodes(inodes);
    for (Node *n : inodes)
        if ((int)n->degree() - 1 > max_children) max_children = (int)n->degree() - 1;
    return max_children;
}

// The assumption gpuBuildContext()'s tip-state upload rests on: the leaf whose node id is `i`
// carries alignment column `i`, so that device tip slot i holds that leaf's states. It is
// PhyloTree::setAlignment() that establishes it (node->id is the sequence index looked up by
// name), and it holds for every tree built on a given alignment -- but the rebind path REUSES
// the uploaded columns instead of re-deriving them, so the assumption stops being an internal
// detail of one build and becomes a claim about a tree the arena was not built for. Cheap to
// verify (O(leafNum) string compares against an O(leafNum*npat) re-upload), and the failure it
// prevents -- every taxon's states attributed to the wrong tip -- is silent.
bool gpu_leaf_ids_match_alignment(PhyloTree *tree) {
    NodeVector taxa;
    tree->getTaxa(taxa);
    if (taxa.size() != (size_t)tree->leafNum) return false;
    for (Node *n : taxa) {
        if (n->id < 0 || n->id >= (int)tree->leafNum) return false;
        if (tree->aln->getSeqName(n->id) != n->name) return false;
    }
    return true;
}

} // namespace

// GPU-eligible models. The engine covers amino acids (20 states, SEQ_PROTEIN) and nucleotides
// (4 states, SEQ_DNA) with a SINGLE reversible non-mixture rate matrix plus +F / +G / +R / +I;
// everything else falls back per call to the saved CPU SIMD pointers. On the DNA side that is
// every reversible ModelDNA -- JC, F81, K80/K2P, HKY, TNe, TN, K81/K3P, K81u, TPM2(u), TPM3(u),
// TIMe, TIM, TIM2(e), TIM3(e), TVMe, TVM, SYM, GTR -- i.e. ModelFinder's whole default DNA
// candidate list (phylotesting.cpp:82-84), with +FQ / +F / +FO frequencies. The predicate tests
// PROPERTIES (single mixture, useRevKernel, not site-specific), never model names, which is why
// no list has to be maintained here. The exclusions below are not cosmetic:
//   - mixtures / site-specific models: the kernel has no mixture dimension at all;
//   - non-reversible models: partials are not in the eigen basis there;
//   - +ASC (unobserved_ptns): adds synthetic patterns past aln->size() with their own
//     constant-site correction, which none of the reductions here implement;
//   - mixlen trees: they drive computeLikelihoodDervMixlenPointer, which is NOT swapped and
//     reads the host theta_all this file never fills;
//   - super trees / tree mixtures: several PhyloTree buffers are re-pointed per partition;
//   - robust phylogeny (-robust-*): redefines tree_lh as an order statistic over _site_lh.
//
// Returns GPU_OK, or the FIRST condition that failed. Splitting the old single `&&` chain into
// named tests changes no behaviour -- every condition is a pure predicate over state that
// cannot change during an evaluation, so testing them one at a time yields the same answer --
// and it is what lets a delegation be reported as "3412 calls: model-is-mixture" instead of as
// an anonymous count. The order is the old chain's order, so the reason attributed to a call is
// the condition the short-circuiting && would have stopped at.
//
// NAMING: the comments throughout this file still speak of gpu_eligible(). This IS that test --
// `gpu_ineligible_reason(t) == GPU_OK` and the old `gpu_eligible(t)` are the same predicate over
// the same state. The boolean wrapper is deliberately not kept: an unused one-liner next to the
// real test is exactly the kind of thing that later gets called by mistake and loses the reason.
static inline int gpu_ineligible_reason(PhyloTree *tree) {
    // g_gpu.failed is sticky: once a device error has forced the host partials to be
    // invalidated and the context torn down, every subsequent call must take the CPU path,
    // including computeLikelihoodFromBuffer, which otherwise has no buffer left to read.
    if (g_gpu.failed) return GPU_R_ENGINE_FAILED;
    ModelSubst        *m = tree->getModel();
    RateHeterogeneity *r = tree->getRate();
    ModelFactory      *f = tree->getModelFactory();
    if (!m || !r || !f || !tree->aln || !tree->params) return GPU_R_NO_MODEL;

    // DATA TYPE. Both halves of each conjunct matter, and testing num_states alone is a bug
    // waiting to happen: a SEQ_MORPH alignment can also have num_states == 4, but it takes the
    // default branch of Alignment::computeUnknownState() (alignment.cpp:1052-1054), which gives
    // STATE_UNKNOWN == 4 and a five-row tip table with no ambiguity rows at all -- a different
    // geometry that this engine has never been validated on. SEQ_POMO is worse: one of its
    // sampling methods sets STATE_UNKNOWN to 0xffffffff (alignment.cpp:1047-1051), which would
    // size the tip table from a garbage row count. Exclude both by construction rather than by
    // hoping the state count separates them.
    const bool aa  = (tree->aln->num_states == 20 && tree->aln->seq_type == SEQ_PROTEIN);
    const bool dna = (tree->aln->num_states == 4  && tree->aln->seq_type == SEQ_DNA);
    if (!aa && !dna) return GPU_R_UNSUPPORTED_DATA;
    // +E (ModelDNAError), reported under the same condition because it is a statement about
    // how the DATA is observed rather than about the rate matrix: it is a reversible ModelDNA
    // subclass whose only override is computeTipLikelihood (modeldnaerror.cpp:89-91), i.e. the
    // map from an observed character to a tip vector. The GPU consumes that pre-baked in
    // tip_partial_lh, so it would PROBABLY just work -- but "probably" is not a validation
    // result, and NO other predicate below excludes it (it is single-matrix, reversible and
    // not site-specific), so it would otherwise reach the engine untested. Excluded explicitly
    // here rather than renumbering the enum, whose positions archived GPUSTATS lines compare.
    if (dynamic_cast<ModelDNAError*>(m) != nullptr) return GPU_R_UNSUPPORTED_DATA;
    if (m->getNMixtures() != 1 || m->isMixture())   return GPU_R_MIXTURE;
    if (!m->useRevKernel())                         return GPU_R_NON_REVERSIBLE;
    if (m->isSiteSpecificModel())                   return GPU_R_SITE_SPECIFIC;
    if (!f->unobserved_ptns.empty() || f->getASC() != ASC_NONE) return GPU_R_ASC;
    if (tree->isMixlen())                           return GPU_R_MIXLEN;
    if (tree->isSuperTree())                        return GPU_R_SUPERTREE;
    if (tree->isTreeMix())                          return GPU_R_TREEMIX;
    // Rooted trees. NOT because the likelihood is different -- ModelMarkov::setReversible()
    // unroots the tree when adapt_tree is set, and phylokernelnew.h:960 (the only place the
    // root would matter) sits inside `if (!model->useRevKernel())`, which is false here by
    // construction. The reason is gpuBuildContext()'s tip-state upload: ModelProtein with
    // -m GTR20 (modelprotein.cpp:992, :1103-1112) passes adapt_tree == false, so it never
    // unroots while useRevKernel() stays true, and on a rooted tree the extra degree-2 node
    // makes the leaf loop read one row past the alignment. Cheaper to exclude the case than
    // to carry a second indexing convention for it.
    if (tree->rooted)                               return GPU_R_ROOTED;
    if (r->getNRate() < 1)                          return GPU_R_NO_RATE_CATS;
    if (tree->params->robust_phy_keep < 1.0 || tree->params->robust_median)
        return GPU_R_ROBUST_PHY;
    return GPU_OK;
}

// Whether the _pattern_lh device-to-host copy may be deferred to the first reader.
//
// Excluded for PARTITIONED runs, and the reason is specific rather than cautious:
// PhyloSuperTree::getBestNNIForBran memcpys previously saved pattern likelihoods INTO a part's
// _pattern_lh (phylosupertree.cpp:1037) and then reads them back through
// PhyloTree::computePatternLikelihood. A deferred device copy landing between the two would
// overwrite host values the GPU never produced. The parts of a super tree are plain PhyloTrees,
// so isSuperTree() is false for them and gpu_eligible() lets them drive the device; the run
// constant is the only thing that distinguishes them. Everything else that reads _pattern_lh
// goes through the two synced readers in phylotree.cpp.
static inline bool gpu_ptn_lh_deferrable(const PhyloTree *tree) {
    return tree->params && tree->params->partition_file == NULL;
}

// ------------------------------------------------------------------------------------------
// Slot index of a neighbour's partial, i.e. the offset of its host buffer within IQ-TREE's
// own arena. Returns -1 for anything unrecognised, which the callers treat as a hard error
// rather than guessing.
// ------------------------------------------------------------------------------------------
int PhyloTree::gpuSlotIndex(PhyloNeighbor *nei) {
    const double *p = nei->partial_lh;
    if (!p) return -1;
    const size_t block = g_gpu.host_block;
    if (p >= central_partial_lh &&
        p < central_partial_lh + (size_t)g_gpu.host_slots * block) {
        ptrdiff_t d = p - central_partial_lh;
        if (d % (ptrdiff_t)block) return -1;
        return (int)(d / (ptrdiff_t)block);
    }
    // The NNI evaluator re-points the two central neighbours at nni_partial_lh
    // (phylotree.cpp:4283). Those are two more slots, appended after the arena.
    if (nni_partial_lh && p >= nni_partial_lh &&
        p < nni_partial_lh + 2 * block) {
        ptrdiff_t d = p - nni_partial_lh;
        if (d % (ptrdiff_t)block) return -1;
        return g_gpu.host_slots + (int)(d / (ptrdiff_t)block);
    }
    return -1;
}

// Device buffers for a slot, allocated on first touch. Allocating lazily means the resident
// footprint is the number of slots the tree actually uses, not the worst-case cap.
static GpuSlot *gpu_slot(int idx) {
    if (idx < 0 || idx >= g_gpu.nslots) return nullptr;
    GpuSlot &s = g_gpu.slots[idx];
    if (!s.lh) {
        s.lh    = iqtree_gpu_lik_alloc_partial(g_gpu.eng);
        s.scale = iqtree_gpu_lik_alloc_scale(g_gpu.eng);
        if (!s.lh || !s.scale) {
            // Leave nothing half-allocated: a slot holding a partial but no scale counters
            // would look valid to the next caller and silently read uninitialised counters.
            iqtree_gpu_lik_free(g_gpu.eng, s.lh);
            iqtree_gpu_lik_free(g_gpu.eng, s.scale);
            s.lh = nullptr;
            s.scale = nullptr;
            gpu_error("slot allocation (out of device memory?)");
            return nullptr;
        }
    }
    return &s;
}

// Slot lookup for a READ. Same as gpu_slot() plus the staleness check described next to
// GpuContext::gen: the slot must have been written by computePartialLikelihoodGPU since the last
// invalidation. Never allocates -- an unallocated slot has by definition never been written, and
// silently handing back a fresh zeroed buffer would hide exactly the bug this catches.
static GpuSlot *gpu_slot_read(int idx, const char *what) {
    if (idx < 0 || idx >= g_gpu.nslots) return nullptr;
    GpuSlot &s = g_gpu.slots[idx];
    if (!s.lh || !s.scale) return nullptr;
    if (s.gen != g_gpu.gen) {
        fprintf(stderr, "[IQ-TREE3-GPU] %s: device slot %d holds a partial from before the "
                        "last context invalidation (slot generation %lld, context %lld); "
                        "abandoning the GPU path rather than using it.\n",
                what, idx, s.gen, g_gpu.gen);
        fflush(stderr);
        g_gpu.error_seen = true;
        return nullptr;
    }
    return &s;
}

// ------------------------------------------------------------------------------------------
// Context creation / per-evaluation refresh
// ------------------------------------------------------------------------------------------

// True when the cached context still describes this tree/model/alignment exactly, so that
// nothing at all has to be done. Unchanged: every condition here still has to hold.
bool PhyloTree::gpuContextMatches() {
    return g_gpu.eng
        && g_gpu.tree == this
        && g_gpu.central_lh == central_partial_lh
        && g_gpu.nni_lh == nni_partial_lh
        && g_gpu.host_block == getPartialLhSize()
        && g_gpu.host_scale == getScaleNumSize()
        && g_gpu.npat == (int)aln->size()
        && g_gpu.ncat == (int)getRate()->getNRate()
        && g_gpu.safe_numeric == (safe_numeric ? 1 : 0)
        && g_gpu.nleaf == (int)leafNum;
}

// ------------------------------------------------------------------------------------------
// THE ARENA TEST. True when every device allocation the existing engine holds is the right SIZE
// and the right CONTENT for this tree -- i.e. when the only thing that can still be wrong is the
// binding (which tree, which host arena, which category count), and that is repairable by
// gpuRebind() / gpuSwitchNcat() without a single cudaMalloc.
//
// Every condition below is a rebuild trigger, and each is here for a specific reason:
//
//   aln                   the device tip-state columns are uploaded once, from the alignment,
//                         and SURVIVE a rebind. Two trees with equal npat and equal leafNum but
//                         different alignments would alias one alignment's states onto the
//                         other's likelihood, silently. Pointer identity, plus the leaf-id check
//                         at the bottom, is what makes reusing the columns sound.
//   npat / safe_numeric   pattern_stride, partial_elems and scale_elems are all derived from
//                         them, so every allocation in the engine would be the wrong size.
//   ntip_states           rows of d_tip_lh and of the collapsed d_leaf tables.
//   nleaf                 rows of d_tip_states, and the tip_slot a leaf's node id indexes.
//   nodenum / max_slots   host_slots is max(max_lh_slots, nodeNum-leafNum) and nslots is
//   host_slots / nslots   host_slots+2; a shorter slot table means gpuSlotIndex() can return an
//                         index the table does not have, a longer one means device slots that
//                         were never written. Checked as inputs AND as the derived counts.
//   echild_slots          d_echild / d_leaf hold cfg.echild_slots slots at a fixed pitch. Fewer
//                         children than the arena holds is fine; more would run off the end.
//   ncat <= ncat_cap      the capacity argument (see the top of this file).
//   G matrix              its rows are indexed by ONE tree's branch ids. There is no rebinding
//                         that; --dating mcmctree rebuilds, as it always did. Tested in both
//                         directions so neither "was dating, now is not" nor the reverse slips
//                         through.
//   leaf id -> alignment  the mapping the resident tip-state columns were uploaded under.
//
// What is deliberately NOT here is exactly the set gpuRebind() repairs: the tree pointer,
// central_partial_lh, nni_partial_lh, host_block and host_scale. host_block / host_scale are
// also ncat-derived (getPartialLhSize() and getScaleNumSize() both multiply by
// site_rate->getNRate(), phylotree.cpp:1228-1245); gpuContextMatches() still checks them, against
// the ncat actually in force afterwards. Nothing is dropped from the rebuild criteria -- the
// derived quantities are simply tested at the right moment, which is after the repair.
bool PhyloTree::gpuGeometryMatches() {
    if (!g_gpu.eng) return false;
    if (g_gpu.aln != aln) return false;
    if (g_gpu.npat != (int)aln->size()) return false;
    if (g_gpu.nstates != aln->num_states) return false;
    if (g_gpu.ntip_states != aln->STATE_UNKNOWN + 1) return false;
    if (g_gpu.safe_numeric != (safe_numeric ? 1 : 0)) return false;
    if (g_gpu.nleaf != (int)leafNum) return false;
    if (g_gpu.nodenum != (int)nodeNum) return false;
    if (g_gpu.max_slots != max_lh_slots) return false;

    const int host_slots = (int)std::max<int64_t>(max_lh_slots,
                                                  (int64_t)nodeNum - (int64_t)leafNum);
    if (g_gpu.host_slots != host_slots || g_gpu.nslots != host_slots + 2) return false;

    const int ncat = (int)getRate()->getNRate();
    if (ncat < 1 || ncat > g_gpu.ncat_cap) return false;

    if (g_gpu.gmat || gpu_wants_gmatrix(this)) return false;
    if (gpu_max_children(this) > g_gpu.echild_slots) return false;
    return gpu_leaf_ids_match_alignment(this);
}

// The other cheap path: re-point the existing arena at a DIFFERENT PhyloTree. No cudaFree, no
// cudaMalloc, no cudaMemset of the pooled slots -- only the binding moves.
//
// The slot table is not a stored array: gpuSlotIndex() derives it on the fly from central_lh,
// nni_lh and host_block, so re-deriving it IS assigning those three. g_gpu.slots itself is the
// pool of DEVICE buffers, indexed by that slot number, and it is exactly what we are keeping:
// gpuGeometryMatches() has already established that it has the right length and that each entry
// is the right size.
//
// iqtree_gpu_lik_rebind() does the engine's half -- theta, buffer_scale, echild and the leaf
// tables zeroed, the rate/proportion vectors marked stale through the same guard set_ncat uses.
// The caller (gpuPrepare) owns the host half; see the block comment there.
bool PhyloTree::gpuRebind() {
    if (iqtree_gpu_lik_rebind(g_gpu.eng) != 0) {
        gpu_error("rebind device context to a new tree");
        return false;
    }
    g_gpu.tree       = this;
    g_gpu.central_lh = central_partial_lh;
    g_gpu.nni_lh     = nni_partial_lh;
    g_gpu.host_block = getPartialLhSize();
    g_gpu.host_scale = getScaleNumSize();

    // _pattern_lh_cat staging is ncat*npat and is overwritten in full by every download that
    // uses it, so only its LENGTH matters -- and npat is pinned by gpuGeometryMatches(). Resize
    // only when the category count moved, because assign() on the largest benchmark genus is a
    // 12 MB memset that would otherwise run on every ModelFinder candidate.
    const size_t want = (size_t)g_gpu.ncat * (size_t)g_gpu.npat;
    if (g_gpu.h_lhcat.size() != want) g_gpu.h_lhcat.assign(want, 0.0);

    // The OUTPUT CONTRACT applies per TREE, not per arena: theta_all, buffer_scale_all,
    // _pattern_lh_cat and G_matrix are the NEW tree's own host allocations, and this file
    // maintains none of them. Poisoning them here is what makes the rebind path invalidate as
    // thoroughly as the rebuild path did -- a rebind that skipped it would hand an untraced
    // consumer whatever the fresh allocation happened to contain. gmatrix_enabled is false by
    // construction: gpuGeometryMatches() refuses to rebind a run that wants the G matrix.
    gpuPoisonUnmaintainedBuffers(false);
    return true;
}

// The cheap path: re-point the existing engine at a new rate-category count. No cudaFree, no
// cudaMalloc, no cudaMemset of the arena -- only the launch geometry and the host-side view of
// IQ-TREE's own block sizes move. The caller (gpuPrepare) owns the invalidation that has to go
// with it; see the comment there for why that is not optional.
bool PhyloTree::gpuSwitchNcat() {
    const int ncat = (int)getRate()->getNRate();
    if (iqtree_gpu_lik_set_ncat(g_gpu.eng, ncat) != 0) {
        gpu_error("set ncat");
        return false;
    }
    g_gpu.ncat       = ncat;
    // NOTE: cached graphs are NOT freed here. Each graph carries its own ncat in the cache key,
    // and gpuReplayOrComputeSweep only replays one whose ncat == the current ncat -- so a graph
    // captured at another ncat is simply never selected. The device slot bases the graph bakes
    // in are CAPACITY-sized (partial_elems_cap / theta pitch), so set_ncat does not move them.
    // Keeping graphs across ncat switches is what lets ModelFinder reuse a full-traversal graph
    // across every candidate that shares its rate-category count.
    // Both are ncat-dependent, and host_block in particular is what gpuSlotIndex() divides by
    // to turn a host partial_lh pointer into a slot index. Leaving it stale would mis-map every
    // neighbour onto the wrong device slot -- silently, and with plausible-looking numbers.
    g_gpu.host_block = getPartialLhSize();
    g_gpu.host_scale = getScaleNumSize();
    // _pattern_lh_cat staging is ncat*npat; a larger ncat would overrun the old vector.
    g_gpu.h_lhcat.assign((size_t)ncat * g_gpu.npat, 0.0);
    return true;
}

// ------------------------------------------------------------------------------------------
// POISONING (mechanism (1) of the OUTPUT CONTRACT at the top of this file)
// ------------------------------------------------------------------------------------------
// Overwrites every host buffer the GPU path does NOT maintain with signalling NaN, so that a
// consumer nobody traced yields obviously broken output instead of plausible-but-wrong
// numbers. The distinction matters: these buffers are allocated once and written by whatever
// ran last, so an untraced reader normally gets a well-formed likelihood from a different
// branch, a different model, or a different tree -- a wrong answer that survives review.
//
// Correctness argument for each buffer is the table at the top of this file: every one of them
// either has no consumer that can run under gpu_eligible(), or has a consumer that writes it
// first. Poisoning therefore cannot break a run that is already correct; it can only convert a
// silent error into a loud one.
//
// Cost: theta_all dominates at mem_size*nstates*ncat doubles (~250 MB for the largest benchmark
// genus). This runs once per FULL context rebuild -- not on an ncat switch, and never per node
// or per evaluation -- alongside cudaMalloc of a multi-GB device arena, so it is noise against
// what the same function already spends.
void PhyloTree::gpuPoisonUnmaintainedBuffers(bool gmatrix_enabled) {
    const double NaN = std::numeric_limits<double>::signaling_NaN();
    const size_t mem_size = gpu_gmatrix_stride(this);
    const size_t nmix     = model_factory->fused_mix_rate ? 1 : model->getNMixtures();

    if (theta_all)
        std::fill_n(theta_all, mem_size * model->num_states * site_rate->getNRate() * nmix, NaN);
    if (buffer_scale_all)
        std::fill_n(buffer_scale_all, mem_size, NaN);
    if (_pattern_lh_cat)
        std::fill_n(_pattern_lh_cat, mem_size * site_rate->getNDiscreteRate() * nmix, NaN);
    if (G_matrix) {
        // With the MCMCTree export active the rows ARE maintained, and the padding columns are
        // required to be exactly 0.0 (that is what makes the rank-k update equal the CPU's --
        // see gpu_kernels.h). Zeroing also removes the uninitialised read that stock IQ-TREE
        // performs there. Without it, nothing legitimately reads G_matrix, so it is poisoned.
        std::fill_n(G_matrix, (size_t)branchNum * mem_size, gmatrix_enabled ? 0.0 : NaN);
    }
}

bool PhyloTree::gpuBuildContext() {
    gpu_destroy();

    IqtreeGpuLikConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.nstates        = aln->num_states;
    cfg.ncat           = (int)getRate()->getNRate();
    cfg.npat           = (int)aln->size();
    cfg.num_tip_states = aln->STATE_UNKNOWN + 1;
    cfg.state_unknown  = aln->STATE_UNKNOWN;
    // (F) NORM_LH vs SAFE_LH comes straight from the tree. phylotreesse.cpp:102-103 excludes
    // num_states 4 and 20 alike from the "force safe" clause, so BOTH supported data types
    // default to NORM_LH and reach SAFE_LH only via -safe or >= numseq_safe_scaling taxa.
    cfg.safe_numeric   = safe_numeric ? 1 : 0;
    cfg.tip_slots      = (int)leafNum;
    cfg.theta_slots    = GPU_THETA_SLOTS;
    cfg.device         = 0;

    // One echild slot per child of the widest node; a bifurcating tree needs two.
    cfg.echild_slots = gpu_max_children(this);

    // ---- rate-category capacity -------------------------------------------------------
    // Slot count first: the capacity has to be chosen against what the pooled slots will cost.
    const int host_slots = (int)std::max<int64_t>(max_lh_slots,
                                                  (int64_t)nodeNum - (int64_t)leafNum);
    const int nslots = host_slots + 2;

    int ncat_cap = gpu_ncat_capacity_wanted(params, cfg.ncat);
    // VRAM guard. A run that reaches +R{cap} would have to hold this footprint anyway, so the
    // capacity does not raise the peak that run needs -- it only reaches it sooner. What it can
    // do is commit memory for categories a run never gets to (the +R search stops early on the
    // information criterion), so it is bounded here rather than trusted. Shrinking one category
    // at a time degrades to the old behaviour instead of falling off a cliff.
    size_t total_vram = 0;
    if (ncat_cap > cfg.ncat && iqtree_gpu_device_mem(nullptr, &total_vram) == 0 && total_vram) {
        const double budget = (double)total_vram * GPU_ARENA_VRAM_FRACTION;
        int requested = ncat_cap;
        while (ncat_cap > cfg.ncat &&
               (double)iqtree_gpu_lik_slot_bytes(cfg.npat, ncat_cap, cfg.nstates,
                                                 cfg.safe_numeric)
                   * (double)nslots > budget)
            ncat_cap--;
        if (ncat_cap < requested) {
            printf("[IQ-TREE3-GPU] rate-category capacity reduced from %d to %d: %d slots at "
                   "ncat=%d would need %.2f GB of the %.2f GB on device.\n",
                   requested, ncat_cap, nslots, requested,
                   (double)iqtree_gpu_lik_slot_bytes(cfg.npat, requested, cfg.nstates,
                                                     cfg.safe_numeric)
                       * (double)nslots / 1073741824.0,
                   (double)total_vram / 1073741824.0);
            fflush(stdout);
        }
    }
    cfg.ncat_cap = ncat_cap;

    if (iqtree_gpu_lik_create(&g_gpu.eng, &cfg) != 0) {
        gpu_error("engine create");
        g_gpu.eng = nullptr;
        return false;
    }

    g_gpu.tree         = this;
    g_gpu.central_lh   = central_partial_lh;
    g_gpu.nni_lh       = nni_partial_lh;
    g_gpu.host_block   = getPartialLhSize();
    g_gpu.host_scale   = getScaleNumSize();
    g_gpu.aln          = aln;
    g_gpu.npat         = cfg.npat;
    g_gpu.nstates      = cfg.nstates;
    g_gpu.ncat         = cfg.ncat;
    g_gpu.ncat_cap     = iqtree_gpu_lik_ncat_cap(g_gpu.eng);
    g_gpu.safe_numeric = cfg.safe_numeric;
    g_gpu.nleaf        = cfg.tip_slots;
    g_gpu.ntip_states  = cfg.num_tip_states;
    g_gpu.echild_slots = cfg.echild_slots;
    g_gpu.nodenum      = (int)nodeNum;
    g_gpu.max_slots    = max_lh_slots;
    // LM_PER_NODE indexes one slot per internal node, LM_MEM_SAVE indexes max_lh_slots of
    // them; take the larger so neither addressing scheme can run off the end.
    g_gpu.host_slots   = host_slots;
    g_gpu.nslots       = nslots;
    g_stats.full_rebuilds++;
    g_stats.ncat_cap = g_gpu.ncat_cap;
    g_gpu.slots.assign(g_gpu.nslots, GpuSlot());
    g_gpu.h_lhcat.assign((size_t)cfg.ncat * cfg.npat, 0.0);
    g_gpu.h_states.assign(cfg.npat, cfg.state_unknown);

    // ---- MCMCTree G matrix ------------------------------------------------------------
    // Enabled only for `--dating mcmctree`, which is the only consumer of G_matrix and is
    // run-constant. Everything about this is a hard requirement rather than an optimisation:
    // if the export is requested and G cannot be produced exactly as the host consumer expects
    // it, the GPU must not run at all, because the resulting Hessian would be silently wrong.
    if (gpu_wants_gmatrix(this)) {
        const size_t stride = gpu_gmatrix_stride(this);
        // The CPU kernel writes G at branch_id * g_matrix_nptn (phylokernelnew.h:2277-2279),
        // rounding the pattern count with VectorClass::size(); the allocation and the consumer
        // both round it with get_safe_upper_limit(). They agree whenever vector_size matches
        // the SIMD level get_safe_upper_limit() reads out of Params, which is every real build
        // -- but if they ever did not, a run that mixed a CPU-fallback branch with GPU branches
        // would interleave two different row pitches in one buffer. Checked, not assumed.
        const size_t vs = (vector_size > 0) ? vector_size : 1;
        const size_t cpu_stride = (((size_t)cfg.npat + vs - 1) / vs) * vs
                                + std::max(get_safe_upper_limit(model->num_states),
                                           get_safe_upper_limit(model_factory->unobserved_ptns.size()));
        if (vector_size == 0 || cpu_stride != stride) {
            printf("[IQ-TREE3-GPU] --dating mcmctree: G_matrix row pitch disagrees between "
                   "the CPU kernel (%llu) and the allocation (%llu); using the CPU kernel.\n",
                   (unsigned long long)cpu_stride, (unsigned long long)stride);
            fflush(stdout);
            gpu_destroy();
            return false;
        }
        if (iqtree_gpu_lik_gmatrix_enable(g_gpu.eng, (int)branchNum, (int)stride) != 0) {
            gpu_error("enable G matrix for --dating mcmctree");
            gpu_destroy();
            return false;
        }
        g_gpu.gmat         = true;
        g_gpu.gmat_nbranch = (int)branchNum;
        g_gpu.gmat_stride  = stride;
        g_gpu.gmat_written.assign(branchNum, 0);
        g_gpu.gmat_rows    = 0;
    }

    // Mechanism (1) of the OUTPUT CONTRACT. Deliberately after the engine exists, so that a
    // failed build leaves the host buffers exactly as the CPU kernel left them.
    gpuPoisonUnmaintainedBuffers(g_gpu.gmat);

    // Leaf state columns are a property of the alignment, so they are uploaded once.
    for (int id = 0; id < cfg.tip_slots; id++) {
        for (int p = 0; p < cfg.npat; p++)
            g_gpu.h_states[p] = (int)(aln->at(p)[id]);
        if (iqtree_gpu_lik_set_tip_states(g_gpu.eng, id, g_gpu.h_states.data()) != 0) {
            gpu_error("upload tip states");
            gpu_destroy();
            return false;
        }
    }

    // ModelFinder re-selects a model (and so rebuilds this context) hundreds of times, so the
    // footprint report is printed only when it actually changes. The MB figures are the
    // ALLOCATED (capacity) sizes, because that is what the slots actually cost; the current
    // ncat only decides how much of each slot the kernels touch.
    const double partial_mb = (double)iqtree_gpu_lik_partial_bytes_cap(g_gpu.eng) / 1048576.0;
    const double scale_mb   = (double)iqtree_gpu_lik_scale_bytes_cap(g_gpu.eng) / 1048576.0;
    static double reported_gb = -1.0;
    const double footprint_gb = (partial_mb + scale_mb) * g_gpu.nslots / 1024.0;
    if (footprint_gb != reported_gb) {
        reported_gb = footprint_gb;
        printf("[IQ-TREE3-GPU] engine: %s (%d states), %d patterns (stride %d), "
               "%d rate categories (arena sized for up to %d), %s scaling.\n",
               cfg.nstates == 4 ? "nucleotide" : "amino-acid", cfg.nstates,
               cfg.npat, iqtree_gpu_lik_pattern_stride(g_gpu.eng), cfg.ncat, g_gpu.ncat_cap,
               cfg.safe_numeric ? "SAFE_LH" : "NORM_LH");
        printf("[IQ-TREE3-GPU] partial slot = %.1f MB + %.1f MB counters; pooling %d slots "
               "(IQ-TREE arena %lld + 2 NNI) -> %.2f GB VRAM at full occupancy.\n",
               partial_mb, scale_mb, g_gpu.nslots, (long long)g_gpu.host_slots, footprint_gb);
        fflush(stdout);
    }
    return true;
}

// Uploads everything that a model or rate update can change. Called once per evaluation that
// drives a traversal, never per node. The eigen decomposition, the rate/proportion vectors,
// tip_partial_lh and ptn_invar all move when the model is re-optimised, and ptn_invar in
// particular is recomputed by computePtnInvar() whenever p_invar or the state frequencies
// change; re-uploading unconditionally is a few hundred kB and removes a whole class of
// stale-buffer bugs.
bool PhyloTree::gpuUploadModel() {
    ModelSubst *model = getModel();
    RateHeterogeneity *rate = getRate();

    // tip_partial_lh, ptn_freq and ptn_invar are all (re)filled by computeTipPartialLikelihood,
    // and clearAllPartialLH() invalidates it. Refreshing here rather than in the traversal is
    // what guarantees the three host arrays are current at the moment they are uploaded.
    if ((tip_partial_lh_computed & 1) == 0) computeTipPartialLikelihood();
    if (!tip_partial_lh || !ptn_freq || !ptn_invar) return false;

    if (iqtree_gpu_lik_set_eigen(g_gpu.eng, model->getEigenvalues(), model->getEigenvectors(),
                                 model->getInverseEigenvectors()) != 0) {
        gpu_error("upload eigen"); return false;
    }
    std::vector<double> rates(g_gpu.ncat), props(g_gpu.ncat);
    for (int c = 0; c < g_gpu.ncat; c++) {
        rates[c] = rate->getRate(c);
        // getProp() already carries the (1 - p_invar) factor for +I models, exactly as the
        // CPU kernel assumes; it must not be applied a second time downstream.
        props[c] = rate->getProp(c);
    }
    if (iqtree_gpu_lik_set_rates(g_gpu.eng, rates.data(), props.data()) != 0) {
        gpu_error("upload rates"); return false;
    }
    if (iqtree_gpu_lik_set_tip_partials(g_gpu.eng, tip_partial_lh) != 0) {
        gpu_error("upload tip partials"); return false;
    }
    if (iqtree_gpu_lik_set_pattern_data(g_gpu.eng, ptn_freq, ptn_invar) != 0) {
        gpu_error("upload pattern data"); return false;
    }
    return true;
}

// Drops the device context and invalidates the host partials, so that a CPU kernel taking
// over never reads host memory the GPU path left untouched. Saves/restores current_it because
// clearAllPartialLH() nulls it and the caller (computeFunction, optimizeOneBranch) still needs
// it after this returns.
// A device error mid-evaluation. Tear the context down, invalidate the host partials so the
// CPU kernel recomputes everything from scratch, and never touch the GPU again in this
// process -- a half-failed device is not something to retry inside a tree search.
void PhyloTree::gpuAbortToCpu() {
    // --gpu-strict: this is the one-way door from "GPU run" to "CPU run with a GPU banner
    // already in the log", so under strict it is where the run stops. In practice every caller
    // has already gone through gpu_error()/gpu_delegate(), which abort first; this is the
    // backstop for the paths that set error_seen directly (gpu_slot_read's staleness check).
    if (g_gpu_strict) {
        fflush(stdout);
        outError("IQ-TREE3-GPU --gpu-strict: a device error has forced the GPU path to be "
                 "abandoned. Plain -gpu would finish this run on the CPU kernel; --gpu-strict "
                 "stops instead, because everything after this point would be a CPU "
                 "measurement reported under a GPU banner.");
    }
    g_gpu.failed = true;
    gpuLeave();
}

void PhyloTree::gpuLeave() {
    if (!g_gpu.eng || g_gpu.tree != this) return;
    gpu_destroy();
    PhyloNeighbor *it = current_it, *itb = current_it_back;
    clearAllPartialLH();
    current_it = it;
    current_it_back = itb;
    theta_computed = false;
}

// ------------------------------------------------------------------------------------------
// PhyloTree hooks
// ------------------------------------------------------------------------------------------

// The consumer side of the deferred _pattern_lh copy. Called from the only two places outside
// the likelihood kernels that read _pattern_lh: PhyloTree::computeLikelihood() before it
// memmoves into the caller's buffer, and PhyloTree::computePatternLikelihood(), which reports
// the per-pattern values of whatever was computed last without recomputing anything.
//
// Note it does NOT check that the pending copy belongs to `this`. It cannot: the destination is
// recorded with the copy, and if it names another tree's buffer then that is the tree the copy
// is owed to, and this tree's _pattern_lh is untouched either way. Paying it here is harmless
// and keeps the debt from outliving the device context.
void PhyloTree::gpuSyncPatternLh() {
    gpu_flush_ptn_lh();
}

// The other side: the destination is about to be freed, so the debt is cancelled rather than
// paid. Guarded on the pointer so that freeing tree A's buffers cannot cancel a copy owed to
// tree B -- ModelFinder has both alive at once.
void PhyloTree::gpuForgetPatternLh() {
    if (g_gpu.ptn_lh_pending && g_gpu.ptn_lh_host == _pattern_lh) {
        g_gpu.ptn_lh_pending = false;
        g_gpu.ptn_lh_host    = nullptr;
    }
}

// Builds (or refreshes) the device context and uploads the current model. Returns false when
// the caller must fall back to the CPU for this call; in that case the host partials have
// already been invalidated so the CPU starts from a clean slate.
bool PhyloTree::gpuPrepare() {
    if (g_gpu.failed) return false;
    if (!gpuContextMatches()) {
        // Any deferred _pattern_lh copy is owed against the device buffer this block is about
        // to destroy (gpuBuildContext -> gpu_destroy) or zero (iqtree_gpu_lik_rebind). Pay it
        // first. Once per ModelFinder candidate at worst, so the transfer it costs is exactly
        // the one the deferral was never trying to avoid.
        gpu_flush_ptn_lh();

        // Entering GPU mode: the CPU may have set partial_lh_computed bits for host slots the
        // device has never seen. Drop them all before the first GPU traversal.
        PhyloNeighbor *it = current_it, *itb = current_it_back;

        // Neither a change of rate-category count nor a change of TREE needs a new arena -- see
        // the two notes at the top of this file. Only a change of geometry does.
        //
        // The post-condition is what makes this safe rather than merely plausible: after either
        // cheap path the context must satisfy gpuContextMatches() in FULL, with every one of its
        // original conditions. If it does not -- because something moved that
        // gpuGeometryMatches() does not model -- we fall through to a real rebuild instead of
        // running on a context we cannot account for. A cheap path that fails part-way is also
        // safe by construction, because gpuBuildContext() starts with gpu_destroy().
        bool ok = false;
        if (gpuGeometryMatches()) {
            const bool want_ncat   = (g_gpu.ncat != (int)getRate()->getNRate());
            const bool want_rebind = (g_gpu.tree       != this
                                   || g_gpu.central_lh != central_partial_lh
                                   || g_gpu.nni_lh     != nni_partial_lh);
            // ncat FIRST, because gpuRebind() sizes the _pattern_lh_cat staging buffer from
            // g_gpu.ncat and must see the count that will actually be in force. The two are
            // independent otherwise: each invalidates a superset of what the other does.
            ok = true;
            if (ok && want_ncat)   ok = gpuSwitchNcat();
            if (ok && want_rebind) ok = gpuRebind();
            ok = ok && gpuContextMatches();
            if (ok) {
                // A rebind subsumes an ncat switch (it re-invalidates everything the switch
                // did), so it is counted as the one transition that happened.
                if (want_rebind)    g_stats.rebinds++;
                else if (want_ncat) g_stats.ncat_switches++;
            }
        }
        if (!ok && !gpuBuildContext()) {
            g_gpu.failed = true;
            clearAllPartialLH();
            current_it = it; current_it_back = itb;
            theta_computed = false;
            return false;
        }
        // THE INVALIDATION IS SHARED BY ALL THREE PATHS, ON PURPOSE.
        // It is tempting to think a cheap path can skip it because nothing was freed. It cannot,
        // for two independent reasons.
        //
        // ncat: a partial is L~[c][k] over c = 0..ncat-1, so the category count is part of what
        // the numbers MEAN, not just of how many of them there are. Every device slot, every
        // scale counter and both theta buffers still hold values computed at the old category
        // count, and IQ-TREE's partial_lh_computed bits still claim they are current. Reusing
        // them would mix categories from two different models into one likelihood.
        //
        // rebind: every device slot holds a partial of the OLD tree -- a different topology and
        // different branch lengths -- while the NEW tree's partial_lh_computed bits are its own
        // and say nothing about what the device contains. clearAllPartialLH() on the new tree
        // clears every one of those bits, which pushes every neighbour into traversal_info,
        // which rewrites its device slot before anything reads it. That is precisely the SLOT
        // POOLING argument at the top of this file, applied to a tree the pool was not built
        // for: "host slot k is valid for neighbour X" and "device slot k is valid for
        // neighbour X" are made the same statement again by clearing all of the former.
        //
        // Either way the alternative is a wrong answer that looks entirely plausible, which is
        // far worse than the reallocation this change removes.
        //
        // The slot generation is bumped HERE, in the same statement group as clearAllPartialLH()
        // and for the same reason, so that the device-side notion of "stale" cannot drift from
        // the host-side one: every device slot written before this point is now stale, and
        // gpu_slot_read() will say so instead of returning it. See GpuContext::gen.
        g_gpu.gen++;
        gpuFreeFullGraphs();   // every cached graph's slots are now stale under the new gen
        clearAllPartialLH();
        current_it = it; current_it_back = itb;
        // theta_computed is a plain member that IQ-TREE never initialises in the constructor.
        // Having just invalidated every partial, any theta the flag still vouches for is
        // meaningless, so it is cleared here as well. On the cheap paths it is doubly required:
        // computeLikelihoodFromBufferGPU reads the Newton theta slot on the strength of this
        // flag alone, and that slot holds either ncat_old*nstates*stride meaningful doubles or the
        // previous tree's theta. On a freshly constructed IQTree the member is uninitialised, so
        // clearing it here is also what stops a rebind from reading a garbage true. This is the
        // only place this file writes it other than the one mandated assignment at the end of
        // computeLikelihoodDervGPU.
        theta_computed = false;
    }
    if (!gpuUploadModel()) {
        g_gpu.failed = true;
        gpuLeave();
        return false;
    }
    return true;
}

// Mirrors PhyloTree::computeTraversalInfo<VectorClass>(node, dad, false), minus the CPU-side
// computePartialInfo() precompute: the GPU builds echild and the collapsed leaf tables itself
// from the eigen decomposition, so info.echildren / info.partial_lh_leaves are never read.
// Everything else -- the mem-slot allocation, reorientPartialLh, the partial_lh_computed
// bookkeeping and the post-order push into traversal_info -- is IQ-TREE's own code, called
// through the non-template overload, so the slot recycling this file depends on is exactly
// IQ-TREE's.
void PhyloTree::gpuTraverse(PhyloNode *node, PhyloNode *dad) {
    if ((tip_partial_lh_computed & 1) == 0) computeTipPartialLikelihood();
    traversal_info.clear();

    const size_t ncat_mix = site_rate->getNRate();
    const size_t block    = aln->num_states * ncat_mix;
    // Same reservation as the CPU path so that the buffer arithmetic inside
    // computeTraversalInfo() stays inside buffer_partial_lh even though we never read it.
    double *buffer = buffer_partial_lh + block * vector_size * num_packets
                   + get_safe_upper_limit(block) * (aln->STATE_UNKNOWN + 2);

    if (params->lh_mem_save == LM_MEM_SAVE) {
        // Traverse the bigger side first, which is what makes max_lh_slots sufficient.
        if (node->computeSize(dad) < dad->computeSize(node)) std::swap(node, dad);
    }

    PhyloNeighbor *dad_branch  = (PhyloNeighbor*)dad->findNeighbor(node);
    PhyloNeighbor *node_branch = (PhyloNeighbor*)node->findNeighbor(dad);
    bool dad_locked  = computeTraversalInfo(dad_branch, dad, buffer);
    bool node_locked = computeTraversalInfo(node_branch, node, buffer);
    if (params->lh_mem_save == LM_MEM_SAVE) {
        if (dad_locked)  mem_slots.unlock(dad_branch);
        if (node_locked) mem_slots.unlock(node_branch);
    }
}

// theta[i] = Ltilde_node[i] * Ltilde_dad[i] over the whole block, plus the per-pattern scale
// combination, written into `slot`. This is the device analogue of
// computeLikelihoodBufferGenericSIMD; see the theta-slot note in gpu_kernels.h for why the
// branch path gets its own slot.
bool PhyloTree::gpuComputeTheta(PhyloNeighbor *dad_branch, PhyloNode *dad, int theta_slot) {
    PhyloNode *node = (PhyloNode*)dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*)node->findNeighbor(dad);

    if (iqtree_gpu_lik_set_theta_slot(g_gpu.eng, theta_slot) != 0) {
        gpu_error("select theta slot"); return false;
    }
    GpuSlot *sd = gpu_slot_read(gpuSlotIndex(dad_branch), "theta: dad slot");
    if (!sd) { gpu_error("theta: dad slot"); return false; }

    int rc = 0;
    if (dad->isLeaf()) {
        // A leaf has no per-category partial: the same nstates-element tip vector multiplies every
        // rate category, and only the node side carries scale counters.
        rc = iqtree_gpu_lik_theta_tip(g_gpu.eng, dad->id, sd->lh, sd->scale);
    } else {
        GpuSlot *sn = gpu_slot_read(gpuSlotIndex(node_branch), "theta: node slot");
        if (!sn) { gpu_error("theta: node slot"); return false; }
        rc = iqtree_gpu_lik_theta_internal(g_gpu.eng, sn->lh, sn->scale, sd->lh, sd->scale);
    }
    if (rc != 0) { gpu_error("theta"); return false; }
    return true;
}

// ------------------------------------------------------------------------------------------
// STEP 2: full-traversal CUDA-graph replay
// ------------------------------------------------------------------------------------------
void PhyloTree::gpuGatherBranchLengths(std::vector<double> &lens) {
    // Indexed by Neighbor::id, which k_echild_dev uses as its branch index. Both directed
    // neighbours of an edge share id and length, so either fills the slot.
    lens.assign((size_t)std::max(0, branchNum), 0.0);
    NodeVector n1, n2;
    getBranches(n1, n2);
    for (size_t i = 0; i < n1.size(); i++) {
        Neighbor *nei = n1[i]->findNeighbor(n2[i]);
        if (nei && nei->id >= 0 && nei->id < branchNum) lens[nei->id] = nei->length;
    }
}

// Off switch for the experimental CUDA-graph replay caches (full-traversal sweep, and the Newton
// derivative/logL singleton graphs). Set env IQTREE_GPU_NO_GRAPH_REPLAY to force the proven per-launch
// path -- the default path, which is correct on every tree size. The small-tree nucleotide NNI
// failure lives in the replay caching, not in the on-device Newton loop.
static int gpu_graph_replay_enabled() {
    // OFF by default. The experimental CUDA-graph replay caching is both UNSOUND and REDUNDANT:
    //  - Unsound: the full-traversal replay (gpuReplayOrComputeSweep) replays a graph for the wrong
    //    topology during NNI on small DNA trees, producing a slightly-off logL that trips
    //    optimizeNNI's "Using one NNI reduces LogL" assertion. Disabling
    //    the full-graph cache alone fixes it; disabling the small-graph derv/logl caches does not.
    //  - Redundant: once the on-device Newton loop is active it carries the entire speedup -- the n30
    //    1-thread speedup is identical (~4.36x) with the caching on or off. The caching buys nothing.
    // The correct, fast, fully-GPU-engaged path is per-launch kernels + the on-device Newton loop.
    // Opt in with IQTREE_GPU_ENABLE_GRAPH_REPLAY only for isolated experiments; never in production.
    static int v = -1;
    if (v < 0) v = (getenv("IQTREE_GPU_ENABLE_GRAPH_REPLAY") != nullptr) ? 1 : 0;
    return v;
}
// Per-cache sub-switches, to pinpoint which replay cache is unsound on small trees. Each is enabled
// only if the master is on AND its own NO_* env is unset. FULL = the full-traversal sweep replay;
// SMALL = the Newton derivative/logL singleton graphs.
static int gpu_full_replay_enabled() {
    static int v = -1;
    if (v < 0) v = (getenv("IQTREE_GPU_NO_FULL_REPLAY") == nullptr) ? 1 : 0;
    return gpu_graph_replay_enabled() && v;
}
static int gpu_small_replay_enabled() {
    static int v = -1;
    if (v < 0) v = (getenv("IQTREE_GPU_NO_SMALL_REPLAY") == nullptr) ? 1 : 0;
    return gpu_graph_replay_enabled() && v;
}

bool PhyloTree::gpuReplayOrComputeSweep(PhyloNeighbor *dad_branch, PhyloNode *dad, int theta_slot) {
    // Signature that pins the exact recorded kernel sequence: the ordered branch ids of the
    // nodes this traversal computes. gen + ncat + theta_slot pin the slot pointers, the launch
    // shapes and the theta buffer respectively. A four-way match is a guarantee the recorded
    // launches are still the correct ones for this evaluation.
    std::vector<int> sig;
    sig.reserve(traversal_info.size());
    for (TraversalInfo &ti : traversal_info) sig.push_back(ti.dad_branch->id);

    const int  ncat = (int)site_rate->getNRate();
    // "Full" == every internal node recomputed. Only full traversals are captured, because only
    // they recur identically (a model-parameter change dirties all partials; the incremental
    // dirty-subset sweeps of branch-length optimisation do not repeat). The threshold is a
    // capture-effort gate only: correctness rests on the exact-signature match, not on this.
    const bool full = traversal_info.size() >= (size_t)std::max(0, (int)leafNum - 2)
                      && !traversal_info.empty();

    // ---- 1. Replay a cached graph if this exact full traversal has been seen. ----
    if (full && gpu_full_replay_enabled()) {
        for (GpuContext::FullGraph &e : g_gpu.full_graphs) {
            if (e.gen != g_gpu.gen || e.theta_slot != theta_slot || e.ncat != ncat || e.sig != sig)
                continue;
            std::vector<double> lens;
            gpuGatherBranchLengths(lens);
            if (lens.empty() ||
                iqtree_gpu_lik_set_branch_lengths(g_gpu.eng, lens.data(), (int)lens.size()) != 0) {
                gpu_error("graph replay: set branch lengths");
                return false;
            }
            // The captured theta step wrote `theta_slot`; the downstream branch_logl/derivatives
            // reads theta_of(current slot), so point the engine at it before replaying.
            if (iqtree_gpu_lik_set_theta_slot(g_gpu.eng, theta_slot) != 0) {
                gpu_error("graph replay: select theta slot");
                return false;
            }
            if (iqtree_gpu_lik_graph_launch(g_gpu.eng, e.graph) != 0) {
                gpu_error("graph replay");
                return false;
            }
            // The replay rewrote every internal node's partial slot (and the theta slot) exactly
            // as the per-node path would; stamp their generation so gpu_slot_read accepts them.
            for (TraversalInfo &ti : traversal_info) {
                PhyloNode *n = (PhyloNode*)ti.dad_branch->node;
                if (n->isLeaf()) continue;
                GpuSlot *o = gpu_slot(gpuSlotIndex(ti.dad_branch));
                if (o) o->gen = g_gpu.gen;
            }
            g_stats.full_replayed++;
            return true;
        }
    }

    // ---- 2. No cached graph: compute this evaluation the proven per-node way. ----
    for (auto it = traversal_info.begin(); it != traversal_info.end(); it++)
        computePartialLikelihoodGPU(*it, 0, 0, 0);
    if (!gpuComputeTheta(dad_branch, dad, theta_slot)) return false;
    if (g_gpu.error_seen) return false;   // this evaluation failed; caller falls back to CPU

    if (!full) { g_stats.subset_traversals++; return true; }

    // ---- 3. Full traversal, first time seen: record it into a replayable graph for next time.
    // The result above is already correct; this pass only RECORDS (capture does not execute), so
    // a failure here is discarded without touching this evaluation's answer.
    std::vector<double> lens;
    gpuGatherBranchLengths(lens);
    if (lens.empty() ||
        iqtree_gpu_lik_set_branch_lengths(g_gpu.eng, lens.data(), (int)lens.size()) != 0)
        return true;   // cannot device-plumb lengths; keep the good result, skip caching

    long long before = iqtree_gpu_lik_launch_count();
    if (iqtree_gpu_lik_graph_capture_begin(g_gpu.eng) != 0) return true;

    g_gpu.capturing = true;                       // -> branch_precompute_dev, stats suppressed
    bool rec_ok = (iqtree_gpu_lik_set_theta_slot(g_gpu.eng, theta_slot) == 0);
    if (rec_ok)
        for (auto it = traversal_info.begin(); it != traversal_info.end(); it++)
            computePartialLikelihoodGPU(*it, 0, 0, 0);
    if (rec_ok) gpuComputeTheta(dad_branch, dad, theta_slot);
    g_gpu.capturing = false;

    IqtreeGpuGraph *graph = nullptr;
    bool end_ok = (iqtree_gpu_lik_graph_capture_end(g_gpu.eng, &graph) == 0);
    if (end_ok && graph && rec_ok && !g_gpu.error_seen) {
        GpuContext::FullGraph e;
        e.sig = std::move(sig); e.gen = g_gpu.gen; e.theta_slot = theta_slot; e.ncat = ncat;
        e.graph = graph;
        e.launches = iqtree_gpu_lik_launch_count() - before;
        if (g_stats.launches_per_full == 0) g_stats.launches_per_full = e.launches;
        g_gpu.full_graphs.push_back(std::move(e));
        g_stats.full_captured++;
    } else {
        if (graph) iqtree_gpu_lik_graph_free(graph);
        // A recording error is not an evaluation error: the answer was produced in step 2.
        g_gpu.error_seen = false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------
// (1) computePartialLikelihood
// ------------------------------------------------------------------------------------------
void PhyloTree::computePartialLikelihoodGPU(TraversalInfo &info, size_t ptn_lower,
                                            size_t ptn_upper, int packet_id) {
    // NOTE ON THREADING: this is the one entry point that can be called concurrently. It is
    // installed as computePartialLikelihoodPointer, so when an ineligible model sends
    // computeLikelihoodBranch to the CPU SIMD kernel, that kernel's traversal reaches this
    // function from an `omp parallel for` over packets (phylokernelnew.h:1271-1277). The
    // counters are therefore atomic; nothing else here is new state.
    gpu_note_strict(this);
    const int reason = gpu_ineligible_reason(this);
    if (reason != GPU_OK) {
        gpu_delegate(GPU_EP_PARTIAL, reason);
        (this->*cpuComputePartialLikelihoodPointer)(info, ptn_lower, ptn_upper, packet_id);
        return;
    }
    // The context test MUST come before the packet test. If a device error has torn the
    // context down, the CPU kernel takes over and it genuinely needs every packet; dropping
    // packets != 0 first would hand it a partially computed node.
    if (!g_gpu.eng || g_gpu.tree != this) {
        gpu_delegate(GPU_EP_PARTIAL, GPU_R_CONTEXT_NOT_READY);
        (this->*cpuComputePartialLikelihoodPointer)(info, ptn_lower, ptn_upper, packet_id);
        return;
    }
    // (B) The kernel does every pattern of the node in one launch, so packet 0 owns the node
    // and the other packets must not re-run it. ptn_lower/ptn_upper are deliberately ignored.
    if (packet_id != 0) {
        g_stats.non_owning_packets.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // Committed to the device from here on: the node is either launched below or reported as a
    // device error, and a leaf has no partial to compute on either kernel.
    // While CAPTURING (STEP 2), this same node is being RE-issued only to record its launches
    // into a graph -- the real work already ran on the normal pass -- so it must not be counted
    // as a second GPU likelihood call.
    if (!g_gpu.capturing) gpu_count(GPU_EP_PARTIAL, GPU_OK);

    PhyloNeighbor *dad_branch = info.dad_branch;
    PhyloNode     *dad        = info.dad;
    PhyloNode     *node       = (PhyloNode*)dad_branch->node;
    if (node->isLeaf()) return;

    GpuSlot *out = gpu_slot(gpuSlotIndex(dad_branch));
    if (!out) { gpu_error("partial: output slot"); return; }

    // Collect children in IQ-TREE's neighbour order.
    std::vector<PhyloNeighbor*> child;
    child.reserve(node->degree());
    FOR_NEIGHBOR_IT(node, dad, it) child.push_back((PhyloNeighbor*)(*it));
    if (child.size() < 2) {
        // A degree-2 internal node only arises on rooted trees, which gpu_eligible() rejects;
        // the engine has no unary recursion, so refuse rather than index past the end.
        gpu_error("partial: internal node of degree < 3");
        return;
    }

    // Precompute echild (and, for leaf children, the collapsed partial_lh_leaf table) into one
    // slot per child. This is the device equivalent of computePartialInfo(). While CAPTURING for
    // a replayable graph, the length must come from the DEVICE buffer (branch_precompute_dev,
    // indexed by the topological branch id) so a later replay picks up fresh lengths; the normal
    // pass keeps the by-value length, byte-identical to the validated path.
    for (size_t k = 0; k < child.size(); k++) {
        int want_leaf = child[k]->node->isLeaf() ? 1 : 0;
        int rc_pre = g_gpu.capturing
            ? iqtree_gpu_lik_branch_precompute_dev(g_gpu.eng, (int)k, child[k]->id, want_leaf)
            : iqtree_gpu_lik_branch_precompute(g_gpu.eng, (int)k, child[k]->length, want_leaf);
        if (rc_pre != 0) {
            gpu_error("branch precompute");
            return;
        }
    }

    int rc = 0;
    if (child.size() > 2) {
        // Degree > 3. Children fold in one at a time with an underflow test after each, which
        // is IQ-TREE's ordering and is observable through the scale counters.
        std::vector<IqtreeGpuChild> ch(child.size());
        for (size_t k = 0; k < child.size(); k++) {
            PhyloNeighbor *nei = child[k];
            memset(&ch[k], 0, sizeof(IqtreeGpuChild));
            ch[k].echild_slot = (int)k;
            if (nei->node->isLeaf()) {
                ch[k].is_tip   = 1;
                ch[k].tip_slot = nei->node->id;
            } else {
                GpuSlot *s = gpu_slot_read(gpuSlotIndex(nei), "partial: child slot");
                if (!s) { gpu_error("partial: child slot"); return; }
                ch[k].d_partial = s->lh;
                ch[k].d_scale   = s->scale;
            }
        }
        rc = iqtree_gpu_lik_partial_multifurcating(g_gpu.eng, out->lh, out->scale,
                                                   (int)ch.size(), ch.data());
    } else {
        PhyloNeighbor *left = child[0], *right = child[1];
        int slot_left = 0, slot_right = 1;
        // IQ-TREE puts the leaf on the left when exactly one child is a leaf; the tip-internal
        // kernel takes the tip as its left argument, so keep the same convention.
        if (!left->node->isLeaf() && right->node->isLeaf()) {
            std::swap(left, right);
            std::swap(slot_left, slot_right);
        }
        if (left->node->isLeaf() && right->node->isLeaf()) {
            rc = iqtree_gpu_lik_partial_tip_tip(g_gpu.eng, out->lh, out->scale,
                                                slot_left, left->node->id,
                                                slot_right, right->node->id);
        } else if (left->node->isLeaf()) {
            GpuSlot *sr = gpu_slot_read(gpuSlotIndex(right), "partial: right slot");
            if (!sr) { gpu_error("partial: right slot"); return; }
            rc = iqtree_gpu_lik_partial_tip_internal(g_gpu.eng, out->lh, out->scale,
                                                     slot_left, left->node->id,
                                                     slot_right, sr->lh, sr->scale);
        } else {
            GpuSlot *sl = gpu_slot_read(gpuSlotIndex(left),  "partial: left slot");
            GpuSlot *sr = gpu_slot_read(gpuSlotIndex(right), "partial: right slot");
            if (!sl || !sr) { gpu_error("partial: child slots"); return; }
            rc = iqtree_gpu_lik_partial_internal_internal(g_gpu.eng, out->lh, out->scale,
                                                          slot_left, sl->lh, sl->scale,
                                                          slot_right, sr->lh, sr->scale);
        }
    }
    if (rc != 0) {
        gpu_error("partial likelihood");
        return;
    }
    // The slot now holds a partial computed under the CURRENT binding. Stamped only on success,
    // so a launch that failed leaves the slot stale and the first reader says so by name rather
    // than consuming whatever the kernel left behind.
    out->gen = g_gpu.gen;
}

// ------------------------------------------------------------------------------------------
// (2) computeLikelihoodBranch
// ------------------------------------------------------------------------------------------
double PhyloTree::computeLikelihoodBranchGPU(PhyloNeighbor *dad_branch, PhyloNode *dad,
                                             bool save_log_value) {
    gpu_note_strict(this);
    {
        const int reason = gpu_ineligible_reason(this);
        if (reason != GPU_OK) {
            gpu_delegate(GPU_EP_BRANCH, reason);
            gpuLeave();
            return (this->*cpuComputeLikelihoodBranchPointer)(dad_branch, dad, save_log_value);
        }
    }
    PhyloNode *node = (PhyloNode*)dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*)node->findNeighbor(dad);
    if (!central_partial_lh) initializeAllPartialLh();
    if (node->isLeaf()) {                       // mirror IQ-TREE's leaf swap
        std::swap(node, dad);
        std::swap(dad_branch, node_branch);
    }
    if (!gpuPrepare()) {
        gpu_delegate(GPU_EP_BRANCH, GPU_R_PREPARE_FAILED);
        return (this->*cpuComputeLikelihoodBranchPointer)(dad_branch, dad, save_log_value);
    }
    g_gpu.error_seen = false;

    gpuTraverse(node, dad);
    // The branch path uses its own theta slot so that theta_computed keeps meaning what it
    // means on the CPU, where computeLikelihoodBranch does not touch theta_all at all. The
    // partial sweep + theta is served by a replayed full-traversal graph when possible, else
    // computed per node (and, if full, captured for next time). See gpuReplayOrComputeSweep.
    gpuReplayOrComputeSweep(dad_branch, dad, GPU_THETA_BRANCH);
    if (g_gpu.error_seen) {
        gpu_delegate(GPU_EP_BRANCH, GPU_R_DEVICE_ERROR);
        gpuAbortToCpu();
        return (this->*cpuComputeLikelihoodBranchPointer)(dad_branch, dad, save_log_value);
    }

    const size_t orig_nptn = aln->size();
    const size_t ncat      = site_rate->getNRate();
    double *lh_cat = gpu_ptn_lh_cat_wanted ? g_gpu.h_lhcat.data() : nullptr;
    // Leave _pattern_lh on the device when it can be. It is ~1.2 MB at 156k patterns and this
    // call sits in the Newton branch-length loop, whose iterations nothing reads it from; the
    // readers that do exist ask for it through PhyloTree::gpuSyncPatternLh().
    const bool defer_ptn_lh = gpu_ptn_lh_deferrable(this);
    double tree_lh = 0.0;
    if (iqtree_gpu_lik_branch_logl_full(g_gpu.eng, dad_branch->length, &tree_lh,
                                        defer_ptn_lh ? nullptr : _pattern_lh,
                                        save_log_value ? nullptr : _pattern_scaling,
                                        lh_cat, save_log_value ? 1 : 0) != 0) {
        gpu_error("branch log-likelihood");
        gpu_delegate(GPU_EP_BRANCH, GPU_R_DEVICE_ERROR);
        gpuAbortToCpu();
        return (this->*cpuComputeLikelihoodBranchPointer)(dad_branch, dad, save_log_value);
    }
    // Record the debt only after the launch succeeded, so a failed call cannot leave a promise
    // against a device buffer that was never written.
    g_gpu.ptn_lh_pending = defer_ptn_lh;
    g_gpu.ptn_lh_host    = defer_ptn_lh ? _pattern_lh : nullptr;
    gpu_count(GPU_EP_BRANCH, GPU_OK);

    if (lh_cat) {
        // The device hands back category-major values; IQ-TREE stores _pattern_lh_cat
        // interleaved by vector_size (transformPatternLhCat() de-interleaves it afterwards),
        // so scatter into that layout here rather than teaching the kernel about SIMD lanes.
        const size_t nptn_pad = ((orig_nptn + vector_size - 1) / vector_size) * vector_size;
        for (size_t base = 0; base < nptn_pad; base += vector_size) {
            double *dst = _pattern_lh_cat + base * ncat;
            for (size_t c = 0; c < ncat; c++)
                for (size_t v = 0; v < vector_size; v++)
                    dst[c * vector_size + v] =
                        (base + v < orig_nptn) ? lh_cat[c * orig_nptn + base + v] : 0.0;
        }
    }

    // Same arbitrary repair IQ-TREE applies when some sites underflow completely. It READS and
    // then REWRITES _pattern_lh, so the deferred copy has to be paid first -- and, once the
    // repair has edited the host values, must not be paid again; gpu_flush_ptn_lh() clears the
    // debt, which is exactly that. Underflow is rare, so the transfer this forces costs
    // nothing on the paths the deferral exists for.
    if (!std::isfinite(tree_lh)) {
        gpu_flush_ptn_lh();
        tree_lh = 0.0;
        for (size_t ptn = 0; ptn < orig_nptn; ptn++) {
            double ptn_lh = _pattern_lh[ptn];
            if (!std::isfinite(_pattern_lh[ptn])) {
                if (save_log_value) {
                    _pattern_lh[ptn] = LOG_SCALING_THRESHOLD * 4;   // log(2^-1024)
                } else {
                    _pattern_lh[ptn] = 1.0;
                    _pattern_scaling[ptn] = LOG_SCALING_THRESHOLD * 4;
                }
                ptn_lh = LOG_SCALING_THRESHOLD * 4;
            }
            tree_lh += ptn_lh * ptn_freq[ptn];
        }
    }
    return tree_lh;
}

// ------------------------------------------------------------------------------------------
// STEP 2: Newton-loop graph replay (host-in-loop fallback)
// ------------------------------------------------------------------------------------------
// One captured graph each serves EVERY Newton iteration of EVERY branch, because the derivative
// and logL-only sequences read only engine-global buffers (theta_of(NEWTON), d_val, d_ptn_*,
// d_len): the per-branch theta content flows through the theta buffer and the candidate length
// through d_len. The graph is (re)captured lazily when its (gen, ncat, theta_slot) tag no longer
// matches. This removes the ~3 per-iteration kernel LAUNCHES; the host still reads df/ddf and
// computes the Newton step (that round-trip remains -- removing it is the on-device-Newton step).
static bool gpuEnsureNewtonGraph(GpuContext::SmallGraph &sg, bool is_derv, int ncat) {
    if (sg.graph && sg.gen == g_gpu.gen && sg.ncat == ncat && sg.theta_slot == GPU_THETA_NEWTON)
        return true;
    if (sg.graph) { iqtree_gpu_lik_graph_free(sg.graph); sg = GpuContext::SmallGraph(); }
    // Select NEWTON so the captured kernels bake theta_of(NEWTON) / bufscale_of(NEWTON).
    if (iqtree_gpu_lik_set_theta_slot(g_gpu.eng, GPU_THETA_NEWTON) != 0) {
        gpu_error("newton graph: select theta slot");
        return false;
    }
    if (iqtree_gpu_lik_graph_capture_begin(g_gpu.eng) != 0) return false;
    int rc = is_derv ? iqtree_gpu_lik_derivatives_dev_issue(g_gpu.eng)
                     : iqtree_gpu_lik_branch_logl_dev_issue(g_gpu.eng, 1);
    IqtreeGpuGraph *graph = nullptr;
    bool end_ok = (iqtree_gpu_lik_graph_capture_end(g_gpu.eng, &graph) == 0);
    if (rc == 0 && end_ok && graph) {
        sg.graph = graph; sg.gen = g_gpu.gen; sg.ncat = ncat; sg.theta_slot = GPU_THETA_NEWTON;
        return true;
    }
    if (graph) iqtree_gpu_lik_graph_free(graph);
    g_gpu.error_seen = false;   // a capture failure must not fail the evaluation; caller falls back
    return false;
}

static bool gpuDervReplay(int ncat, double len, double *lh, double *df, double *ddf) {
    if (!gpu_small_replay_enabled()) return false;   // forced per-launch derivative path
    if (!gpuEnsureNewtonGraph(g_gpu.derv_graph, true, ncat)) return false;
    if (iqtree_gpu_lik_set_eval_len(g_gpu.eng, len) != 0) { gpu_error("newton: set eval len"); return false; }
    if (iqtree_gpu_lik_graph_launch(g_gpu.eng, g_gpu.derv_graph.graph) != 0) { gpu_error("newton: derv replay"); return false; }
    if (iqtree_gpu_lik_derivatives_readback(g_gpu.eng, lh, df, ddf) != 0) { gpu_error("newton: derv readback"); return false; }
    g_stats.derv_replayed++;
    return true;
}

static bool gpuLoglReplay(int ncat, double len, double *logl) {
    if (!gpu_small_replay_enabled()) return false;   // forced per-launch logL path
    if (!gpuEnsureNewtonGraph(g_gpu.logl_graph, false, ncat)) return false;
    if (iqtree_gpu_lik_set_eval_len(g_gpu.eng, len) != 0) { gpu_error("newton: set eval len"); return false; }
    if (iqtree_gpu_lik_graph_launch(g_gpu.eng, g_gpu.logl_graph.graph) != 0) { gpu_error("newton: logl replay"); return false; }
    if (iqtree_gpu_lik_branch_logl_readback(g_gpu.eng, logl) != 0) { gpu_error("newton: logl readback"); return false; }
    g_stats.logl_replayed++;
    return true;
}

// ------------------------------------------------------------------------------------------
// STEP 3: on-device Newton loop
// ------------------------------------------------------------------------------------------
// The whole per-branch Newton optimisation runs on the GPU in one conditional-WHILE graph launch,
// removing the ~8 us host round-trip paid per iteration when the host loop drives the device.
// OPT-IN: selected with `--gpu-newton device` (Params::gpu_newton_device). The default,
// `--gpu-newton host`, keeps IQ-TREE's minimizeNewton on the host with device-evaluated
// derivatives -- the configuration of every run in the published benchmark. A run-constant
// Params flag rather than an environment variable, so that the choice is on the command line
// that the run record stores, and is echoed in the banner and the GPUSTATS line.
static inline bool gpu_ondevice_newton_enabled(const PhyloTree *tree) {
    return tree && tree->params && tree->params->gpu_newton_device;
}

static bool gpuEnsureNewtonLoopGraph(int ncat) {
    if (g_gpu.newton_gen == g_gpu.gen && g_gpu.newton_ncat == ncat)
        return true;   // already instantiated for this (gen, ncat)
    // Select NEWTON so the WHILE body's k_derivatives bakes theta_of(NEWTON)/bufscale_of(NEWTON).
    if (iqtree_gpu_lik_set_theta_slot(g_gpu.eng, GPU_THETA_NEWTON) != 0) {
        gpu_error("newton loop: select theta slot");
        return false;
    }
    if (iqtree_gpu_lik_newton_build(g_gpu.eng) != 0) {
        gpu_error("newton loop: build");
        return false;
    }
    g_gpu.newton_gen  = g_gpu.gen;
    g_gpu.newton_ncat = ncat;
    return true;
}

// Returns true if the branch was optimised on-device (optx set); false means "not done, use the
// host optimiser". Mirrors optimizeOneBranch's minimizeNewton call semantics exactly.
bool PhyloTree::gpuNewtonOptimize(double x1, double xguess, double x2, double xacc,
                                  int maxNRStep, double &optx) {
    if (!gpu_ondevice_newton_enabled(this)) return false;
    if (!g_gpu.eng || g_gpu.tree != this) return false;
    if (gpu_ineligible_reason(this) != GPU_OK) return false;
    if (!current_it || !current_it_back) return false;

    // Iteration 0 (minimizeNewton:428-433): evaluate at the clamped guess. This also computes
    // theta into the NEWTON slot (theta_computed false->true); theta is independent of the branch
    // being optimised, so the on-device loop reuses it for every candidate length. We reuse the
    // validated GPU derivative path here; its df/ddf are recomputed in the loop's first pass.
    double rts = xguess;
    if (rts < x1) rts = x1;
    if (rts > x2) rts = x2;
    current_it->length = rts;
    current_it_back->length = rts;
    double df0 = 0.0, ddf0 = 0.0;
    theta_computed = false;
    // Snapshot the on-GPU derivative count: if computeLikelihoodDervGPU delegated to the CPU
    // (e.g. gpuPrepare failed), theta was computed in HOST memory, NOT the device NEWTON slot, and
    // the on-device loop must NOT run against a stale device buffer. The count increments only on
    // the GPU branch (gpu_count(GPU_EP_DERV)), so an unchanged count means "ran on CPU -> fall back".
    long long derv_gpu_before = g_stats.on_gpu[GPU_EP_DERV].load(std::memory_order_relaxed);
    computeLikelihoodDervGPU(current_it, (PhyloNode*)current_it_back->node, &df0, &ddf0);
    if (g_gpu.error_seen || !theta_computed) return false;
    if (g_stats.on_gpu[GPU_EP_DERV].load(std::memory_order_relaxed) == derv_gpu_before)
        return false;   // the derivative (and thus theta) ran on the CPU; use the host optimiser

    const int ncat = (int)site_rate->getNRate();
    if (!gpuEnsureNewtonLoopGraph(ncat)) return false;

    double optx_dev = rts; int err = 0;
    if (iqtree_gpu_lik_newton_optimize(g_gpu.eng, x1, x2, xacc, maxNRStep, rts, &optx_dev, &err) != 0) {
        gpu_error("newton loop: optimize");
        return false;
    }
    if (err) return false;   // non-finite derivative on device -> host optimiser instead
    optx = optx_dev;
    // Leave the branch at optx so optimizeOneBranch's divergence-reset compares logL at optx.
    current_it->length = optx;
    current_it_back->length = optx;
    g_stats.newton_ondevice++;
    return true;
}

// ------------------------------------------------------------------------------------------
// (3) computeLikelihoodDerv -- analytic first and second derivative
// ------------------------------------------------------------------------------------------
void PhyloTree::computeLikelihoodDervGPU(PhyloNeighbor *dad_branch, PhyloNode *dad,
                                         double *df, double *ddf) {
    gpu_note_strict(this);
    {
        const int reason = gpu_ineligible_reason(this);
        if (reason != GPU_OK) {
            gpu_delegate(GPU_EP_DERV, reason);
            gpuLeave();
            (this->*cpuComputeLikelihoodDervPointer)(dad_branch, dad, df, ddf);
            return;
        }
    }
    PhyloNode *node = (PhyloNode*)dad_branch->node;
    PhyloNeighbor *node_branch = (PhyloNeighbor*)node->findNeighbor(dad);
    if (!central_partial_lh) initializeAllPartialLh();
    if (node->isLeaf()) {
        std::swap(node, dad);
        std::swap(dad_branch, node_branch);
    }
    const int branch_id = node_branch->id;

    if (!gpuPrepare()) {
        gpu_delegate(GPU_EP_DERV, GPU_R_PREPARE_FAILED);
        (this->*cpuComputeLikelihoodDervPointer)(dad_branch, dad, df, ddf);
        return;
    }

    // IQ-TREE calls computeTraversalInfo unconditionally here, but only computes the partials
    // and theta when theta_computed is false (computeLikelihoodBufferSIMD does both). Same
    // structure, same condition -- (D).
    g_gpu.error_seen = false;
    gpuTraverse(node, dad);
    if (!theta_computed) {
        // Full or subset partial sweep + theta; replayed from a cached full-traversal graph when
        // the identical traversal recurs (the first Newton derivative of each ModelFinder
        // candidate on a fixed topology is such a full traversal). Subsequent Newton iterations
        // take the else-branch: theta is already computed, so no traversal runs at all.
        gpuReplayOrComputeSweep(dad_branch, dad, GPU_THETA_NEWTON);
    } else if (iqtree_gpu_lik_set_theta_slot(g_gpu.eng, GPU_THETA_NEWTON) != 0) {
        gpu_error("select theta slot");
    }

    // G row wanted only for `--dating mcmctree`; a negative index makes the kernel skip the
    // store entirely, so a normal Newton iteration pays nothing for it.
    const bool want_g = g_gpu.gmat && G_matrix &&
                        branch_id >= 0 && branch_id < g_gpu.gmat_nbranch;
    double all_lh = 0.0, all_df = 0.0, all_ddf = 0.0;
    // Newton-loop replay: when no G row is wanted (every case except --dating mcmctree), the
    // 3-kernel derivative sequence is served by the cached singleton graph. Falls back to the
    // per-launch path on any capture/replay failure or when a G row IS wanted.
    bool derv_done = false;
    if (!g_gpu.error_seen && !want_g)
        derv_done = gpuDervReplay((int)site_rate->getNRate(), dad_branch->length,
                                  &all_lh, &all_df, &all_ddf);
    if (!derv_done && !g_gpu.error_seen &&
        iqtree_gpu_lik_derivatives_g(g_gpu.eng, dad_branch->length, want_g ? branch_id : -1,
                                     &all_lh, &all_df, &all_ddf) != 0) {
        gpu_error("derivatives");
    }
    if (g_gpu.error_seen) {
        gpu_delegate(GPU_EP_DERV, GPU_R_DEVICE_ERROR);
        gpuAbortToCpu();
        (this->*cpuComputeLikelihoodDervPointer)(dad_branch, dad, df, ddf);
        return;
    }

    // MCMCTree's export reads all three. G_matrix stays DEVICE-RESIDENT as well, so that the
    // Hessian can be formed there with one cuBLAS DSYRK; this copy exists because
    // printMCMCFileFormat() memmoves host rows for the partitioned and super-tree paths
    // (phyloanalysis.cpp:2945, :2961, :3002), and because it is the fallback whenever the
    // device-side path declines. The whole row including its zero padding is copied, which
    // also repairs stock IQ-TREE's uninitialised tail there.
    if (want_g) {
        if (iqtree_gpu_lik_gmatrix_download_row(g_gpu.eng, branch_id,
                                                G_matrix + (size_t)branch_id * g_gpu.gmat_stride) != 0) {
            gpu_error("download G row");
            gpu_delegate(GPU_EP_DERV, GPU_R_DEVICE_ERROR);
            gpuAbortToCpu();
            (this->*cpuComputeLikelihoodDervPointer)(dad_branch, dad, df, ddf);
            return;
        }
        if (!g_gpu.gmat_written[branch_id]) {
            g_gpu.gmat_written[branch_id] = 1;
            g_gpu.gmat_rows++;
        }
    }
    if (gradient_vector)  gradient_vector[branch_id]  = all_df;
    if (hessian_diagonal) hessian_diagonal[branch_id] = all_ddf;
    gpu_count(GPU_EP_DERV, GPU_OK);

    // (D) theta_computed is set here and ONLY here, exactly as in
    // computeLikelihoodDervGenericSIMD. It is never initialised in the constructor and never
    // cleared by clearAllPartialLH(); the five sites that clear it are IQ-TREE's own.
    theta_computed = true;

    *df  = all_df;
    *ddf = all_ddf;
    if (!std::isfinite(*df)) {
        getModel()->writeInfo(cout);
        getRate()->writeInfo(cout);
        if (!safe_numeric)
            outError("Numerical underflow (lh-derivative). Run again with the safe likelihood "
                     "kernel via `-safe` option");
        cout << "WARNING: Numerical underflow for lh-derivative" << endl;
        *df = *ddf = 0.0;
    }
}

// ------------------------------------------------------------------------------------------
// (4) computeLikelihoodFromBuffer -- reuses the theta left by computeLikelihoodDerv
// ------------------------------------------------------------------------------------------
double PhyloTree::computeLikelihoodFromBufferGPU() {
    gpu_note_strict(this);
    {
        const int reason = gpu_ineligible_reason(this);
        if (reason != GPU_OK) {
            gpu_delegate(GPU_EP_FROMBUFFER, reason);
            gpuLeave();
            return (this->*cpuComputeLikelihoodFromBufferPointer)();
        }
    }
    // Same precondition as the CPU kernel (ASSERT(theta_all && theta_computed)).
    ASSERT(theta_computed && g_gpu.eng && g_gpu.tree == this);
    ASSERT(current_it);

    g_gpu.error_seen = false;
    // Same deferral as the branch entry point, and this is the hotter of the two: every Newton
    // iteration of every optimizeOneBranch passes through here.
    const bool defer_ptn_lh = gpu_ptn_lh_deferrable(this);
    double tree_lh = 0.0;
    if (iqtree_gpu_lik_set_theta_slot(g_gpu.eng, GPU_THETA_NEWTON) != 0) {
        gpu_error("select theta slot");
    } else {
        // Newton-loop replay for the logL-only re-eval: only when the per-pattern copy is
        // deferred (the hot Newton case), because the replay leaves _pattern_lh on the device
        // exactly as the deferred path does. Otherwise take the full path that copies it.
        bool logl_done = false;
        if (defer_ptn_lh)
            logl_done = gpuLoglReplay((int)site_rate->getNRate(), current_it->length, &tree_lh);
        if (!logl_done &&
            iqtree_gpu_lik_branch_logl_full(g_gpu.eng, current_it->length, &tree_lh,
                                            defer_ptn_lh ? nullptr : _pattern_lh,
                                            nullptr, nullptr, 1) != 0)
            gpu_error("log-likelihood from buffer");
    }
    if (g_gpu.error_seen) {
        // There is no honest CPU fallback here: theta_all was never maintained on the host,
        // so the CPU kernel has nothing to read. Fail loudly instead of returning a number.
        gpu_delegate(GPU_EP_FROMBUFFER, GPU_R_DEVICE_ERROR);
        gpuAbortToCpu();
        outError("IQ-TREE3-GPU: device failure in computeLikelihoodFromBuffer; "
                 "re-run without -gpu");
    }
    g_gpu.ptn_lh_pending = defer_ptn_lh;
    g_gpu.ptn_lh_host    = defer_ptn_lh ? _pattern_lh : nullptr;
    gpu_count(GPU_EP_FROMBUFFER, GPU_OK);

    if (!safe_numeric && !std::isfinite(tree_lh))
        outError("Numerical underflow (lh-from-buffer). Run again with the safe likelihood "
                 "kernel via `-safe` option");
    if (!std::isfinite(tree_lh)) {
        // Reads and rewrites _pattern_lh; see the identical repair in computeLikelihoodBranchGPU.
        gpu_flush_ptn_lh();
        tree_lh = 0.0;
        for (size_t ptn = 0; ptn < aln->size(); ptn++) {
            if (!std::isfinite(_pattern_lh[ptn]))
                _pattern_lh[ptn] = LOG_SCALING_THRESHOLD * 4;
            tree_lh += _pattern_lh[ptn] * ptn_freq[ptn];
        }
    }
    return tree_lh;
}

// ------------------------------------------------------------------------------------------
// (5) MCMCTree Hessian, straight off the device
// ------------------------------------------------------------------------------------------
// H = G * diag(ptn_freq) * G^T, UNNEGATED and with its true diagonal -- i.e. exactly the
// intermediate at main/phyloanalysis.cpp:2855, before `hessian = (-1) * hessian` and before
// the diagonal is replaced by hessian_diagonal[]. The caller applies both, so that the two
// paths differ only in where the rank-k update happened.
//
// Rows are indexed by IQ-TREE branch id, NOT by the export's branch ORDER; printMCMCFileFormat
// permutes rows via branch_ids, and it has to apply the same permutation to both indices of
// the returned matrix.
//
// The preconditions are what make this safe to prefer over the host path. All of them are
// necessary: the context must still belong to this tree (a rebuild frees the device G), no
// device error may have been seen, the branch count must match, and EVERY row must have been
// written since this G was created. computeHessian() (main/timetree.cpp:376-389) walks all
// branches immediately before printHessian(), so the complete case is the normal one; anything
// less means part of G is still zero and the host copy -- which is filled from the same device
// rows -- is the honest thing to use.
bool PhyloTree::gpuMCMCHessian(double *hessian_out, int nbranch) {
    if (!hessian_out || nbranch <= 0) return false;
    if (g_gpu.failed || !g_gpu.eng || g_gpu.tree != this) return false;
    if (!g_gpu.gmat || nbranch != g_gpu.gmat_nbranch) return false;
    if (g_gpu.gmat_rows != g_gpu.gmat_nbranch) return false;
    if (iqtree_gpu_lik_gmatrix_hessian(g_gpu.eng, hessian_out) != 0) {
        gpu_error("MCMCTree Hessian (cuBLAS DSYRK)");
        return false;
    }
    return true;
}
