/*
 * gpu_kernels_test.cu -- standalone validation of the CUDA likelihood engine.
 *
 * Builds a tiny tree (6 taxa, 4 rate categories) on seeded random model matrices, runs every
 * kernel in gpu_kernels.cu, and compares against a scalar CPU reference written in this same
 * file. The reference is an independent transcription of the formulation documented in
 * gpu_kernels.h, so agreement is evidence about the kernels rather than about a shared helper.
 *
 * THE WHOLE SUITE RUNS TWICE, at both state counts the engine supports:
 *   - 20 states (amino acid), 100 patterns, 24 tip rows, STATE_UNKNOWN 23
 *   - 4 states (DNA), 183 patterns, 19 tip rows, STATE_UNKNOWN 18
 * Both the reference and the driver are templates on the state count, so the two runs cannot
 * drift apart -- which is the reason the DNA suite is not a copy of the protein one. 183 is
 * the real distinct-pattern count of benchmark/data/dna/dna_n04 and gives nine padding lanes out of
 * a 192-wide stride, the largest padding fraction available anywhere in this repository.
 *
 * Checked, to a relative tolerance of 1e-12 unless noted:
 *   - every row of tip_partial_lh is exercised in every taxon (the first NROWS patterns are
 *     forced onto rows 0..NROWS-1), so the DNA run covers all twelve IUPAC ambiguity rows and
 *     the four dense-but-unreachable duplicate rows as well as A/C/G/T and unknown
 *   - the STATE_UNKNOWN row of a collapsed leaf table is the literal 1.0 over the whole block
 *   - CROSS-SHAPE (4 states only): the dedicated 4-state k_partial / k_mf_child and the
 *     generic templated shape instantiated at 4 are both run against the same reference, so a
 *     disagreement localises immediately to the specialisation
 *   - echild and the collapsed tip (partial_lh_leaf) tables
 *   - partials for all four cases: tip-tip, tip-internal, internal-internal, multifurcating
 *   - power-of-two rescaling and the scale counters, in both NORM_LH and SAFE_LH modes
 *   - theta for an internal dad and for a leaf dad, including the SAFE_LH renormalisation
 *     to the minimum scale count and the hard-zeroing of categories more than one step below
 *   - branch log-likelihood
 *   - analytic df and ddf, additionally cross-checked against a central finite difference
 *     of the CPU log-likelihood (loose tolerance -- that check exists to prove the
 *     analytic formula really is the derivative, not to measure precision)
 *   - the MCMCTree G matrix: the per-pattern df/L row emitted by the derivative kernel, its
 *     zero padding out to the host row pitch, its invariance under a plain (non-G) derivative
 *     call, and the refusal to write a row when no G matrix is enabled
 *   - the MCMCTree Hessian H = G*diag(ptn_freq)*G^T formed on the device with cuBLAS DSYRK,
 *     compared ELEMENTWISE against a scalar triple loop (an orientation, pitch or
 *     column-scaling error yields a complete, plausible matrix, so a norm would hide it)
 *   - the rate-category CAPACITY: an arena allocated for 10 categories and first exercised
 *     at 10, then dropped to 4 with iqtree_gpu_lik_set_ncat() and re-run, must reproduce the
 *     4-category reference exactly -- i.e. nothing leaks in from the category planes the
 *     smaller run does not own
 *   - REBIND: one arena driven hard as "tree A" (different branch lengths, different rate
 *     model), then iqtree_gpu_lik_rebind()'d and re-run as "tree B" against the same references
 *     the freshly-created engine was validated against, with no buffer reallocated in between;
 *     plus the invalidation the rebind owes the caller (theta and buffer_scale zeroed, rate
 *     vectors marked stale so every kernel that reads them fails by name) and the two refusals
 *     (a rebind on top of an enabled G matrix, and a rebind after a category change)
 *
 * Build (needs no GPU to compile, needs one to run):
 *   nvcc -std=c++17 -arch=sm_89 -fmad=false gpu_kernels_test.cu gpu_kernels.cu \
 *        -lcublas -o gpu_kernels_test
 *   (-lcublas is required: the Hessian is one cublasDsyrk. nvcc does not link it by default.)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

#include "gpu_kernels.h"

using std::vector;
using std::max;


/* Same constants as the engine. */
static const double SCALING_THRESHOLD     = 0x1p-256;
static const int    SCALING_THRESHOLD_EXP = 256;
static const double LOG_SCALING_THRESHOLD = -256.0 * 0.69314718055994530942;

static int g_failures = 0;

/* ------------------------------------------------------------------------------------
 * Comparison helpers
 * ---------------------------------------------------------------------------------- */

static double rel_diff(double a, double b) {
    double d = fabs(a - b);
    double s = fmax(fabs(a), fabs(b));
    return (s > 0.0) ? d / s : d;
}

/* SCALE-AWARE array comparison.
 *
 * Partials are stored in the EIGEN basis, where entries are signed and frequently cancel: an
 * element can fall from O(1) inputs to O(1e-5), which amplifies its RELATIVE error by the same
 * factor even though the absolute error is still one or two ulps of the inputs. Judging such an
 * element by its own magnitude therefore measures cancellation, not kernel accuracy -- the GPU
 * and CPU legitimately sum in different orders (IQ-TREE pre-collapses tip children through
 * partial_lh_leaf with a chunked horizontal_add; the GPU does a flat 20-element dot product).
 *
 * So the error is normalised against the largest magnitude present in the array, which is the
 * scale the arithmetic was actually carried out at. This stays strict where it matters -- a genuine
 * kernel bug perturbs large elements and is caught immediately -- while not failing on cancelled
 * intermediates whose absolute error is at machine precision.
 *
 * NOTE: this deliberately does NOT just loosen `tol`. Loosening would blind the test to real
 * regressions in the large elements; normalising by the array scale keeps those as tight as ever.
 * The quantities the ALGORITHM consumes (logL, df, ddf) are scalars checked by check_scalar() at
 * plain relative tolerance, and they agree at ~1e-16 -- which is the real evidence that cancelled
 * intermediates do not propagate.
 */
static void check_arr(const char *name, const vector<double> &a, const vector<double> &b,
                      double tol) {
    if (a.size() != b.size()) {
        printf("  [FAIL] %-34s size mismatch (%zu vs %zu)\n", name, a.size(), b.size());
        g_failures++;
        return;
    }
    double scale = 0.0;
    for (size_t i = 0; i < b.size(); i++) scale = fmax(scale, fabs(b[i]));

    double worst = 0.0, worst_selfrel = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = fabs(a[i] - b[i]);
        double r = (scale > 0.0) ? d / scale : d;      // error relative to the array's own scale
        if (r > worst) { worst = r; wi = i; worst_selfrel = rel_diff(a[i], b[i]); }
    }
    if (worst <= tol) {
        printf("  [ ok ] %-34s max rel-to-scale %.3e\n", name, worst);
    } else {
        printf("  [FAIL] %-34s max rel-to-scale %.3e at %zu (gpu %.17g cpu %.17g, "
               "self-rel %.3e, array scale %.3e)\n",
               name, worst, wi, a[wi], b[wi], worst_selfrel, scale);
        g_failures++;
    }
}

static void check_scale(const char *name, const vector<unsigned char> &a,
                        const vector<unsigned char> &b) {
    if (a.size() != b.size()) {
        printf("  [FAIL] %-34s size mismatch\n", name);
        g_failures++;
        return;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            printf("  [FAIL] %-34s counter differs at %zu (gpu %u cpu %u)\n",
                   name, i, (unsigned)a[i], (unsigned)b[i]);
            g_failures++;
            return;
        }
    }
    printf("  [ ok ] %-34s all counters identical\n", name);
}

static void check_scalar(const char *name, double gpu, double cpu, double tol) {
    double r = rel_diff(gpu, cpu);
    if (r <= tol) {
        printf("  [ ok ] %-34s rel %.3e (gpu %.17g)\n", name, r, gpu);
    } else {
        printf("  [FAIL] %-34s rel %.3e (gpu %.17g cpu %.17g)\n", name, r, gpu, cpu);
        g_failures++;
    }
}

/* ------------------------------------------------------------------------------------
 * CPU reference
 * ---------------------------------------------------------------------------------- */

struct Ref {
    int ncat, npat, stride, block, ntip_rows, unknown;
    int safe;
    vector<double> eval, evec, inv_evec, rates, props, tip_lh;
    vector<double> ptn_freq, ptn_invar;   /* length stride, zero-padded */

    size_t pelems() const { return (size_t)block * stride; }
    size_t selems() const { return (size_t)stride * (safe ? ncat : 1); }
};

/* echild[(c*20+x)*20+i] = evec[x*20+i] * exp(eval[i] * (rate_c * len)) */
template <int NST>
static vector<double> ref_echild(const Ref &r, double len) {
    vector<double> e((size_t)r.ncat * NST * NST);
    for (int c = 0; c < r.ncat; c++)
        for (int x = 0; x < NST; x++)
            for (int i = 0; i < NST; i++)
                e[((size_t)c * NST + x) * NST + i] =
                    r.evec[x * NST + i] * exp(r.eval[i] * (r.rates[c] * len));
    return e;
}

/* leaf[state*block + c*20 + x] = SUM_i echild[(c*20+x)*20+i] * tip_lh[state*20+i],
   with the STATE_UNKNOWN row forced to exactly 1.0 across the whole block. */
template <int NST>
static vector<double> ref_leaf(const Ref &r, const vector<double> &ech) {
    vector<double> leaf((size_t)r.ntip_rows * r.block);
    for (int st = 0; st < r.ntip_rows; st++) {
        for (int b = 0; b < r.block; b++) {
            if (st == r.unknown) { leaf[(size_t)st * r.block + b] = 1.0; continue; }
            double acc = 0.0;
            for (int i = 0; i < NST; i++)
                acc += ech[(size_t)b * NST + i] * r.tip_lh[(size_t)st * NST + i];
            leaf[(size_t)st * r.block + b] = acc;
        }
    }
    return leaf;
}

/* IQ-TREE's rescale, NORM_LH: one counter per pattern, one test per pattern taken on the
   maximum over the whole block. No "max > 0" guard, deliberately. */
template <int NST>
static void ref_rescale(const Ref &r, vector<double> &L, vector<unsigned char> &scale) {
    if (!r.safe) {
        for (int p = 0; p < r.stride; p++) {
            double m = 0.0;
            for (int b = 0; b < r.block; b++)
                m = fmax(m, fabs(L[(size_t)b * r.stride + p]));
            if (m < SCALING_THRESHOLD && r.ptn_invar[p] == 0.0) {
                for (int b = 0; b < r.block; b++) {
                    size_t o = (size_t)b * r.stride + p;
                    L[o] = ldexp(L[o], SCALING_THRESHOLD_EXP);
                }
                scale[p] = (unsigned char)(scale[p] + 1);
            }
        }
    } else {
        for (int c = 0; c < r.ncat; c++) {
            for (int p = 0; p < r.stride; p++) {
                double m = 0.0;
                for (int x = 0; x < NST; x++)
                    m = fmax(m, fabs(L[((size_t)(c * NST + x)) * r.stride + p]));
                if (m < SCALING_THRESHOLD && r.ptn_invar[p] == 0.0) {
                    for (int x = 0; x < NST; x++) {
                        size_t o = ((size_t)(c * NST + x)) * r.stride + p;
                        L[o] = ldexp(L[o], SCALING_THRESHOLD_EXP);
                    }
                    scale[(size_t)c * r.stride + p]++;
                }
            }
        }
    }
}

/* One bifurcating partial-likelihood update. Any child may be a tip (leaf table plus a
   per-pattern state) or internal (echild matrices plus the child's eigen-basis partial). */
template <int NST>
static void ref_partial(const Ref &r,
                        bool ltip, const vector<double> &leafL, const vector<int> &stateL,
                        const vector<double> &echL, const vector<double> &inL,
                        bool rtip, const vector<double> &leafR, const vector<int> &stateR,
                        const vector<double> &echR, const vector<double> &inR,
                        vector<double> &out) {
    out.assign(r.pelems(), 0.0);
    vector<double> u(NST);
    for (int c = 0; c < r.ncat; c++) {
        for (int p = 0; p < r.stride; p++) {
            for (int x = 0; x < NST; x++) {
                double vL, vR;
                if (ltip) {
                    vL = leafL[(size_t)stateL[p] * r.block + c * NST + x];
                } else {
                    vL = 0.0;
                    for (int i = 0; i < NST; i++)
                        vL += echL[((size_t)(c * NST + x)) * NST + i]
                            * inL[((size_t)(c * NST + i)) * r.stride + p];
                }
                if (rtip) {
                    vR = leafR[(size_t)stateR[p] * r.block + c * NST + x];
                } else {
                    vR = 0.0;
                    for (int i = 0; i < NST; i++)
                        vR += echR[((size_t)(c * NST + x)) * NST + i]
                            * inR[((size_t)(c * NST + i)) * r.stride + p];
                }
                u[x] = vL * vR;
            }
            for (int k = 0; k < NST; k++) {
                double acc = 0.0;
                for (int x = 0; x < NST; x++) acc += r.inv_evec[k * NST + x] * u[x];
                out[((size_t)(c * NST + k)) * r.stride + p] = acc;
            }
        }
    }
}

/* theta[i] = Ltilde_node[i] * Ltilde_dad[i]; then the scale renormalisation. */
template <int NST>
static void ref_theta(const Ref &r, const vector<double> &node, const vector<double> &dad,
                      const unsigned char *snode, const unsigned char *sdad,
                      vector<double> &theta, vector<double> &bufscale) {
    theta.resize(r.pelems());
    for (size_t i = 0; i < r.pelems(); i++) theta[i] = node[i] * dad[i];
    bufscale.assign(r.stride, 0.0);

    if (!r.safe) {
        for (int p = 0; p < r.stride; p++) {
            int s = sdad[p] + (snode ? snode[p] : 0);
            bufscale[p] = (double)s * LOG_SCALING_THRESHOLD;
        }
    } else {
        for (int p = 0; p < r.stride; p++) {
            int mn = 1 << 30;
            for (int c = 0; c < r.ncat; c++) {
                int s = sdad[(size_t)c * r.stride + p]
                      + (snode ? snode[(size_t)c * r.stride + p] : 0);
                if (s < mn) mn = s;
            }
            for (int c = 0; c < r.ncat; c++) {
                int s = sdad[(size_t)c * r.stride + p]
                      + (snode ? snode[(size_t)c * r.stride + p] : 0);
                if (s == mn + 1) {
                    for (int x = 0; x < NST; x++)
                        theta[((size_t)(c * NST + x)) * r.stride + p] *= SCALING_THRESHOLD;
                } else if (s > mn + 1) {
                    for (int x = 0; x < NST; x++)
                        theta[((size_t)(c * NST + x)) * r.stride + p] = 0.0;
                }
            }
            bufscale[p] = (double)mn * LOG_SCALING_THRESHOLD;
        }
    }
}

/* Branch log-likelihood: val0 uses exp(eval[i] * (rate_c*len)), IQ-TREE's ordering in
   computeLikelihoodFromBuffer. */
template <int NST>
static double ref_branch_logl(const Ref &r, const vector<double> &theta,
                              const vector<double> &bufscale, double len) {
    vector<double> val0(r.block);
    for (int c = 0; c < r.ncat; c++)
        for (int i = 0; i < NST; i++)
            val0[c * NST + i] = exp(r.eval[i] * (r.rates[c] * len)) * r.props[c];

    double tot = 0.0;
    for (int p = 0; p < r.npat; p++) {
        double lh = 0.0;
        for (int b = 0; b < r.block; b++) lh += val0[b] * theta[(size_t)b * r.stride + p];
        lh = fabs(lh) + r.ptn_invar[p];
        tot += r.ptn_freq[p] * (log(fabs(lh)) + bufscale[p]);
    }
    return tot;
}

/* Analytic derivatives: cof = eval*rate, val0 = exp(cof*t)*prop, val1 = cof*val0,
   val2 = cof*val1. Never finite differences. */
template <int NST>
static void ref_derivatives(const Ref &r, const vector<double> &theta,
                            const vector<double> &bufscale, double len,
                            double *logl, double *df, double *ddf) {
    vector<double> v0(r.block), v1(r.block), v2(r.block);
    for (int c = 0; c < r.ncat; c++)
        for (int i = 0; i < NST; i++) {
            double cof = r.eval[i] * r.rates[c];
            double lhval = exp(cof * len) * r.props[c];
            double dfval = cof * lhval;
            v0[c * NST + i] = lhval;
            v1[c * NST + i] = dfval;
            v2[c * NST + i] = cof * dfval;
        }
    double a_lh = 0.0, a_df = 0.0, a_ddf = 0.0;
    for (int p = 0; p < r.npat; p++) {
        double lh = 0.0, d1 = 0.0, d2 = 0.0;
        for (int b = 0; b < r.block; b++) {
            double t = theta[(size_t)b * r.stride + p];
            lh += v0[b] * t;
            d1 += v1[b] * t;
            d2 += v2[b] * t;
        }
        double L = fabs(lh) + r.ptn_invar[p];
        double f = r.ptn_freq[p];
        double dfr = d1 / L, ddfr = d2 / L;
        a_lh  += f * (log(L) + bufscale[p]);
        a_df  += f * dfr;
        a_ddf += f * (ddfr - dfr * dfr);
    }
    *logl = a_lh; *df = a_df; *ddf = a_ddf;
}

/* One row of the MCMCTree G matrix: G[i][p] = (dL_p/dt_i)/L_p, i.e. IQ-TREE's
   `df_ptn *= inv_lh_ptn` immediately before store_a(&G_matrix[g_index+ptn])
   (phylokernelnew.h:2526-2532). The row is gstride long and everything past npat must stay
   exactly 0.0 -- ptn_freq is zero there, so the rank-k update below is unaffected either way,
   but an uninitialised tail would turn the product into NaN. */
template <int NST>
static void ref_gmatrix_row(const Ref &r, const vector<double> &theta, double len,
                            int gstride, vector<double> &row) {
    vector<double> v0(r.block), v1(r.block);
    for (int c = 0; c < r.ncat; c++)
        for (int i = 0; i < NST; i++) {
            double cof = r.eval[i] * r.rates[c];
            v0[c * NST + i] = exp(cof * len) * r.props[c];
            v1[c * NST + i] = cof * v0[c * NST + i];
        }
    row.assign(gstride, 0.0);
    for (int p = 0; p < r.npat; p++) {
        double lh = 0.0, d1 = 0.0;
        for (int b = 0; b < r.block; b++) {
            double t = theta[(size_t)b * r.stride + p];
            lh += v0[b] * t;
            d1 += v1[b] * t;
        }
        row[p] = d1 / (fabs(lh) + r.ptn_invar[p]);
    }
}

/* H = G * diag(ptn_freq) * G^T, the unnegated intermediate of
   main/phyloanalysis.cpp:2853-2855. Written as the plain triple loop on purpose: the point of
   the comparison is to check the GPU's DSYRK orientation and column scaling against something
   that shares no code with it. */
template <int NST>
static void ref_gmatrix_hessian(const Ref &r, const vector< vector<double> > &rows,
                                vector<double> &h) {
    const size_t n = rows.size();
    h.assign(n * n, 0.0);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (int p = 0; p < r.npat; p++) s += r.ptn_freq[p] * rows[i][p] * rows[j][p];
            h[i * n + j] = s;
        }
}

/* Log-likelihood in the derivative parenthesisation, used only for the finite-difference
   cross-check of df/ddf. */
template <int NST>
static double ref_logl_derv_form(const Ref &r, const vector<double> &theta,
                                 const vector<double> &bufscale, double len) {
    double tot = 0.0;
    for (int p = 0; p < r.npat; p++) {
        double lh = 0.0;
        for (int c = 0; c < r.ncat; c++)
            for (int i = 0; i < NST; i++) {
                double cof = r.eval[i] * r.rates[c];
                lh += exp(cof * len) * r.props[c] * theta[((size_t)(c * NST + i)) * r.stride + p];
            }
        double L = fabs(lh) + r.ptn_invar[p];
        tot += r.ptn_freq[p] * (log(L) + bufscale[p]);
    }
    return tot;
}

/* ------------------------------------------------------------------------------------
 * Test driver
 * ---------------------------------------------------------------------------------- */

static void die(const char *what) {
    printf("FATAL %s: %s\n", what, iqtree_gpu_lik_last_error());
    exit(2);
}

/* A faithful HOST port of Optimization::minimizeNewton (utils/optimization.cpp:422-499) that uses
 * the engine's by-value derivative kernels as computeFuncDerv (negating, as phylotree.cpp:2606-2607
 * does). The on-device k_newton_step must trace the SAME sequence of candidate lengths and converge
 * to the SAME value, because both read df/ddf from the same reduction kernels and apply the same
 * scalar logic. This is the reference the on-device loop is checked against. */
static double host_minimize_newton(IqtreeGpuLik *ctx, double x1, double xguess, double x2,
                                   double xacc, int maxNRStep, int *err) {
    *err = 0;
    double f, df, l, temp;
    double rts = xguess;
    if (rts < x1) rts = x1;
    if (rts > x2) rts = x2;
    if (iqtree_gpu_lik_derivatives(ctx, rts, &l, &f, &df)) die("host_newton derivatives 0");
    f = -f; df = -df;
    if (!isfinite(f) || !isfinite(df)) { *err = 1; return rts; }
    if (df >= 0.0 && fabs(f) < xacc) return rts;
    double xl, xh;
    if (f < 0.0) { xl = rts; xh = x2; } else { xh = rts; xl = x1; }
    double dx, dxold; dx = dxold = fabs(xh - xl);
    double rts_old;
    for (int j = 1; j <= maxNRStep; j++) {
        rts_old = rts;
        if (df <= 0.0 || (((rts - xh) * df - f) * ((rts - xl) * df - f) >= 0.0)) {
            dxold = dx; dx = 0.5 * (xh - xl); rts = xl + dx;
            if (xl == rts) return rts;
        } else {
            dxold = dx; dx = f / df; temp = rts; rts -= dx;
            if (temp == rts) return rts;
        }
        if (fabs(dx) < xacc || j == maxNRStep) return rts_old;
        if (iqtree_gpu_lik_derivatives(ctx, rts, &l, &f, &df)) die("host_newton derivatives j");
        f = -f; df = -df;
        if (!isfinite(f) || !isfinite(df)) { *err = 1; return rts; }
        if (df > 0.0 && fabs(f) < xacc) return rts;
        if (f < 0.0) xl = rts; else if (f > 0.0) xh = rts;
    }
    return 0.0;   // minimizeNewton nrerror's here; unreachable within maxNRStep for our data
}

/* ONE SUITE, TWO STATE COUNTS.
 *
 * The whole of what used to be main() is a template on NST, run at <20> and at <4>. That is
 * not tidiness: the point of the suite is that the scalar CPU reference above is an
 * INDEPENDENT transcription of the formulation in gpu_kernels.h, so agreement is evidence
 * about the kernels. Copying the suite for DNA would have produced a second reference that
 * could drift from the first, which is precisely the failure mode that a single templated
 * reference makes impossible.
 *
 *   NROWS / UNKNOWN are the two data-type constants: (24, 23) for protein, (19, 18) for DNA.
 *   NPAT is a parameter so that the DNA run can use 183 -- the real distinct-pattern count of
 *   the repository's dna_n04 dataset -- which gives stride 192 and NINE padding lanes, a 4.7%
 *   padding fraction that no protein case comes close to. Padding bugs show up there.
 */
template <int NST>
static void run_suite(const char *label, int NPAT, int NROWS, int UNKNOWN) {
    printf("\n================ %s (%d states, %d patterns, %d tip rows) ================\n",
           label, NST, NPAT, NROWS);

    const int NCAT   = 4;
    /* The capacity the ncat-reuse section allocates for. 10 is IQ-TREE's default
       params.max_rate_cats, i.e. the largest +R{k} ModelFinder will test. */
    const int NCAT_CAP = 10;
    const int NTIP   = 6;

    std::mt19937_64 rng(20260721ULL);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    std::uniform_real_distribution<double> P01(0.0, 1.0);

    Ref r;
    r.ncat = NCAT; r.npat = NPAT; r.block = NCAT * NST;
    r.stride = ((NPAT + 31) / 32) * 32;
    r.ntip_rows = NROWS; r.unknown = UNKNOWN; r.safe = 0;

    /* Random but well-conditioned model matrices. The kernels treat eval/evec/inv_evec as
       opaque arrays, so they need not form a genuine eigendecomposition; keeping them
       close to the identity simply keeps every intermediate positive and O(1), which is
       what makes a 1e-12 relative comparison meaningful. */
    r.eval.resize(NST);
    for (int i = 0; i < NST; i++) r.eval[i] = -(0.05 + 0.9 * P01(rng));
    r.eval[0] = 0.0;                       /* the stationary eigenvalue */

    r.evec.resize(NST * NST);
    r.inv_evec.resize(NST * NST);
    for (int a = 0; a < NST; a++)
        for (int b = 0; b < NST; b++) {
            r.evec[a * NST + b]     = (a == b ? 1.0 : 0.0) + 0.05 * U(rng);
            r.inv_evec[a * NST + b] = (a == b ? 1.0 : 0.0) + 0.05 * U(rng);
        }

    r.rates.resize(NCAT); r.props.resize(NCAT);
    double psum = 0.0;
    for (int c = 0; c < NCAT; c++) { r.rates[c] = 0.3 + 0.6 * c; r.props[c] = 0.2 + 0.2 * P01(rng); psum += r.props[c]; }
    /* props already carry the (1 - p_invar) factor, so normalise to 0.9 with p_invar 0.1. */
    for (int c = 0; c < NCAT; c++) r.props[c] = r.props[c] / psum * 0.9;

    r.tip_lh.resize((size_t)NROWS * NST);
    for (size_t i = 0; i < r.tip_lh.size(); i++) r.tip_lh[i] = 0.4 + 0.6 * P01(rng);

    r.ptn_freq.assign(r.stride, 0.0);
    r.ptn_invar.assign(r.stride, 0.0);
    for (int p = 0; p < NPAT; p++) {
        r.ptn_freq[p] = 1.0 + (double)(p % 5);
        if (p % 7 == 0) r.ptn_invar[p] = 1e-3 * (1.0 + P01(rng));  /* constant sites */
    }

    /* Tip state columns, padded with STATE_UNKNOWN exactly as the engine pads them.
     *
     * The first NROWS patterns of every taxon are forced onto rows 0, 1, ... NROWS-1 in turn,
     * so EVERY row of tip_partial_lh is exercised in every taxon rather than merely likely to
     * be. That matters most for DNA, where rows 4..17 are the IUPAC ambiguity codes and rows
     * 4, 5, 7 and 11 are the dense-but-unreachable duplicates of A, C, G and T -- all of which
     * the engine must carry and index correctly even though a FASTA file cannot produce the
     * last four. Row UNKNOWN in particular must collapse to a leaf table of exactly 1.0 across
     * the whole block, which is checked as a literal below. The remaining patterns stay
     * random, so the mix of rows within a warp is still irregular. */
    vector< vector<int> > states(NTIP), states_full(NTIP);
    for (int t = 0; t < NTIP; t++) {
        states[t].resize(NPAT);
        states_full[t].assign(r.stride, UNKNOWN);
        for (int p = 0; p < NPAT; p++) {
            int s = (int)(P01(rng) * NROWS);
            if (s >= NROWS) s = NROWS - 1;
            if (p < NROWS) s = p;              /* deterministic coverage of every row */
            states[t][p] = s;
            states_full[t][p] = s;
        }
    }

    if (NPAT < NROWS) {
        printf("  [FAIL] NPAT < NROWS: every-row coverage is impossible\n");
        g_failures++;
    }

    /* ---------------- engine ---------------- */
    IqtreeGpuLikConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.nstates = NST; cfg.ncat = NCAT; cfg.npat = NPAT;
    cfg.num_tip_states = NROWS; cfg.state_unknown = UNKNOWN;
    cfg.safe_numeric = 0; cfg.echild_slots = 8; cfg.tip_slots = NTIP; cfg.device = 0;

    IqtreeGpuLik *ctx = NULL;
    if (iqtree_gpu_lik_create(&ctx, &cfg)) die("create");
    if (iqtree_gpu_lik_pattern_stride(ctx) != r.stride) die("stride mismatch");
    if (iqtree_gpu_lik_nstates(ctx) != NST) die("nstates mismatch");

    if (iqtree_gpu_lik_set_eigen(ctx, r.eval.data(), r.evec.data(), r.inv_evec.data())) die("set_eigen");
    if (iqtree_gpu_lik_set_rates(ctx, r.rates.data(), r.props.data())) die("set_rates");
    if (iqtree_gpu_lik_set_tip_partials(ctx, r.tip_lh.data())) die("set_tip_partials");
    if (iqtree_gpu_lik_set_pattern_data(ctx, r.ptn_freq.data(), r.ptn_invar.data())) die("set_pattern_data");
    for (int t = 0; t < NTIP; t++)
        if (iqtree_gpu_lik_set_tip_states(ctx, t, states[t].data())) die("set_tip_states");

    /* Branch lengths, one echild slot per branch of the little tree:
         slot 0..5 -> pendant branches of taxa 0..5
         slot 6    -> internal branch n5 -> n7
         slot 7    -> internal branch n6 -> n8                                    */
    double blen[8] = { 0.11, 0.23, 0.07, 0.31, 0.17, 0.29, 0.13, 0.19 };
    for (int s = 0; s < 8; s++)
        if (iqtree_gpu_lik_branch_precompute(ctx, s, blen[s], /*want_leaf=*/1)) die("branch_precompute");

    vector< vector<double> > ech(8), leaf(8);
    for (int s = 0; s < 8; s++) {
        ech[s]  = ref_echild<NST>(r, blen[s]);
        leaf[s] = ref_leaf<NST>(r, ech[s]);
    }

    /* The STATE_UNKNOWN row of a collapsed leaf table is the LITERAL 1.0 across the whole
       block, not a computed value -- IQ-TREE writes it that way (phylokernelnew.h:997-1001)
       and the engine copies it (k_leaf_partials). Asserted here as an exact comparison to 1.0
       so that the property is pinned independently of the partial comparisons that carry it
       to the GPU (every pattern whose tip state is UNKNOWN reads this row, and the first
       NROWS patterns of every taxon include exactly one such pattern by construction). */
    {
        int bad = 0;
        for (int s = 0; s < 8; s++)
            for (int b = 0; b < r.block; b++)
                if (leaf[s][(size_t)UNKNOWN * r.block + b] != 1.0) bad++;
        if (bad) {
            printf("  [FAIL] %-34s %d entries of the STATE_UNKNOWN row are not exactly 1.0\n",
                   "leaf table unknown row", bad);
            g_failures++;
        } else {
            printf("  [ ok ] %-34s row %d is exactly 1.0 over all %d block entries\n",
                   "leaf table unknown row", UNKNOWN, r.block);
        }
    }

    const size_t PE = r.pelems();
    const size_t SE = r.selems();
    vector<double> gpu(PE), cpu(PE);
    vector<unsigned char> gscale(SE), cscale(SE);
    const vector<double> nil;
    const vector<int> nili;

    printf("\n--- NORM_LH (default amino-acid path) ---\n");

    /* ---- case 1: tip-tip cherry n5 = (t0, t1). No scaling, counters zeroed. ---- */
    double *d_n5 = iqtree_gpu_lik_alloc_partial(ctx);
    unsigned char *s_n5 = iqtree_gpu_lik_alloc_scale(ctx);
    if (!d_n5 || !s_n5) die("alloc");
    if (iqtree_gpu_lik_partial_tip_tip(ctx, d_n5, s_n5, 0, 0, 1, 1)) die("tip_tip");
    if (iqtree_gpu_lik_sync(ctx)) die("sync");

    vector<double> ref_n5;
    ref_partial<NST>(r, true, leaf[0], states_full[0], nil, nil,
                   true, leaf[1], states_full[1], nil, nil, ref_n5);
    vector<unsigned char> ref_s5(SE, 0);

    if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), d_n5)) die("download");
    if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), s_n5)) die("download scale");
    check_arr("partial tip-tip", gpu, ref_n5, 1e-12);
    check_scale("scale tip-tip (all zero)", gscale, ref_s5);

    /* ---- case 2: tip-internal n7 = (t2, n5) ---- */
    double *d_n7 = iqtree_gpu_lik_alloc_partial(ctx);
    unsigned char *s_n7 = iqtree_gpu_lik_alloc_scale(ctx);
    if (iqtree_gpu_lik_partial_tip_internal(ctx, d_n7, s_n7, 2, 2, 6, d_n5, s_n5)) die("tip_internal");
    if (iqtree_gpu_lik_sync(ctx)) die("sync");

    vector<double> ref_n7;
    ref_partial<NST>(r, true,  leaf[2], states_full[2], nil,    nil,
                   false, nil,     nili,           ech[6], ref_n5, ref_n7);
    vector<unsigned char> ref_s7 = ref_s5;
    ref_rescale<NST>(r, ref_n7, ref_s7);

    if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), d_n7)) die("download");
    if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), s_n7)) die("download scale");
    check_arr("partial tip-internal", gpu, ref_n7, 1e-12);
    check_scale("scale tip-internal", gscale, ref_s7);

    /* ---- case 3: the other cherry n6 = (t3, t4), then internal-internal n8 = (n5, n6) ---- */
    double *d_n6 = iqtree_gpu_lik_alloc_partial(ctx);
    unsigned char *s_n6 = iqtree_gpu_lik_alloc_scale(ctx);
    if (iqtree_gpu_lik_partial_tip_tip(ctx, d_n6, s_n6, 3, 3, 4, 4)) die("tip_tip 2");
    vector<double> ref_n6;
    ref_partial<NST>(r, true, leaf[3], states_full[3], nil, nil,
                   true, leaf[4], states_full[4], nil, nil, ref_n6);
    vector<unsigned char> ref_s6(SE, 0);

    double *d_n8 = iqtree_gpu_lik_alloc_partial(ctx);
    unsigned char *s_n8 = iqtree_gpu_lik_alloc_scale(ctx);
    if (iqtree_gpu_lik_partial_internal_internal(ctx, d_n8, s_n8, 6, d_n5, s_n5,
                                                 7, d_n6, s_n6)) die("internal_internal");
    if (iqtree_gpu_lik_sync(ctx)) die("sync");

    vector<double> ref_n8;
    ref_partial<NST>(r, false, nil, nili, ech[6], ref_n5,
                   false, nil, nili, ech[7], ref_n6, ref_n8);
    vector<unsigned char> ref_s8(SE);
    for (size_t i = 0; i < SE; i++) ref_s8[i] = (unsigned char)(ref_s5[i] + ref_s6[i]);
    ref_rescale<NST>(r, ref_n8, ref_s8);

    if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), d_n8)) die("download");
    if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), s_n8)) die("download scale");
    check_arr("partial internal-internal", gpu, ref_n8, 1e-12);
    check_scale("scale internal-internal", gscale, ref_s8);

    /* ---- CUDA-graph capture/replay equivalence (STEP 2) ----
     * The launch-overhead fix replays a captured graph in place of re-issuing a traversal's
     * launches. A replay must be EQUIVALENT BY CONSTRUCTION -- same kernels, same order, same
     * arguments -- so this runs the SAME four-node partial sweep two ways and requires the
     * final device buffers to be BIT-IDENTICAL (tol exactly 0.0, not 1e-12): the graph must not
     * perturb a single ulp.
     *
     *   arm A (reference): issue the four partial ops normally into slot set A, sync, download.
     *   arm B (captured):  alloc slot set B; capture the identical op sequence into a graph
     *                      (capture RECORDS, it does not execute); replay the graph; sync;
     *                      download. Then replay a SECOND time to prove replay is idempotent.
     *
     * The captured graph is self-contained: it recomputes its two cherries (n5b,n6b) and both
     * internal nodes (n7b,n8b) from the tip/echild data already on the device, so the data
     * dependencies n7b<-n5b and n8b<-{n5b,n6b} are recorded as in-stream ordering and replay
     * reproduces them. */
    {
        /* Arm A -- uncaptured reference, into an independent slot set. */
        double        *a_n5 = iqtree_gpu_lik_alloc_partial(ctx), *a_n6 = iqtree_gpu_lik_alloc_partial(ctx);
        double        *a_n7 = iqtree_gpu_lik_alloc_partial(ctx), *a_n8 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *as5  = iqtree_gpu_lik_alloc_scale(ctx),   *as6  = iqtree_gpu_lik_alloc_scale(ctx);
        unsigned char *as7  = iqtree_gpu_lik_alloc_scale(ctx),   *as8  = iqtree_gpu_lik_alloc_scale(ctx);
        if (!a_n5 || !a_n6 || !a_n7 || !a_n8 || !as5 || !as6 || !as7 || !as8) die("cap: alloc A");
        if (iqtree_gpu_lik_partial_tip_tip(ctx, a_n5, as5, 0, 0, 1, 1)) die("cap A tip_tip n5");
        if (iqtree_gpu_lik_partial_tip_tip(ctx, a_n6, as6, 3, 3, 4, 4)) die("cap A tip_tip n6");
        if (iqtree_gpu_lik_partial_tip_internal(ctx, a_n7, as7, 2, 2, 6, a_n5, as5)) die("cap A tip_internal n7");
        if (iqtree_gpu_lik_partial_internal_internal(ctx, a_n8, as8, 6, a_n5, as5, 7, a_n6, as6))
            die("cap A internal_internal n8");
        if (iqtree_gpu_lik_sync(ctx)) die("cap A sync");
        vector<double>        uncap_lh(PE);  vector<unsigned char> uncap_sc(SE);
        if (iqtree_gpu_lik_download_partial(ctx, uncap_lh.data(), a_n8)) die("cap A download lh");
        if (iqtree_gpu_lik_download_scale(ctx, uncap_sc.data(), as8)) die("cap A download scale");

        /* Arm B -- capture the identical sequence into a graph, then replay it. */
        double        *b_n5 = iqtree_gpu_lik_alloc_partial(ctx), *b_n6 = iqtree_gpu_lik_alloc_partial(ctx);
        double        *b_n7 = iqtree_gpu_lik_alloc_partial(ctx), *b_n8 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *bs5  = iqtree_gpu_lik_alloc_scale(ctx),   *bs6  = iqtree_gpu_lik_alloc_scale(ctx);
        unsigned char *bs7  = iqtree_gpu_lik_alloc_scale(ctx),   *bs8  = iqtree_gpu_lik_alloc_scale(ctx);
        if (!b_n5 || !b_n6 || !b_n7 || !b_n8 || !bs5 || !bs6 || !bs7 || !bs8) die("cap: alloc B");

        long long replays_before = iqtree_gpu_lik_graph_replay_count();

        IqtreeGpuGraph *graph = NULL;
        if (iqtree_gpu_lik_graph_capture_begin(ctx)) die("graph_capture_begin");
        /* These record onto the capture stream; they do NOT execute here. No sync/alloc/blocking
           copy may appear between begin and end -- the partial wrappers issue none. */
        if (iqtree_gpu_lik_partial_tip_tip(ctx, b_n5, bs5, 0, 0, 1, 1)) die("cap B tip_tip n5");
        if (iqtree_gpu_lik_partial_tip_tip(ctx, b_n6, bs6, 3, 3, 4, 4)) die("cap B tip_tip n6");
        if (iqtree_gpu_lik_partial_tip_internal(ctx, b_n7, bs7, 2, 2, 6, b_n5, bs5)) die("cap B tip_internal n7");
        if (iqtree_gpu_lik_partial_internal_internal(ctx, b_n8, bs8, 6, b_n5, bs5, 7, b_n6, bs6))
            die("cap B internal_internal n8");
        if (iqtree_gpu_lik_graph_capture_end(ctx, &graph) || !graph) die("graph_capture_end");

        if (iqtree_gpu_lik_graph_launch(ctx, graph)) die("graph_launch #1");
        if (iqtree_gpu_lik_sync(ctx)) die("cap B sync #1");
        vector<double>        cap_lh(PE);  vector<unsigned char> cap_sc(SE);
        if (iqtree_gpu_lik_download_partial(ctx, cap_lh.data(), b_n8)) die("cap B download lh");
        if (iqtree_gpu_lik_download_scale(ctx, cap_sc.data(), bs8)) die("cap B download scale");

        check_arr("graph replay partial (bit-exact)", cap_lh, uncap_lh, 0.0);
        check_scale("graph replay scale (bit-exact)", cap_sc, uncap_sc);

        /* A second replay must reproduce the first exactly -- an instantiated graph is
           re-runnable without re-capture, which is the whole point. */
        if (iqtree_gpu_lik_graph_launch(ctx, graph)) die("graph_launch #2");
        if (iqtree_gpu_lik_sync(ctx)) die("cap B sync #2");
        vector<double> cap_lh2(PE);
        if (iqtree_gpu_lik_download_partial(ctx, cap_lh2.data(), b_n8)) die("cap B download lh #2");
        check_arr("graph replay idempotent (bit-exact)", cap_lh2, cap_lh, 0.0);

        long long replays_after = iqtree_gpu_lik_graph_replay_count();
        if (replays_after - replays_before == 2) {
            printf("  [ ok ] %-34s counted %lld replays\n", "graph replay accounting",
                   replays_after - replays_before);
        } else {
            printf("  [FAIL] %-34s expected 2 replays, counted %lld\n", "graph replay accounting",
                   replays_after - replays_before);
            g_failures++;
        }

        iqtree_gpu_lik_graph_free(graph);
        iqtree_gpu_lik_free(ctx, a_n5); iqtree_gpu_lik_free(ctx, a_n6);
        iqtree_gpu_lik_free(ctx, a_n7); iqtree_gpu_lik_free(ctx, a_n8);
        iqtree_gpu_lik_free(ctx, b_n5); iqtree_gpu_lik_free(ctx, b_n6);
        iqtree_gpu_lik_free(ctx, b_n7); iqtree_gpu_lik_free(ctx, b_n8);
        iqtree_gpu_lik_free(ctx, as5); iqtree_gpu_lik_free(ctx, as6);
        iqtree_gpu_lik_free(ctx, as7); iqtree_gpu_lik_free(ctx, as8);
        iqtree_gpu_lik_free(ctx, bs5); iqtree_gpu_lik_free(ctx, bs6);
        iqtree_gpu_lik_free(ctx, bs7); iqtree_gpu_lik_free(ctx, bs8);
    }

    /* ---- DEVICE-LEN full-traversal replay with DIFFERENT branch lengths (STEP 2) ----
     * This is the enabling change under test: a captured graph precomputes echild through the
     * DEVICE-LEN path (branch_precompute_dev, reading d_branch_len[branch_id]), so replaying it
     * after uploading a NEW length vector must reproduce exactly what the by-value path computes
     * for those new lengths. We capture ONE graph and replay it under two different length sets,
     * each time comparing bit-exactly (tol 0.0) against an independent by-value recomputation.
     *
     * Slot k is given branch id k (identity), so set_branch_lengths([blen...]) makes
     * d_branch_len[k] == blen[k] and k_echild_dev(slot k) reads exactly the length
     * branch_precompute(slot k, blen[k]) used. The graph writes the g_* output slots; the by-
     * value reference writes independent a_* slots; echild slots are reused transiently within
     * each pass, so the two never collide. */
    {
        double lensA[8] = { 0.11, 0.23, 0.07, 0.31, 0.17, 0.29, 0.13, 0.19 };  // == blen above
        double lensB[8] = { 0.41, 0.05, 0.27, 0.09, 0.33, 0.02, 0.22, 0.37 };  // a different tree

        double        *g5 = iqtree_gpu_lik_alloc_partial(ctx), *g6 = iqtree_gpu_lik_alloc_partial(ctx);
        double        *g7 = iqtree_gpu_lik_alloc_partial(ctx), *g8 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *gc5 = iqtree_gpu_lik_alloc_scale(ctx),  *gc6 = iqtree_gpu_lik_alloc_scale(ctx);
        unsigned char *gc7 = iqtree_gpu_lik_alloc_scale(ctx),  *gc8 = iqtree_gpu_lik_alloc_scale(ctx);
        double        *r5 = iqtree_gpu_lik_alloc_partial(ctx), *r6 = iqtree_gpu_lik_alloc_partial(ctx);
        double        *r7 = iqtree_gpu_lik_alloc_partial(ctx), *r8 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *rc5 = iqtree_gpu_lik_alloc_scale(ctx),  *rc6 = iqtree_gpu_lik_alloc_scale(ctx);
        unsigned char *rc7 = iqtree_gpu_lik_alloc_scale(ctx),  *rc8 = iqtree_gpu_lik_alloc_scale(ctx);
        if (!g8 || !gc8 || !r8 || !rc8) die("devlen: alloc");

        /* The four echild slots that carry a length into this tree: 0,1 (cherry n5), 3,4
           (cherry n6), 2 (tip of n7), 6 (n5->n7/n8), 7 (n6->n8). Precompute helper for one pass. */
        auto precompute_dev = [&](void) {
            int slots[7] = {0, 1, 2, 3, 4, 6, 7};
            for (int i = 0; i < 7; i++)
                if (iqtree_gpu_lik_branch_precompute_dev(ctx, slots[i], slots[i], 1))
                    die("devlen: branch_precompute_dev");
        };
        auto precompute_val = [&](const double *L) {
            int slots[7] = {0, 1, 2, 3, 4, 6, 7};
            for (int i = 0; i < 7; i++)
                if (iqtree_gpu_lik_branch_precompute(ctx, slots[i], L[slots[i]], 1))
                    die("devlen: branch_precompute");
        };
        auto sweep_into = [&](double *n5, unsigned char *s5, double *n6, unsigned char *s6,
                              double *n7, unsigned char *s7, double *n8, unsigned char *s8) {
            if (iqtree_gpu_lik_partial_tip_tip(ctx, n5, s5, 0, 0, 1, 1)) die("devlen: tt n5");
            if (iqtree_gpu_lik_partial_tip_tip(ctx, n6, s6, 3, 3, 4, 4)) die("devlen: tt n6");
            if (iqtree_gpu_lik_partial_tip_internal(ctx, n7, s7, 2, 2, 6, n5, s5)) die("devlen: ti n7");
            if (iqtree_gpu_lik_partial_internal_internal(ctx, n8, s8, 6, n5, s5, 7, n6, s6))
                die("devlen: ii n8");
        };

        /* Capture ONE graph: device-len echild precompute + the four-node sweep, into g_* slots. */
        if (iqtree_gpu_lik_set_branch_lengths(ctx, lensA, 8)) die("devlen: set_branch_lengths A(pre)");
        IqtreeGpuGraph *dg = NULL;
        if (iqtree_gpu_lik_graph_capture_begin(ctx)) die("devlen: capture_begin");
        precompute_dev();
        sweep_into(g5, gc5, g6, gc6, g7, gc7, g8, gc8);
        if (iqtree_gpu_lik_graph_capture_end(ctx, &dg) || !dg) die("devlen: capture_end");

        vector<double> R(PE), REF(PE);
        vector<unsigned char> RS(SE), REFS(SE);

        /* --- length set A: replay, compare to a fresh by-value computation --- */
        if (iqtree_gpu_lik_set_branch_lengths(ctx, lensA, 8)) die("devlen: set A");
        if (iqtree_gpu_lik_graph_launch(ctx, dg)) die("devlen: replay A");
        if (iqtree_gpu_lik_sync(ctx)) die("devlen: sync A");
        if (iqtree_gpu_lik_download_partial(ctx, R.data(), g8)) die("devlen: dl A");
        if (iqtree_gpu_lik_download_scale(ctx, RS.data(), gc8)) die("devlen: dls A");
        precompute_val(lensA);
        sweep_into(r5, rc5, r6, rc6, r7, rc7, r8, rc8);
        if (iqtree_gpu_lik_sync(ctx)) die("devlen: ref sync A");
        if (iqtree_gpu_lik_download_partial(ctx, REF.data(), r8)) die("devlen: ref dl A");
        if (iqtree_gpu_lik_download_scale(ctx, REFS.data(), rc8)) die("devlen: ref dls A");
        check_arr("devlen replay A partial (bit-exact)", R, REF, 0.0);
        check_scale("devlen replay A scale (bit-exact)", RS, REFS);

        /* --- length set B (different): SAME graph, new lengths uploaded, must track --- */
        if (iqtree_gpu_lik_set_branch_lengths(ctx, lensB, 8)) die("devlen: set B");
        if (iqtree_gpu_lik_graph_launch(ctx, dg)) die("devlen: replay B");
        if (iqtree_gpu_lik_sync(ctx)) die("devlen: sync B");
        if (iqtree_gpu_lik_download_partial(ctx, R.data(), g8)) die("devlen: dl B");
        if (iqtree_gpu_lik_download_scale(ctx, RS.data(), gc8)) die("devlen: dls B");
        precompute_val(lensB);
        sweep_into(r5, rc5, r6, rc6, r7, rc7, r8, rc8);
        if (iqtree_gpu_lik_sync(ctx)) die("devlen: ref sync B");
        if (iqtree_gpu_lik_download_partial(ctx, REF.data(), r8)) die("devlen: ref dl B");
        if (iqtree_gpu_lik_download_scale(ctx, REFS.data(), rc8)) die("devlen: ref dls B");
        check_arr("devlen replay B partial (bit-exact)", R, REF, 0.0);
        check_scale("devlen replay B scale (bit-exact)", RS, REFS);

        /* Sanity: A and B must actually DIFFER, else the length change did nothing. */
        {
            if (iqtree_gpu_lik_set_branch_lengths(ctx, lensA, 8)) die("devlen: set A2");
            if (iqtree_gpu_lik_graph_launch(ctx, dg)) die("devlen: replay A2");
            if (iqtree_gpu_lik_sync(ctx)) die("devlen: sync A2");
            vector<double> RA(PE);
            if (iqtree_gpu_lik_download_partial(ctx, RA.data(), g8)) die("devlen: dl A2");
            double maxdiff = 0.0;
            for (size_t i = 0; i < PE; i++) maxdiff = fmax(maxdiff, fabs(RA[i] - R[i]));
            if (maxdiff > 0.0)
                printf("  [ ok ] %-34s A vs B differ (max %.3e), length change took effect\n",
                       "devlen replay distinctness", maxdiff);
            else {
                printf("  [FAIL] %-34s A and B replays identical -- device length not read\n",
                       "devlen replay distinctness");
                g_failures++;
            }
        }

        iqtree_gpu_lik_graph_free(dg);
        double *dfree[8] = {g5,g6,g7,g8,r5,r6,r7,r8};
        for (int i = 0; i < 8; i++) iqtree_gpu_lik_free(ctx, dfree[i]);
        unsigned char *sfree[8] = {gc5,gc6,gc7,gc8,rc5,rc6,rc7,rc8};
        for (int i = 0; i < 8; i++) iqtree_gpu_lik_free(ctx, sfree[i]);

        /* Restore every echild slot to the original blen state: this block overwrote slots
           0..7 with lensA/lensB, and the multifurcating case below (and any later test) reads
           slots 5,6,7 expecting the blen-based echild computed at setup. */
        for (int s = 0; s < 8; s++)
            if (iqtree_gpu_lik_branch_precompute(ctx, s, blen[s], 1)) die("devlen: restore echild");
        if (iqtree_gpu_lik_sync(ctx)) die("devlen: restore sync");
    }

    /* ---- case 4: multifurcating node with three children (t5, n5, n6) ---- */
    double *d_mf = iqtree_gpu_lik_alloc_partial(ctx);
    unsigned char *s_mf = iqtree_gpu_lik_alloc_scale(ctx);
    IqtreeGpuChild kids[3];
    memset(kids, 0, sizeof(kids));
    kids[0].echild_slot = 5; kids[0].is_tip = 1; kids[0].tip_slot = 5;
    kids[1].echild_slot = 6; kids[1].is_tip = 0; kids[1].d_partial = d_n5; kids[1].d_scale = s_n5;
    kids[2].echild_slot = 7; kids[2].is_tip = 0; kids[2].d_partial = d_n6; kids[2].d_scale = s_n6;
    if (iqtree_gpu_lik_partial_multifurcating(ctx, d_mf, s_mf, 3, kids)) die("multifurcating");
    if (iqtree_gpu_lik_sync(ctx)) die("sync");

    /* Reference: fold children in one at a time, testing for underflow after each, all in
       the x basis; rotate into the eigen basis only at the end. */
    vector<unsigned char> sc(SE, 0);   /* kept in scope for the cross-shape check below */
    {
        vector<double> u(PE, 1.0);
        for (int k = 0; k < 3; k++) {
            const vector<double> *src = NULL;
            const vector<unsigned char> *ssrc = NULL;
            if (k == 1) { src = &ref_n5; ssrc = &ref_s5; }
            if (k == 2) { src = &ref_n6; ssrc = &ref_s6; }
            if (ssrc) for (size_t i = 0; i < SE; i++) sc[i] = (unsigned char)(sc[i] + (*ssrc)[i]);
            for (int c = 0; c < NCAT; c++)
                for (int p = 0; p < r.stride; p++)
                    for (int x = 0; x < NST; x++) {
                        double v;
                        if (k == 0) {
                            v = leaf[5][(size_t)states_full[5][p] * r.block + c * NST + x];
                        } else {
                            v = 0.0;
                            const vector<double> &e = ech[k == 1 ? 6 : 7];
                            for (int i = 0; i < NST; i++)
                                v += e[((size_t)(c * NST + x)) * NST + i]
                                   * (*src)[((size_t)(c * NST + i)) * r.stride + p];
                        }
                        u[((size_t)(c * NST + x)) * r.stride + p] *= v;
                    }
            ref_rescale<NST>(r, u, sc);
        }
        cpu.assign(PE, 0.0);
        for (int c = 0; c < NCAT; c++)
            for (int p = 0; p < r.stride; p++)
                for (int k = 0; k < NST; k++) {
                    double acc = 0.0;
                    for (int x = 0; x < NST; x++)
                        acc += r.inv_evec[k * NST + x] * u[((size_t)(c * NST + x)) * r.stride + p];
                    cpu[((size_t)(c * NST + k)) * r.stride + p] = acc;
                }
        if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), d_mf)) die("download");
        if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), s_mf)) die("download scale");
        check_arr("partial multifurcating", gpu, cpu, 1e-12);
        check_scale("scale multifurcating", gscale, sc);
    }

    /* ================================================================================
     * CROSS-SHAPE AGREEMENT (4 states only)
     *
     * At nstates == 4 the engine runs a dedicated kernel shape for k_partial and k_mf_child:
     * one thread owns all four states of a (pattern, category), there is no shared-memory
     * staging and no __syncthreads, and the tile is 128 patterns wide instead of 32. The
     * generic shape -- literally the same source the 20-state path runs, instantiated at 4 --
     * is also compiled, and iqtree_gpu_lik_force_generic_shape() selects it.
     *
     * Running BOTH against the SAME scalar reference is what makes a bug in the new shape
     * cheap to find: if the generic shape agrees and the specialisation does not, the fault is
     * in the specialisation and nowhere else -- not in the reference, not in the host wrappers,
     * not in the scaling semantics, all of which are shared. The tolerance is the same 1e-12;
     * the reduction order was deliberately kept identical between the two shapes precisely so
     * that it could be.
     * ============================================================================== */
    if (NST == 4) {
        printf("\n--- cross-shape: generic k_partial<4> vs the 4-state specialisation ---\n");
        if (iqtree_gpu_lik_force_generic_shape(ctx, 1)) die("force_generic_shape");

        double        *g_n5 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *g_s5 = iqtree_gpu_lik_alloc_scale(ctx);
        double        *g_n6 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *g_s6 = iqtree_gpu_lik_alloc_scale(ctx);
        double        *g_n7 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *g_s7 = iqtree_gpu_lik_alloc_scale(ctx);
        double        *g_n8 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *g_s8 = iqtree_gpu_lik_alloc_scale(ctx);
        double        *g_mf = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *g_smf = iqtree_gpu_lik_alloc_scale(ctx);
        if (!g_n5 || !g_s5 || !g_n6 || !g_s6 || !g_n7 || !g_s7 || !g_n8 || !g_s8 ||
            !g_mf || !g_smf) die("alloc cross-shape");

        if (iqtree_gpu_lik_partial_tip_tip(ctx, g_n5, g_s5, 0, 0, 1, 1)) die("generic tip_tip");
        if (iqtree_gpu_lik_partial_tip_tip(ctx, g_n6, g_s6, 3, 3, 4, 4)) die("generic tip_tip2");
        if (iqtree_gpu_lik_partial_tip_internal(ctx, g_n7, g_s7, 2, 2, 6, g_n5, g_s5))
            die("generic tip_internal");
        if (iqtree_gpu_lik_partial_internal_internal(ctx, g_n8, g_s8, 6, g_n5, g_s5,
                                                     7, g_n6, g_s6)) die("generic int_int");
        {
            IqtreeGpuChild gk[3];
            memset(gk, 0, sizeof(gk));
            gk[0].echild_slot = 5; gk[0].is_tip = 1; gk[0].tip_slot = 5;
            gk[1].echild_slot = 6; gk[1].is_tip = 0; gk[1].d_partial = g_n5; gk[1].d_scale = g_s5;
            gk[2].echild_slot = 7; gk[2].is_tip = 0; gk[2].d_partial = g_n6; gk[2].d_scale = g_s6;
            if (iqtree_gpu_lik_partial_multifurcating(ctx, g_mf, g_smf, 3, gk))
                die("generic multifurcating");
        }
        if (iqtree_gpu_lik_sync(ctx)) die("sync");

        if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), g_n5)) die("download");
        check_arr("generic-shape partial tip-tip", gpu, ref_n5, 1e-12);
        if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), g_n7)) die("download");
        if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), g_s7)) die("download scale");
        check_arr("generic-shape partial tip-internal", gpu, ref_n7, 1e-12);
        check_scale("generic-shape scale tip-internal", gscale, ref_s7);
        if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), g_n8)) die("download");
        if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), g_s8)) die("download scale");
        check_arr("generic-shape partial internal-int", gpu, ref_n8, 1e-12);
        check_scale("generic-shape scale internal-int", gscale, ref_s8);
        if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), g_mf)) die("download");
        if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), g_smf)) die("download scale");
        check_arr("generic-shape partial multifurcating", gpu, cpu, 1e-12);
        check_scale("generic-shape scale multifurcating", gscale, sc);

        iqtree_gpu_lik_free(ctx, g_n5); iqtree_gpu_lik_free(ctx, g_s5);
        iqtree_gpu_lik_free(ctx, g_n6); iqtree_gpu_lik_free(ctx, g_s6);
        iqtree_gpu_lik_free(ctx, g_n7); iqtree_gpu_lik_free(ctx, g_s7);
        iqtree_gpu_lik_free(ctx, g_n8); iqtree_gpu_lik_free(ctx, g_s8);
        iqtree_gpu_lik_free(ctx, g_mf); iqtree_gpu_lik_free(ctx, g_smf);

        /* Back to the shape the integration layer actually uses; everything after this point
           must exercise the specialisation, not the generic fallback. */
        if (iqtree_gpu_lik_force_generic_shape(ctx, 0)) die("force_generic_shape reset");
    }

    /* ---- theta with an internal dad: branch n7 -- n8 ---- */
    if (iqtree_gpu_lik_theta_internal(ctx, d_n7, s_n7, d_n8, s_n8)) die("theta_internal");
    if (iqtree_gpu_lik_sync(ctx)) die("sync");

    vector<double> ref_theta_ii, ref_buf_ii;
    ref_theta<NST>(r, ref_n7, ref_n8, ref_s7.data(), ref_s8.data(), ref_theta_ii, ref_buf_ii);

    vector<double> gbuf(r.stride);
    if (iqtree_gpu_lik_download_theta(ctx, gpu.data())) die("download theta");
    if (iqtree_gpu_lik_download_buffer_scale(ctx, gbuf.data())) die("download bufscale");
    check_arr("theta internal-internal", gpu, ref_theta_ii, 1e-12);
    check_arr("buffer_scale internal", gbuf, ref_buf_ii, 1e-12);

    /* ---- branch logL and analytic derivatives on that branch ---- */
    {
        const double t = 0.137;
        double g_logl = 0.0;
        if (iqtree_gpu_lik_branch_logl(ctx, t, &g_logl)) die("branch_logl");
        double c_logl = ref_branch_logl<NST>(r, ref_theta_ii, ref_buf_ii, t);
        check_scalar("branch logL", g_logl, c_logl, 1e-12);

        double g_l2 = 0.0, g_df = 0.0, g_ddf = 0.0;
        if (iqtree_gpu_lik_derivatives(ctx, t, &g_l2, &g_df, &g_ddf)) die("derivatives");
        double c_l2, c_df, c_ddf;
        ref_derivatives<NST>(r, ref_theta_ii, ref_buf_ii, t, &c_l2, &c_df, &c_ddf);
        check_scalar("derivative-path logL", g_l2, c_l2, 1e-12);
        check_scalar("df (analytic)",        g_df, c_df, 1e-12);
        check_scalar("ddf (analytic)",       g_ddf, c_ddf, 1e-12);

        /* Independent evidence that the analytic formula really is the derivative of the
           log-likelihood: central differences of the CPU logL. The tolerances here are
           deliberately loose -- a central difference is limited by truncation at h^2 and
           by cancellation at eps/h^2, so it can only ever confirm the formula, never
           measure the kernel's precision. That is what the 1e-12 checks above are for. */
        const double h = 1e-3;
        double lp = ref_logl_derv_form<NST>(r, ref_theta_ii, ref_buf_ii, t + h);
        double lm = ref_logl_derv_form<NST>(r, ref_theta_ii, ref_buf_ii, t - h);
        double l0 = ref_logl_derv_form<NST>(r, ref_theta_ii, ref_buf_ii, t);
        check_scalar("df vs finite difference",  g_df,  (lp - lm) / (2 * h), 1e-5);
        check_scalar("ddf vs finite difference", g_ddf, (lp - 2 * l0 + lm) / (h * h), 1e-3);
    }

    /* ---- Newton-loop graph replay: derivative + logL over DIFFERENT candidate lengths (STEP 2).
     * The per-iteration derivative and logL-only sequences read the candidate branch length from
     * the device scalar d_len, so ONE captured graph each replays for any length. Capture each
     * once (theta is in slot 0 == GPU_THETA_NEWTON from the theta_internal above), then replay
     * across several lengths and require the replayed (logL, df, ddf) to be BYTE-IDENTICAL
     * (exact ==) to a fresh by-value evaluation at the same length: same buffers, same kernels,
     * the length arriving from memory instead of an argument must not move a single bit. This is
     * the exact per-iteration building block the on-device Newton loop would iterate. */
    {
        if (iqtree_gpu_lik_set_theta_slot(ctx, 0)) die("newton: select theta slot");

        IqtreeGpuGraph *dgraph = NULL, *lgraph = NULL;
        if (iqtree_gpu_lik_graph_capture_begin(ctx)) die("newton: derv capture begin");
        if (iqtree_gpu_lik_derivatives_dev_issue(ctx)) die("newton: derv issue (capture)");
        if (iqtree_gpu_lik_graph_capture_end(ctx, &dgraph) || !dgraph) die("newton: derv capture end");

        if (iqtree_gpu_lik_graph_capture_begin(ctx)) die("newton: logl capture begin");
        if (iqtree_gpu_lik_branch_logl_dev_issue(ctx, 1)) die("newton: logl issue (capture)");
        if (iqtree_gpu_lik_graph_capture_end(ctx, &lgraph) || !lgraph) die("newton: logl capture end");

        const double newton_lens[5] = { 0.05, 0.137, 0.42, 0.9, 0.011 };
        int derv_ok = 1, logl_ok = 1;
        double first_df = 0.0, last_df = 0.0;
        for (int i = 0; i < 5; i++) {
            double t = newton_lens[i];
            double rl = 0, rdf = 0, rddf = 0, rlogl = 0;   // by-value reference
            if (iqtree_gpu_lik_derivatives(ctx, t, &rl, &rdf, &rddf)) die("newton: ref derivatives");
            if (iqtree_gpu_lik_branch_logl(ctx, t, &rlogl)) die("newton: ref branch_logl");

            double pl = 0, pdf = 0, pddf = 0, plogl = 0;   // replayed
            if (iqtree_gpu_lik_set_eval_len(ctx, t)) die("newton: set_eval_len (derv)");
            if (iqtree_gpu_lik_graph_launch(ctx, dgraph)) die("newton: derv replay");
            if (iqtree_gpu_lik_sync(ctx)) die("newton: derv sync");
            if (iqtree_gpu_lik_derivatives_readback(ctx, &pl, &pdf, &pddf)) die("newton: derv readback");
            if (iqtree_gpu_lik_set_eval_len(ctx, t)) die("newton: set_eval_len (logl)");
            if (iqtree_gpu_lik_graph_launch(ctx, lgraph)) die("newton: logl replay");
            if (iqtree_gpu_lik_sync(ctx)) die("newton: logl sync");
            if (iqtree_gpu_lik_branch_logl_readback(ctx, &plogl)) die("newton: logl readback");

            if (pl != rl || pdf != rdf || pddf != rddf) derv_ok = 0;
            if (plogl != rlogl) logl_ok = 0;
            if (i == 0) first_df = pdf;
            if (i == 4) last_df  = pdf;
        }
        printf(derv_ok ? "  [ ok ] %-34s bit-exact vs by-value over 5 lengths\n"
                       : "  [FAIL] %-34s replay != by-value\n", "newton derv replay");
        printf(logl_ok ? "  [ ok ] %-34s bit-exact vs by-value over 5 lengths\n"
                       : "  [FAIL] %-34s replay != by-value\n", "newton logl replay");
        if (!derv_ok) g_failures++;
        if (!logl_ok) g_failures++;
        /* The device length must actually be read: df at t=0.05 and t=0.011 must differ. */
        if (first_df == last_df) {
            printf("  [FAIL] %-34s df identical across lengths -- d_len not read\n",
                   "newton replay distinctness");
            g_failures++;
        } else {
            printf("  [ ok ] %-34s df varies with length as expected\n",
                   "newton replay distinctness");
        }

        iqtree_gpu_lik_graph_free(dgraph);
        iqtree_gpu_lik_graph_free(lgraph);
    }

    /* ---- On-device Newton loop vs host minimizeNewton (STEP 3) ----
     * Optimise a single branch entirely on-device (one conditional-WHILE graph launch) and via the
     * host port of minimizeNewton, from the SAME starting guesses. Both drive the same derivative
     * kernels and apply the same scalar logic, so the converged length must match -- ideally bit
     * for bit; a tolerance guard catches any rare fma-contraction difference in the bracket test.
     * theta is already in slot 0 (== GPU_THETA_NEWTON) from the theta_internal above and is
     * independent of the branch length, so it is the fixed theta the loop reuses each iteration. */
    {
        if (iqtree_gpu_lik_set_theta_slot(ctx, 0)) die("newton loop: select theta slot");
        if (iqtree_gpu_lik_newton_build(ctx)) die("newton loop: build");

        const double x1 = 1e-6, x2 = 10.0, xacc = 1e-6;
        const int    maxNR = 100;
        const double guesses[4] = { 0.05, 0.5, 0.137, 2.0 };
        int tol_ok = 1, bitexact = 1;
        double worst = 0.0;
        for (int i = 0; i < 4; i++) {
            int eh = 0, ed = 0;
            double oh = host_minimize_newton(ctx, x1, guesses[i], x2, xacc, maxNR, &eh);
            double od = 0.0;
            if (iqtree_gpu_lik_newton_optimize(ctx, x1, x2, xacc, maxNR, guesses[i], &od, &ed))
                die("newton loop: optimize");
            if (eh != ed) { tol_ok = 0; bitexact = 0; }
            if (!eh && !ed) {
                if (oh != od) bitexact = 0;
                double d = fabs(oh - od);
                if (d > worst) worst = d;
                if (d > 1e-9 + 1e-7 * fabs(oh)) tol_ok = 0;
            }
        }
        if (tol_ok)
            printf("  [ ok ] %-34s %s (worst |dlen| %.3e over 4 starts)\n",
                   "newton on-device loop",
                   bitexact ? "bit-exact vs host minimizeNewton" : "within tol vs host", worst);
        else {
            printf("  [FAIL] %-34s on-device != host minimizeNewton (worst |dlen| %.3e)\n",
                   "newton on-device loop", worst);
            g_failures++;
        }
    }

    /* ================================================================================
     * G MATRIX AND THE MCMCTree HESSIAN
     *
     * `--dating mcmctree` needs the per-pattern first derivative G[i][p] = (dL_p/dt_i)/L_p
     * for every branch i, and then H = G*diag(ptn_freq)*G^T. Both are checked here against
     * the scalar reference above, because getting either the row pitch or the DSYRK
     * orientation wrong produces a full, plausible-looking matrix rather than an error.
     *
     * The row pitch is deliberately NOT the pattern stride: IQ-TREE's host G_matrix is
     * mem_size wide (padded pattern count PLUS num_states), and the kernel has to write into
     * that pitch while touching only the first npat lanes of each row.
     * ============================================================================== */
    printf("\n--- G matrix / MCMCTree Hessian ---\n");
    {
        const int NBR      = 5;                 /* stand-in for branchNum                  */
        const int G_STRIDE = r.stride + NST;    /* mimics mem_size = padded_nptn + nstates */

        /* Asking for a G row before enabling one must fail, not write somewhere. */
        {
            double l = 0, d1 = 0, d2 = 0;
            if (iqtree_gpu_lik_derivatives_g(ctx, 0.1, 0, &l, &d1, &d2) == 0) {
                printf("  [FAIL] a G row was accepted with no G matrix enabled\n");
                g_failures++;
            } else {
                printf("  [ ok ] %-34s %s\n", "G row without enable", iqtree_gpu_lik_last_error());
            }
        }

        if (iqtree_gpu_lik_gmatrix_enable(ctx, NBR, G_STRIDE)) die("gmatrix_enable");

        /* One "branch" per row, each at its own length. Different lengths give genuinely
           different rows, which is what makes the Hessian comparison able to see a pitch or
           scaling error instead of agreeing by symmetry. */
        vector< vector<double> > ref_rows(NBR);
        vector<double> grow(G_STRIDE);
        int pad_bad = 0;
        for (int b = 0; b < NBR; b++) {
            const double len = 0.05 + 0.09 * b;
            double g_l = 0.0, g_d1 = 0.0, g_d2 = 0.0;
            if (iqtree_gpu_lik_derivatives_g(ctx, len, b, &g_l, &g_d1, &g_d2))
                die("derivatives_g");

            /* The scalars must be exactly what the plain derivative call produces: emitting a
               G row must not perturb the reduction it is emitted from. */
            double c_l, c_d1, c_d2;
            ref_derivatives<NST>(r, ref_theta_ii, ref_buf_ii, len, &c_l, &c_d1, &c_d2);
            if (b == 0) {
                check_scalar("derivatives_g: logL", g_l,  c_l,  1e-12);
                check_scalar("derivatives_g: df",   g_d1, c_d1, 1e-12);
                check_scalar("derivatives_g: ddf",  g_d2, c_d2, 1e-12);
            }

            if (iqtree_gpu_lik_gmatrix_download_row(ctx, b, grow.data()))
                die("gmatrix_download_row");
            ref_gmatrix_row<NST>(r, ref_theta_ii, len, G_STRIDE, ref_rows[b]);

            char name[64];
            snprintf(name, sizeof(name), "G row %d (per-pattern df/L)", b);
            check_arr(name, grow, ref_rows[b], 1e-12);

            /* The padding is a correctness requirement, not tidiness: the host consumer maps
               the whole mem_size-wide row and multiplies it by a ptn_freq diagonal that is
               exactly 0.0 there, so anything non-zero -- and in particular anything
               non-finite -- has to be excluded by construction, not by luck. */
            for (int p = r.npat; p < G_STRIDE; p++)
                if (grow[p] != 0.0) pad_bad++;
        }
        if (pad_bad) {
            printf("  [FAIL] %d padding lanes of G are non-zero\n", pad_bad);
            g_failures++;
        } else {
            printf("  [ ok ] %-34s lanes [%d,%d) are exactly 0.0 in every row\n",
                   "G padding", r.npat, G_STRIDE);
        }

        /* A plain derivative call must leave G alone -- that is what makes the device-to-host
           copy safe to gate on --dating mcmctree without the Newton loop disturbing it. */
        {
            double l = 0, d1 = 0, d2 = 0;
            if (iqtree_gpu_lik_derivatives(ctx, 0.42, &l, &d1, &d2)) die("derivatives plain");
            if (iqtree_gpu_lik_gmatrix_download_row(ctx, 0, grow.data()))
                die("gmatrix_download_row");
            check_arr("G untouched by a plain derivative call", grow, ref_rows[0], 1e-12);
        }

        /* H = G*diag(ptn_freq)*G^T on the device (one cuBLAS DSYRK) against the triple loop.
           Compared ELEMENTWISE rather than by a norm, because an orientation or pitch error
           shows up in individual entries long before it moves an aggregate. */
        {
            vector<double> gh(NBR * NBR, 0.0), ch;
            if (iqtree_gpu_lik_gmatrix_hessian(ctx, gh.data())) die("gmatrix_hessian");
            ref_gmatrix_hessian<NST>(r, ref_rows, ch);

            double scale = 0.0;
            for (size_t i = 0; i < ch.size(); i++) scale = max(scale, fabs(ch[i]));
            double worst = 0.0;
            int    wi = 0, wj = 0, nbad = 0, nasym = 0;
            for (int i = 0; i < NBR; i++)
                for (int j = 0; j < NBR; j++) {
                    double d = fabs(gh[i * NBR + j] - ch[i * NBR + j]) / (scale > 0 ? scale : 1.0);
                    if (d > worst) { worst = d; wi = i; wj = j; }
                    if (d > 1e-12) nbad++;
                    /* Both triangles must be populated: DSYRK writes only one and the engine
                       mirrors the other, so an unmirrored half would leave garbage that a
                       symmetric reference would never reveal. */
                    if (fabs(gh[i * NBR + j] - gh[j * NBR + i]) > 0.0) nasym++;
                }
            if (nbad) {
                printf("  [FAIL] %-30s %d of %d entries differ; worst %.3e at (%d,%d) "
                       "[gpu %.17g cpu %.17g]\n", "Hessian G*diag(freq)*G^T", nbad, NBR * NBR,
                       worst, wi, wj, gh[wi * NBR + wj], ch[wi * NBR + wj]);
                g_failures++;
            } else {
                printf("  [ ok ] %-34s max rel %.3e over %d entries\n",
                       "Hessian G*diag(freq)*G^T", worst, NBR * NBR);
            }
            if (nasym) {
                printf("  [FAIL] returned Hessian is not symmetric in %d entries "
                       "(the second triangle was not mirrored)\n", nasym);
                g_failures++;
            }
            /* Guard against a degenerate comparison: if every entry were zero the loop above
               would pass without testing anything. */
            if (!(scale > 0.0)) {
                printf("  [FAIL] the reference Hessian is identically zero; nothing was tested\n");
                g_failures++;
            }
        }
    }

    /* ---- theta with a leaf dad (tip slot 5) ---- */
    if (iqtree_gpu_lik_theta_tip(ctx, 5, d_n8, s_n8)) die("theta_tip");
    if (iqtree_gpu_lik_sync(ctx)) die("sync");
    {
        vector<double> tipnode(PE);
        for (int c = 0; c < NCAT; c++)
            for (int i = 0; i < NST; i++)
                for (int p = 0; p < r.stride; p++)
                    tipnode[((size_t)(c * NST + i)) * r.stride + p] =
                        r.tip_lh[(size_t)states_full[5][p] * NST + i];
        vector<double> th, bs;
        ref_theta<NST>(r, tipnode, ref_n8, NULL, ref_s8.data(), th, bs);
        if (iqtree_gpu_lik_download_theta(ctx, gpu.data())) die("download theta");
        if (iqtree_gpu_lik_download_buffer_scale(ctx, gbuf.data())) die("download bufscale");
        check_arr("theta tip-dad", gpu, th, 1e-12);
        check_arr("buffer_scale tip-dad", gbuf, bs, 1e-12);
    }

    /* ---- rescaling: force a genuine underflow so the ldexp path and the counter fire ---- */
    {
        vector<double> tiny(PE);
        for (size_t i = 0; i < PE; i++) tiny[i] = ldexp(ref_n8[i], -600);   /* far below 2^-256 */
        if (iqtree_gpu_lik_upload_partial(ctx, d_mf, tiny.data())) die("upload");
        vector<unsigned char> zero(SE, 0);
        if (iqtree_gpu_lik_upload_scale(ctx, s_mf, zero.data())) die("upload scale");

        /* Drive the rescale through the public tip-internal path with the tiny buffer as
           the internal child: the whole pipeline (dot products, inv_evec rotation,
           underflow test, ldexp, counter bump) is then compared against the reference. */
        double *d_t2 = iqtree_gpu_lik_alloc_partial(ctx);
        unsigned char *s_t2 = iqtree_gpu_lik_alloc_scale(ctx);
        if (iqtree_gpu_lik_partial_tip_internal(ctx, d_t2, s_t2, 2, 2, 6, d_mf, s_mf))
            die("tip_internal tiny");
        if (iqtree_gpu_lik_sync(ctx)) die("sync");

        vector<double> ref_t2;
        ref_partial<NST>(r, true,  leaf[2], states_full[2], nil,    nil,
                       false, nil,     nili,           ech[6], tiny, ref_t2);
        vector<unsigned char> ref_st2(SE, 0);
        ref_rescale<NST>(r, ref_t2, ref_st2);

        if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), d_t2)) die("download");
        if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), s_t2)) die("download scale");
        check_arr("partial after underflow", gpu, ref_t2, 1e-12);
        check_scale("scale counter after underflow", gscale, ref_st2);

        int nscaled = 0;
        for (size_t i = 0; i < SE; i++) if (ref_st2[i]) nscaled++;
        printf("  [info] %d of %zu counters incremented (rescale path exercised)\n", nscaled, SE);
        if (nscaled == 0) {
            printf("  [FAIL] underflow test did not actually trigger a rescale\n");
            g_failures++;
        }

        iqtree_gpu_lik_free(ctx, d_t2);
        iqtree_gpu_lik_free(ctx, s_t2);
    }

    iqtree_gpu_lik_free(ctx, d_n5); iqtree_gpu_lik_free(ctx, s_n5);
    iqtree_gpu_lik_free(ctx, d_n6); iqtree_gpu_lik_free(ctx, s_n6);
    iqtree_gpu_lik_free(ctx, d_n7); iqtree_gpu_lik_free(ctx, s_n7);
    iqtree_gpu_lik_free(ctx, d_n8); iqtree_gpu_lik_free(ctx, s_n8);
    iqtree_gpu_lik_free(ctx, d_mf); iqtree_gpu_lik_free(ctx, s_mf);
    iqtree_gpu_lik_destroy(ctx);

    /* ================================================================================
     * RATE-CATEGORY CAPACITY
     *
     * The engine sizes every category-dependent buffer for cfg.ncat_cap and then runs at any
     * ncat <= that, because buf[(cat*20 + state)*stride + pattern] makes the smaller layout an
     * exact PREFIX of the larger one. That claim is what removes a multi-GB reallocation every
     * time ModelFinder moves between +G4 and +R10, so it is worth testing directly rather than
     * inferring from the layout.
     *
     * The test is deliberately adversarial about staleness: the arena is first driven at the
     * FULL capacity, so every category plane -- including the six that the 4-category run does
     * not own -- holds real, non-zero, differently-parameterised data. Only then is the engine
     * dropped to 4 categories and re-run. If anything read past its own prefix, or if a launch
     * shape were still sized from the capacity, those planes are what would show up. The
     * results are compared against the SAME references the freshly-created 4-category engine
     * was checked against above, at the same scale-aware tolerance.
     * ============================================================================== */
    printf("\n--- ncat capacity (arena for %d categories, running at %d) ---\n",
           NCAT_CAP, NCAT);
    {
        IqtreeGpuLikConfig ccfg = cfg;      /* cfg is still the NORM_LH configuration here */
        ccfg.ncat     = NCAT_CAP;
        ccfg.ncat_cap = NCAT_CAP;

        IqtreeGpuLik *cap = NULL;
        if (iqtree_gpu_lik_create(&cap, &ccfg)) die("create cap");
        if (iqtree_gpu_lik_ncat_cap(cap) != NCAT_CAP || iqtree_gpu_lik_ncat(cap) != NCAT_CAP) {
            printf("  [FAIL] capacity/ncat not reported as configured (%d / %d)\n",
                   iqtree_gpu_lik_ncat_cap(cap), iqtree_gpu_lik_ncat(cap));
            g_failures++;
        }

        /* Allocated ONCE, at the capacity. Nothing below reallocates them. */
        double        *c_n5 = iqtree_gpu_lik_alloc_partial(cap);
        unsigned char *c_s5 = iqtree_gpu_lik_alloc_scale(cap);
        double        *c_n6 = iqtree_gpu_lik_alloc_partial(cap);
        unsigned char *c_s6 = iqtree_gpu_lik_alloc_scale(cap);
        double        *c_n7 = iqtree_gpu_lik_alloc_partial(cap);
        unsigned char *c_s7 = iqtree_gpu_lik_alloc_scale(cap);
        double        *c_n8 = iqtree_gpu_lik_alloc_partial(cap);
        unsigned char *c_s8 = iqtree_gpu_lik_alloc_scale(cap);
        if (!c_n5 || !c_s5 || !c_n6 || !c_s6 || !c_n7 || !c_s7 || !c_n8 || !c_s8) die("alloc cap");

        /* ---- phase 1: run at the full capacity, to dirty every category plane ---- */
        vector<double> rates_hi(NCAT_CAP), props_hi(NCAT_CAP);
        for (int c = 0; c < NCAT_CAP; c++) {
            rates_hi[c] = 0.2 + 0.4 * c;
            props_hi[c] = 0.9 / NCAT_CAP;
        }
        if (iqtree_gpu_lik_set_eigen(cap, r.eval.data(), r.evec.data(), r.inv_evec.data()))
            die("cap set_eigen");
        if (iqtree_gpu_lik_set_rates(cap, rates_hi.data(), props_hi.data())) die("cap set_rates hi");
        if (iqtree_gpu_lik_set_tip_partials(cap, r.tip_lh.data())) die("cap set_tip_partials");
        if (iqtree_gpu_lik_set_pattern_data(cap, r.ptn_freq.data(), r.ptn_invar.data()))
            die("cap set_pattern_data");
        for (int t = 0; t < NTIP; t++)
            if (iqtree_gpu_lik_set_tip_states(cap, t, states[t].data())) die("cap set_tip_states");
        for (int s = 0; s < 8; s++)
            if (iqtree_gpu_lik_branch_precompute(cap, s, blen[s], 1)) die("cap precompute hi");
        if (iqtree_gpu_lik_partial_tip_tip(cap, c_n5, c_s5, 0, 0, 1, 1)) die("cap tip_tip hi");
        if (iqtree_gpu_lik_partial_tip_tip(cap, c_n6, c_s6, 3, 3, 4, 4)) die("cap tip_tip2 hi");
        if (iqtree_gpu_lik_partial_tip_internal(cap, c_n7, c_s7, 2, 2, 6, c_n5, c_s5))
            die("cap tip_internal hi");
        if (iqtree_gpu_lik_partial_internal_internal(cap, c_n8, c_s8, 6, c_n5, c_s5,
                                                     7, c_n6, c_s6)) die("cap int_int hi");
        /* Leaves theta dirty at 10 categories too. */
        if (iqtree_gpu_lik_theta_internal(cap, c_n7, c_s7, c_n8, c_s8)) die("cap theta hi");
        if (iqtree_gpu_lik_sync(cap)) die("sync");

        /* ---- phase 2: drop to NCAT. No allocation, no free. ---- */
        if (iqtree_gpu_lik_set_ncat(cap, NCAT)) die("set_ncat down");
        if (iqtree_gpu_lik_ncat(cap) != NCAT || iqtree_gpu_lik_block(cap) != NCAT * NST ||
            iqtree_gpu_lik_partial_elems(cap) != PE || iqtree_gpu_lik_scale_elems(cap) != SE) {
            printf("  [FAIL] geometry not recomputed by set_ncat "
                   "(ncat %d block %d pelems %zu selems %zu)\n",
                   iqtree_gpu_lik_ncat(cap), iqtree_gpu_lik_block(cap),
                   iqtree_gpu_lik_partial_elems(cap), iqtree_gpu_lik_scale_elems(cap));
            g_failures++;
        } else {
            printf("  [ ok ] %-34s ncat %d -> %d with no reallocation\n",
                   "set_ncat geometry", NCAT_CAP, NCAT);
        }
        /* Above the capacity must be refused, and must leave the engine as it was. */
        if (iqtree_gpu_lik_set_ncat(cap, NCAT_CAP + 1) == 0 || iqtree_gpu_lik_ncat(cap) != NCAT) {
            printf("  [FAIL] set_ncat accepted a value above the capacity\n");
            g_failures++;
        } else {
            printf("  [ ok ] %-34s %s\n", "set_ncat rejects > capacity",
                   iqtree_gpu_lik_last_error());
        }
        /* The rate/prop vectors are per-category DATA. set_ncat drops them on purpose, and the
           engine must say so instead of quietly reading the 10-category ones. */
        {
            double dummy = 0.0;
            if (iqtree_gpu_lik_branch_logl(cap, 0.1, &dummy) == 0) {
                printf("  [FAIL] stale rate vectors were not caught after set_ncat\n");
                g_failures++;
            } else {
                printf("  [ ok ] %-34s %s\n", "stale-rate guard", iqtree_gpu_lik_last_error());
            }
        }
        if (iqtree_gpu_lik_set_rates(cap, r.rates.data(), r.props.data())) die("cap set_rates lo");

        /* ---- phase 3: the same pipeline as the NORM_LH section, at 4 categories ---- */
        for (int s = 0; s < 8; s++)
            if (iqtree_gpu_lik_branch_precompute(cap, s, blen[s], 1)) die("cap precompute lo");
        if (iqtree_gpu_lik_partial_tip_tip(cap, c_n5, c_s5, 0, 0, 1, 1)) die("cap tip_tip");
        if (iqtree_gpu_lik_partial_tip_tip(cap, c_n6, c_s6, 3, 3, 4, 4)) die("cap tip_tip2");
        if (iqtree_gpu_lik_partial_tip_internal(cap, c_n7, c_s7, 2, 2, 6, c_n5, c_s5))
            die("cap tip_internal");
        if (iqtree_gpu_lik_partial_internal_internal(cap, c_n8, c_s8, 6, c_n5, c_s5,
                                                     7, c_n6, c_s6)) die("cap int_int");
        if (iqtree_gpu_lik_theta_internal(cap, c_n7, c_s7, c_n8, c_s8)) die("cap theta");
        if (iqtree_gpu_lik_sync(cap)) die("sync");

        if (iqtree_gpu_lik_download_partial(cap, gpu.data(), c_n5)) die("download");
        if (iqtree_gpu_lik_download_scale(cap, gscale.data(), c_s5)) die("download scale");
        check_arr("cap: partial tip-tip",           gpu, ref_n5, 1e-12);
        check_scale("cap: scale tip-tip",           gscale, ref_s5);

        if (iqtree_gpu_lik_download_partial(cap, gpu.data(), c_n7)) die("download");
        if (iqtree_gpu_lik_download_scale(cap, gscale.data(), c_s7)) die("download scale");
        check_arr("cap: partial tip-internal",      gpu, ref_n7, 1e-12);
        check_scale("cap: scale tip-internal",      gscale, ref_s7);

        if (iqtree_gpu_lik_download_partial(cap, gpu.data(), c_n8)) die("download");
        if (iqtree_gpu_lik_download_scale(cap, gscale.data(), c_s8)) die("download scale");
        check_arr("cap: partial internal-internal", gpu, ref_n8, 1e-12);
        check_scale("cap: scale internal-internal", gscale, ref_s8);

        if (iqtree_gpu_lik_download_theta(cap, gpu.data())) die("download theta");
        if (iqtree_gpu_lik_download_buffer_scale(cap, gbuf.data())) die("download bufscale");
        check_arr("cap: theta",        gpu,  ref_theta_ii, 1e-12);
        check_arr("cap: buffer_scale", gbuf, ref_buf_ii,   1e-12);

        {
            const double t = 0.137;
            double g_logl = 0.0, g_l2 = 0.0, g_df = 0.0, g_ddf = 0.0;
            if (iqtree_gpu_lik_branch_logl(cap, t, &g_logl)) die("cap branch_logl");
            check_scalar("cap: branch logL", g_logl,
                         ref_branch_logl<NST>(r, ref_theta_ii, ref_buf_ii, t), 1e-12);
            if (iqtree_gpu_lik_derivatives(cap, t, &g_l2, &g_df, &g_ddf)) die("cap derivatives");
            double c_l2, c_df, c_ddf;
            ref_derivatives<NST>(r, ref_theta_ii, ref_buf_ii, t, &c_l2, &c_df, &c_ddf);
            check_scalar("cap: derivative-path logL", g_l2,  c_l2,  1e-12);
            check_scalar("cap: df (analytic)",        g_df,  c_df,  1e-12);
            check_scalar("cap: ddf (analytic)",       g_ddf, c_ddf, 1e-12);
        }

        iqtree_gpu_lik_free(cap, c_n5); iqtree_gpu_lik_free(cap, c_s5);
        iqtree_gpu_lik_free(cap, c_n6); iqtree_gpu_lik_free(cap, c_s6);
        iqtree_gpu_lik_free(cap, c_n7); iqtree_gpu_lik_free(cap, c_s7);
        iqtree_gpu_lik_free(cap, c_n8); iqtree_gpu_lik_free(cap, c_s8);
        iqtree_gpu_lik_destroy(cap);
    }

    /* ================================================================================
     * REBIND: one arena, two trees
     *
     * Every allocation iqtree_gpu_lik_create() makes is a function of GEOMETRY alone -- npat,
     * ncat_cap, nstates, safe_numeric, echild_slots, tip_slots, num_tip_states. Nothing in it
     * depends on WHICH tree the numbers came from. iqtree_gpu_lik_rebind() detaches the tree
     * binding and keeps the arena, which is what stops ModelFinder's per-candidate `new IQTree`
     * from costing a multi-GB cudaFree + cudaMalloc + cudaMemset each time.
     *
     * The test is adversarial in the same way the capacity test is: the engine is first driven
     * hard as "tree A" -- different branch lengths, different rate and proportion vectors, every
     * slot, every echild slot and theta all left holding tree-A values -- and only then rebound
     * and re-run as "tree B", which is the SAME configuration the freshly-created engine at the
     * top of this file was validated against. Reusing those references is the point: if anything
     * of tree A survived into a buffer tree B reads, the comparison fails.
     *
     * The pooled partial/scale buffers are deliberately NOT reallocated between the two, because
     * that is exactly what the integration layer does; the caller's obligation to invalidate its
     * own "this slot is current" bookkeeping (IQ-TREE: clearAllPartialLH) is modelled here by
     * simply recomputing every slot after the rebind, which is what that invalidation forces.
     * ============================================================================== */
    printf("\n--- rebind (one arena, two trees) ---\n");
    {
        IqtreeGpuLik *rb = NULL;
        if (iqtree_gpu_lik_create(&rb, &cfg)) die("create rebind");   /* NORM_LH, NCAT cats */

        double        *b_n5 = iqtree_gpu_lik_alloc_partial(rb);
        unsigned char *b_s5 = iqtree_gpu_lik_alloc_scale(rb);
        double        *b_n6 = iqtree_gpu_lik_alloc_partial(rb);
        unsigned char *b_s6 = iqtree_gpu_lik_alloc_scale(rb);
        double        *b_n7 = iqtree_gpu_lik_alloc_partial(rb);
        unsigned char *b_s7 = iqtree_gpu_lik_alloc_scale(rb);
        double        *b_n8 = iqtree_gpu_lik_alloc_partial(rb);
        unsigned char *b_s8 = iqtree_gpu_lik_alloc_scale(rb);
        if (!b_n5 || !b_s5 || !b_n6 || !b_s6 || !b_n7 || !b_s7 || !b_n8 || !b_s8)
            die("alloc rebind");

        /* ---- tree A: a different topology's worth of branch lengths and a different rate
               model, so that every device buffer the rebind is supposed to clear holds
               distinctive, non-zero data. ---- */
        double blen_a[8] = { 0.53, 0.41, 0.67, 0.29, 0.71, 0.37, 0.61, 0.47 };
        vector<double> rates_a(NCAT), props_a(NCAT);
        for (int c = 0; c < NCAT; c++) { rates_a[c] = 1.7 - 0.3 * c; props_a[c] = 0.8 / NCAT; }

        if (iqtree_gpu_lik_set_eigen(rb, r.eval.data(), r.evec.data(), r.inv_evec.data()))
            die("rb set_eigen A");
        if (iqtree_gpu_lik_set_rates(rb, rates_a.data(), props_a.data())) die("rb set_rates A");
        if (iqtree_gpu_lik_set_tip_partials(rb, r.tip_lh.data())) die("rb set_tip_partials A");
        if (iqtree_gpu_lik_set_pattern_data(rb, r.ptn_freq.data(), r.ptn_invar.data()))
            die("rb set_pattern_data A");
        for (int t = 0; t < NTIP; t++)
            if (iqtree_gpu_lik_set_tip_states(rb, t, states[t].data())) die("rb set_tip_states");
        for (int s = 0; s < 8; s++)
            if (iqtree_gpu_lik_branch_precompute(rb, s, blen_a[s], 1)) die("rb precompute A");
        if (iqtree_gpu_lik_partial_tip_tip(rb, b_n5, b_s5, 0, 0, 1, 1)) die("rb tip_tip A");
        if (iqtree_gpu_lik_partial_tip_tip(rb, b_n6, b_s6, 3, 3, 4, 4)) die("rb tip_tip2 A");
        if (iqtree_gpu_lik_partial_tip_internal(rb, b_n7, b_s7, 2, 2, 6, b_n5, b_s5))
            die("rb tip_internal A");
        if (iqtree_gpu_lik_partial_internal_internal(rb, b_n8, b_s8, 6, b_n5, b_s5,
                                                     7, b_n6, b_s6)) die("rb int_int A");
        if (iqtree_gpu_lik_theta_internal(rb, b_n7, b_s7, b_n8, b_s8)) die("rb theta A");
        if (iqtree_gpu_lik_sync(rb)) die("sync");

        /* Prove tree A actually left something behind: theta must be non-zero going in, or the
           "theta is zeroed" check below would pass vacuously. */
        {
            if (iqtree_gpu_lik_download_theta(rb, gpu.data())) die("rb download theta A");
            size_t nz = 0;
            for (size_t i = 0; i < PE; i++) if (gpu[i] != 0.0) nz++;
            if (nz == 0) {
                printf("  [FAIL] tree A left theta all-zero; the rebind checks would be vacuous\n");
                g_failures++;
            } else {
                printf("  [info] tree A left %zu of %zu theta entries non-zero\n", nz, PE);
            }
        }

        /* ---- the rebind itself. No free, no allocation. ---- */
        if (iqtree_gpu_lik_rebind(rb)) die("rebind");

        /* (a) the rate/proportion vectors are per-TREE data as far as the caller is concerned
               (a new tree arrives with a new model), so rebind drops them and the existing
               rates_ready guard must refuse every kernel that reads them. Loud, not silent. */
        {
            double dummy = 0.0;
            if (iqtree_gpu_lik_branch_logl(rb, 0.1, &dummy) == 0) {
                printf("  [FAIL] stale rate vectors were not caught after rebind\n");
                g_failures++;
            } else {
                printf("  [ ok ] %-34s %s\n", "rebind stale-rate guard",
                       iqtree_gpu_lik_last_error());
            }
        }

        /* (b) theta and buffer_scale describe one branch of tree A. They must be gone, so that a
               caller that forgets to clear its theta_computed flag gets log(0) = -inf rather
               than a plausible number from the previous tree. */
        {
            if (iqtree_gpu_lik_download_theta(rb, gpu.data())) die("rb download theta");
            if (iqtree_gpu_lik_download_buffer_scale(rb, gbuf.data())) die("rb download bufscale");
            size_t bad = 0;
            for (size_t i = 0; i < PE; i++) if (gpu[i] != 0.0) bad++;
            for (int p = 0; p < r.stride; p++) if (gbuf[p] != 0.0) bad++;
            if (bad) {
                printf("  [FAIL] rebind left %zu non-zero theta/buffer_scale entries\n", bad);
                g_failures++;
            } else {
                printf("  [ ok ] %-34s theta and buffer_scale zeroed\n", "rebind invalidation");
            }
        }

        /* ---- tree B: the reference configuration. Same arena, same slot buffers, nothing
               reallocated -- and every slot recomputed, which is what the caller's
               clearAllPartialLH() forces on the real path. ---- */
        if (iqtree_gpu_lik_set_eigen(rb, r.eval.data(), r.evec.data(), r.inv_evec.data()))
            die("rb set_eigen B");
        if (iqtree_gpu_lik_set_rates(rb, r.rates.data(), r.props.data())) die("rb set_rates B");
        if (iqtree_gpu_lik_set_tip_partials(rb, r.tip_lh.data())) die("rb set_tip_partials B");
        if (iqtree_gpu_lik_set_pattern_data(rb, r.ptn_freq.data(), r.ptn_invar.data()))
            die("rb set_pattern_data B");
        for (int s = 0; s < 8; s++)
            if (iqtree_gpu_lik_branch_precompute(rb, s, blen[s], 1)) die("rb precompute B");
        if (iqtree_gpu_lik_partial_tip_tip(rb, b_n5, b_s5, 0, 0, 1, 1)) die("rb tip_tip B");
        if (iqtree_gpu_lik_partial_tip_tip(rb, b_n6, b_s6, 3, 3, 4, 4)) die("rb tip_tip2 B");
        if (iqtree_gpu_lik_partial_tip_internal(rb, b_n7, b_s7, 2, 2, 6, b_n5, b_s5))
            die("rb tip_internal B");
        if (iqtree_gpu_lik_partial_internal_internal(rb, b_n8, b_s8, 6, b_n5, b_s5,
                                                     7, b_n6, b_s6)) die("rb int_int B");
        if (iqtree_gpu_lik_theta_internal(rb, b_n7, b_s7, b_n8, b_s8)) die("rb theta B");
        if (iqtree_gpu_lik_sync(rb)) die("sync");

        if (iqtree_gpu_lik_download_partial(rb, gpu.data(), b_n5)) die("download");
        if (iqtree_gpu_lik_download_scale(rb, gscale.data(), b_s5)) die("download scale");
        check_arr("rebind: partial tip-tip",           gpu, ref_n5, 1e-12);
        check_scale("rebind: scale tip-tip",           gscale, ref_s5);

        if (iqtree_gpu_lik_download_partial(rb, gpu.data(), b_n7)) die("download");
        if (iqtree_gpu_lik_download_scale(rb, gscale.data(), b_s7)) die("download scale");
        check_arr("rebind: partial tip-internal",      gpu, ref_n7, 1e-12);
        check_scale("rebind: scale tip-internal",      gscale, ref_s7);

        if (iqtree_gpu_lik_download_partial(rb, gpu.data(), b_n8)) die("download");
        if (iqtree_gpu_lik_download_scale(rb, gscale.data(), b_s8)) die("download scale");
        check_arr("rebind: partial internal-internal", gpu, ref_n8, 1e-12);
        check_scale("rebind: scale internal-internal", gscale, ref_s8);

        if (iqtree_gpu_lik_download_theta(rb, gpu.data())) die("download theta");
        if (iqtree_gpu_lik_download_buffer_scale(rb, gbuf.data())) die("download bufscale");
        check_arr("rebind: theta",        gpu,  ref_theta_ii, 1e-12);
        check_arr("rebind: buffer_scale", gbuf, ref_buf_ii,   1e-12);

        {
            const double t = 0.137;
            double g_logl = 0.0, g_l2 = 0.0, g_df = 0.0, g_ddf = 0.0;
            if (iqtree_gpu_lik_branch_logl(rb, t, &g_logl)) die("rb branch_logl");
            check_scalar("rebind: branch logL", g_logl,
                         ref_branch_logl<NST>(r, ref_theta_ii, ref_buf_ii, t), 1e-12);
            if (iqtree_gpu_lik_derivatives(rb, t, &g_l2, &g_df, &g_ddf)) die("rb derivatives");
            double c_l2, c_df, c_ddf;
            ref_derivatives<NST>(r, ref_theta_ii, ref_buf_ii, t, &c_l2, &c_df, &c_ddf);
            check_scalar("rebind: derivative-path logL", g_l2,  c_l2,  1e-12);
            check_scalar("rebind: df (analytic)",        g_df,  c_df,  1e-12);
            check_scalar("rebind: ddf (analytic)",       g_ddf, c_ddf, 1e-12);
        }

        /* (c) rebind must also survive a category change on the same arena, since the
               integration layer can hit both at once (a new ModelFinder candidate with a new
               +R{k}). Drop to 2 categories, rebind, and confirm the engine still refuses to run
               on the dropped rate vectors -- the loud path is the one that must not regress. */
        {
            if (iqtree_gpu_lik_set_ncat(rb, 2)) die("rb set_ncat 2");
            if (iqtree_gpu_lik_rebind(rb)) die("rb rebind after set_ncat");
            double dummy = 0.0;
            if (iqtree_gpu_lik_ncat(rb) != 2) {
                printf("  [FAIL] rebind disturbed the category count\n");
                g_failures++;
            } else if (iqtree_gpu_lik_branch_logl(rb, 0.1, &dummy) == 0) {
                printf("  [FAIL] stale rate vectors were not caught after set_ncat + rebind\n");
                g_failures++;
            } else {
                printf("  [ ok ] %-34s ncat kept at %d, rates still stale\n",
                       "rebind after set_ncat", iqtree_gpu_lik_ncat(rb));
            }
        }

        iqtree_gpu_lik_free(rb, b_n5); iqtree_gpu_lik_free(rb, b_s5);
        iqtree_gpu_lik_free(rb, b_n6); iqtree_gpu_lik_free(rb, b_s6);
        iqtree_gpu_lik_free(rb, b_n7); iqtree_gpu_lik_free(rb, b_s7);
        iqtree_gpu_lik_free(rb, b_n8); iqtree_gpu_lik_free(rb, b_s8);
        iqtree_gpu_lik_destroy(rb);
    }

    /* (d) a G matrix cannot be rebound: its rows are indexed by the branch ids of ONE tree.
           The engine must refuse and leave the context exactly as it was, so that the
           integration layer's "--dating mcmctree always rebuilds" rule has a second line of
           defence in the engine itself. */
    {
        IqtreeGpuLik *gm = NULL;
        if (iqtree_gpu_lik_create(&gm, &cfg)) die("create rebind/gmat");
        if (iqtree_gpu_lik_gmatrix_enable(gm, 4, r.stride)) die("gmatrix_enable");
        if (iqtree_gpu_lik_rebind(gm) == 0) {
            printf("  [FAIL] rebind accepted a context with an enabled G matrix\n");
            g_failures++;
        } else {
            printf("  [ ok ] %-34s %s\n", "rebind refuses a G matrix",
                   iqtree_gpu_lik_last_error());
        }
        iqtree_gpu_lik_destroy(gm);
    }

    /* ================================================================================
     * SAFE_LH: one counter per (pattern, category), and the theta-time renormalisation
     * to the minimum scale count with hard-zeroing of anything more than one step below.
     * ============================================================================== */
    printf("\n--- SAFE_LH (-safe kernel) ---\n");
    r.safe = 1;
    cfg.safe_numeric = 1;
    if (iqtree_gpu_lik_create(&ctx, &cfg)) die("create safe");
    if (iqtree_gpu_lik_set_eigen(ctx, r.eval.data(), r.evec.data(), r.inv_evec.data())) die("set_eigen");
    if (iqtree_gpu_lik_set_rates(ctx, r.rates.data(), r.props.data())) die("set_rates");
    if (iqtree_gpu_lik_set_tip_partials(ctx, r.tip_lh.data())) die("set_tip_partials");
    if (iqtree_gpu_lik_set_pattern_data(ctx, r.ptn_freq.data(), r.ptn_invar.data())) die("set_pattern_data");
    for (int t = 0; t < NTIP; t++)
        if (iqtree_gpu_lik_set_tip_states(ctx, t, states[t].data())) die("set_tip_states");
    for (int s = 0; s < 8; s++)
        if (iqtree_gpu_lik_branch_precompute(ctx, s, blen[s], 1)) die("branch_precompute");

    const size_t SE2 = r.selems();
    gscale.assign(SE2, 0);

    double *e_n5 = iqtree_gpu_lik_alloc_partial(ctx);
    unsigned char *e_s5 = iqtree_gpu_lik_alloc_scale(ctx);
    double *e_n7 = iqtree_gpu_lik_alloc_partial(ctx);
    unsigned char *e_s7 = iqtree_gpu_lik_alloc_scale(ctx);
    if (iqtree_gpu_lik_partial_tip_tip(ctx, e_n5, e_s5, 0, 0, 1, 1)) die("safe tip_tip");
    if (iqtree_gpu_lik_partial_tip_internal(ctx, e_n7, e_s7, 2, 2, 6, e_n5, e_s5)) die("safe tip_internal");
    if (iqtree_gpu_lik_sync(ctx)) die("sync");

    vector<unsigned char> ref_e5(SE2, 0), ref_e7(SE2, 0);
    vector<double> ref_e_n5, ref_e_n7;
    ref_partial<NST>(r, true, leaf[0], states_full[0], nil, nil,
                   true, leaf[1], states_full[1], nil, nil, ref_e_n5);
    ref_partial<NST>(r, true,  leaf[2], states_full[2], nil,    nil,
                   false, nil,     nili,           ech[6], ref_e_n5, ref_e_n7);
    ref_rescale<NST>(r, ref_e_n7, ref_e7);

    if (iqtree_gpu_lik_download_partial(ctx, gpu.data(), e_n7)) die("download");
    if (iqtree_gpu_lik_download_scale(ctx, gscale.data(), e_s7)) die("download scale");
    check_arr("safe partial tip-internal", gpu, ref_e_n7, 1e-12);
    check_scale("safe scale tip-internal", gscale, ref_e7);

    /* Hand-set per-category counters so that every branch of the renormalisation is taken:
       one category at the minimum, one exactly one step above (scaled by 2^-256), and one
       more than one step above (hard-zeroed). */
    {
        vector<unsigned char> sa(SE2), sb(SE2, 0);
        for (int c = 0; c < NCAT; c++)
            for (int p = 0; p < r.stride; p++)
                sa[(size_t)c * r.stride + p] = (unsigned char)((p + c) % 4);   /* 0..3 */
        if (iqtree_gpu_lik_upload_scale(ctx, e_s5, sa.data())) die("upload scale");
        if (iqtree_gpu_lik_upload_scale(ctx, e_s7, sb.data())) die("upload scale");
        if (iqtree_gpu_lik_upload_partial(ctx, e_n5, ref_e_n5.data())) die("upload");
        if (iqtree_gpu_lik_upload_partial(ctx, e_n7, ref_e_n7.data())) die("upload");

        /* node = n5 (counters sa, spread over 0..3), dad = n7 (counters sb, all zero). */
        if (iqtree_gpu_lik_theta_internal(ctx, e_n5, e_s5, e_n7, e_s7)) die("safe theta");
        if (iqtree_gpu_lik_sync(ctx)) die("sync");

        vector<double> th, bs;
        /* dad = n7 (counters sb, all zero), node = n5 (counters sa). */
        ref_theta<NST>(r, ref_e_n5, ref_e_n7, sa.data(), sb.data(), th, bs);

        if (iqtree_gpu_lik_download_theta(ctx, gpu.data())) die("download theta");
        if (iqtree_gpu_lik_download_buffer_scale(ctx, gbuf.data())) die("download bufscale");
        check_arr("safe theta (min renormalise)", gpu, th, 1e-12);
        check_arr("safe buffer_scale", gbuf, bs, 1e-12);

        int nzeroed = 0;
        for (int c = 0; c < NCAT; c++)
            for (int p = 0; p < r.stride; p++)
                if (th[((size_t)(c * NST)) * r.stride + p] == 0.0
                    && ref_e_n5[((size_t)(c * NST)) * r.stride + p] != 0.0) nzeroed++;
        printf("  [info] %d (pattern,category) cells hard-zeroed by the min renormalisation\n",
               nzeroed);
        if (nzeroed == 0) {
            printf("  [FAIL] hard-zero branch of the renormalisation never taken\n");
            g_failures++;
        }
    }

    iqtree_gpu_lik_free(ctx, e_n5); iqtree_gpu_lik_free(ctx, e_s5);
    iqtree_gpu_lik_free(ctx, e_n7); iqtree_gpu_lik_free(ctx, e_s7);
    iqtree_gpu_lik_destroy(ctx);
}

int main(void) {
    if (iqtree_gpu_device_count() < 1) {
        printf("No CUDA device visible -- nothing to test.\n");
        return 77;   /* automake convention for "skipped" */
    }
    printf("Device: %s\n", iqtree_gpu_device_name());

    /* Protein: 100 patterns (stride 128), 24 tip rows -- the configuration every existing
       amino-acid result in this repository was validated against. Unchanged, deliberately: DNA
       support must not cost AA coverage. */
    run_suite<20>("amino acid", /*NPAT=*/100, /*NROWS=*/24, /*UNKNOWN=*/23);

    /* DNA: 183 patterns, the real distinct-pattern count of benchmark/data/dna/dna_n04 (195 456 sites
       collapsing to 183 patterns, because the number of distinct DNA site patterns is bounded
       by 4^n and n = 4). Stride 192, so nine of 192 lanes are padding -- 4.7%, far above any
       protein case, which is exactly where a padding bug is visible. 19 tip rows. */
    run_suite<4>("DNA", /*NPAT=*/183, /*NROWS=*/19, /*UNKNOWN=*/18);

    printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL CHECKS PASSED",
           g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
