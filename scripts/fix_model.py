#!/usr/bin/env python3
"""Build a FULLY PINNED IQ-TREE model string from a converged .iqtree report. Stdlib only.

WHY THIS EXISTS
---------------
The differential harness compares per-site log-likelihoods between the CPU and GPU kernels. Fixing
the topology with `-te` is not enough: IQ-TREE still OPTIMISES the model parameters, so the two arms
converge to slightly different points and the comparison measures optimiser divergence rather than
kernel arithmetic.

That is not hypothetical. On 2026-07-21 the harness reported per-site differences up to 4.6e-02 for
+R models while the TOTAL tree log-likelihood agreed to 6.5e-08, and the reports showed the arms had
landed on different proportions (0.3459 vs 0.3458). The maths matches exactly: at an optimum
dlogL/dtheta = 0, so a parameter perturbation moves the TOTAL by O(dtheta^2) but each SITE's
contribution by O(dtheta) -- first order per site, second order in total. +G4 has a single parameter
and both arms landed identically, which is why it alone matched to 0.000e+00.

Pinning every free parameter removes that degree of freedom, so any remaining difference is the
kernel.

Two knobs, per IQ-TREE's own parsers:
  +Rk{p1,r1,p2,r2,...}  -> ratefree.cpp:51-60, sets fix_params = 2 (no optimisation at all).
                           NOTE the order is PROPORTION FIRST, then rate, interleaved, 2k values.
  +I{p}                 -> pins the invariable-site proportion.
Empirical +F is computed from the alignment identically in both arms, so it needs no pinning.

For DNA, set TRUE_MODEL_FILE=<path> to pin from the known simulated model instead of the report:
DNA models (GTR) have free exchangeabilities AND estimated base frequencies, and the report prints
frequencies at only 3 decimals -- too coarse to re-pin without perturbing the comparison.

Usage: fix_model.py <converged.iqtree> <base_model>
Prints the pinned model string, or the base model unchanged if there is nothing to pin.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import re, sys, os


def main():
    if len(sys.argv) < 3:
        print("usage: fix_model.py <converged.iqtree> <base_model>", file=sys.stderr)
        sys.exit(2)
    iqfile, base = sys.argv[1], sys.argv[2]

    # ---- DNA: prefer the a-priori TRUE model over anything re-parsed from the report ----------
    # For protein, +F needs no pinning: the exchangeability matrix (LG/WAG/VT...) is FIXED by the
    # model, so only the rate parameters are free. For DNA that is false -- GTR has 5 free
    # exchangeabilities AND estimated base frequencies, all of which the optimiser fits.
    #
    # They cannot be recovered from the .iqtree report either: base frequencies are printed at
    # setprecision(3) (model/modelmarkov.cpp:429-431), so re-pinning from the report injects a
    # ~1e-3 perturbation -- precisely the magnitude that made the AA differential "fail" before the
    # model was pinned (the two arms had landed on 0.3459 vs 0.3458).
    #
    # Our DNA datasets are simulated, so the exact generating model is known a priori. Using it pins
    # every free parameter with no parsing and no precision loss.
    true_model = os.environ.get("TRUE_MODEL_FILE")
    if true_model and os.path.exists(true_model):
        tm = open(true_model).read().strip()
        if tm:
            print(tm)
            return

    if not os.path.exists(iqfile):
        print(base); return                      # nothing to pin from; caller still works
    txt = open(iqfile, errors="ignore").read()

    out = base

    # ---- FreeRate: pin proportions and rates -------------------------------------------------
    m = re.search(r"\+R(\d+)", base)
    if m:
        k = int(m.group(1))
        # The report table is:  Category  Relative_rate  Proportion
        rows = re.findall(r"^\s*(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$", txt, re.M)
        vals = [(int(a), float(b), float(c)) for a, b, c in rows]
        # category 0 is the invariable class when +I is present; keep only the k rate categories
        cats = [v for v in vals if v[0] >= 1][:k]
        if len(cats) == k:
            # interleave PROPORTION first then rate, matching ratefree.cpp
            joined = ",".join(f"{p:.10g},{r:.10g}" for _, r, p in cats)
            out = re.sub(r"\+R\d+", f"+R{k}{{{joined}}}", out)
        else:
            print(f"WARNING: found {len(cats)} of {k} rate categories in {os.path.basename(iqfile)}; "
                  f"not pinning +R", file=sys.stderr)

    # ---- invariable sites --------------------------------------------------------------------
    if "+I" in base:
        mi = re.search(r"Proportion of invariable sites:\s*([0-9.eE+-]+)", txt)
        if mi:
            # only the standalone +I, never the +I inside a model name
            out = re.sub(r"\+I(?![\w{])", f"+I{{{float(mi.group(1)):.10g}}}", out, count=1)
        else:
            print(f"WARNING: no invariable-site proportion in {os.path.basename(iqfile)}",
                  file=sys.stderr)

    print(out)


if __name__ == "__main__":
    main()
