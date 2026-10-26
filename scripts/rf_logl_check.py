#!/usr/bin/env python3
"""Dependency-free science-exactness check for two IQ-TREE runs (system python3, NO dendropy).

Computes the unrooted Robinson-Foulds distance between two .treefile Newick trees from their
nontrivial bipartitions, and compares the log-likelihood parsed from the two .iqtree reports.
PASS  <=>  RF == 0  AND  |logL_a - logL_b| / |logL_a| <= rtol (default 1e-4).

Usage: rf_logl_check.py <prefixA> <prefixB> [label] [rtol]
  expects <prefix>.treefile and <prefix>.iqtree for each. Prints a one-line VERDICT and appends a
  TSV row to $RF_LOGL_TSV if that env var is set.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import sys, os, re


def _leaves_and_splits(newick):
    """Return (taxon_set, set_of_canonical_nontrivial_bipartitions) for an unrooted tree.
    A bipartition is the frozenset of leaves on the side NOT containing the lexicographically first
    taxon; trivial splits (size 1 or n-1) are dropped. Branch lengths / support values are ignored."""
    s = newick.strip()
    if s.endswith(";"):
        s = s[:-1]
    i = 0
    all_leaves = []

    def parse():
        nonlocal i
        # returns list of leaf-name-sets for each subtree rooted here; also records internal splits
        clade = set()
        if s[i] == "(":
            i += 1  # consume '('
            while True:
                child = parse()
                clade |= child
                if s[i] == ",":
                    i += 1
                    continue
                if s[i] == ")":
                    i += 1
                    break
            # consume optional internal label / support and branch length
            while i < len(s) and s[i] not in ",()":
                i += 1
            splits.append(frozenset(clade))
            return clade
        else:
            # leaf name: read until a structural char
            j = i
            while i < len(s) and s[i] not in ",()":
                i += 1
            name = s[j:i]
            # strip a trailing :branchlength if present
            name = name.split(":")[0]
            all_leaves.append(name)
            return {name}

    splits = []
    parse()
    taxa = set(all_leaves)
    n = len(taxa)
    first = sorted(taxa)[0]
    canon = set()
    for cl in splits:
        if len(cl) <= 1 or len(cl) >= n - 1:
            continue  # trivial
        side = cl if first not in cl else frozenset(taxa - cl)
        canon.add(side)
    return taxa, canon


def rf_distance(fa, fb):
    """RF distance, or (None, reason). NEVER raises.

    A crashed or timed-out IQ-TREE run leaves a missing, empty or truncated .treefile. Previously
    those cases escaped as an IndexError/FileNotFoundError traceback and NO 'VERDICT' line was
    printed at all -- so a failed cell looked like a gap rather than a failure, and anything
    grepping for VERDICT skipped it silently. Every failure mode must produce a FAIL verdict.
    """
    def load(f):
        if not os.path.exists(f):
            raise ValueError(f"missing {os.path.basename(f)}")
        txt = open(f, errors="ignore").read().strip()
        if not txt:
            raise ValueError(f"empty {os.path.basename(f)}")
        return _leaves_and_splits(txt)
    try:
        ta, sa = load(fa)
        tb, sb = load(fb)
    except ValueError as e:
        return None, str(e)
    except (IndexError, RecursionError):
        return None, "malformed-newick"
    if not ta or not tb:
        return None, "no-taxa-parsed"
    if ta != tb:
        return None, "taxon-set-mismatch"
    return len(sa ^ sb), None


def parse_logl(iqfile):
    if not os.path.exists(iqfile):
        return None
    t = open(iqfile, errors="ignore").read()
    m = re.search(r"Log-likelihood of the tree:\s*(-?[0-9.]+)", t)
    if not m:
        m = re.search(r"BEST SCORE FOUND :\s*(-?[0-9.]+)", t)
    return float(m.group(1)) if m else None


def main():
    if len(sys.argv) < 3:
        print("usage: rf_logl_check.py <prefixA> <prefixB> [label] [rtol]"); sys.exit(2)
    pa, pb = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "cmp"
    rtol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    rf, err = rf_distance(pa + ".treefile", pb + ".treefile")
    la, lb = parse_logl(pa + ".iqtree"), parse_logl(pb + ".iqtree")
    if err:
        print(f"VERDICT[{label}]: FAIL ({err})   logl_a={la}  logl_b={lb}"); verdict = "FAIL"
        reldiff = None
    else:
        # `is not None`, not truthiness: a log-likelihood of exactly 0.0 is falsy and would be
        # misread as "failed to parse". Guard the divisor separately for the same reason.
        have_both = (la is not None and lb is not None)
        reldiff = abs(la - lb) / abs(la) if (have_both and la != 0.0) else None
        logl_ok = (reldiff is not None and reldiff <= rtol)
        ok = (rf == 0 and logl_ok)
        verdict = "PASS" if ok else "FAIL"
        print(f"VERDICT[{label}]: {verdict}   rf_dist={rf}   "
              f"logl_a={la}  logl_b={lb}  logl_reldiff={reldiff:.2e}" if reldiff is not None
              else f"VERDICT[{label}]: {verdict}   rf_dist={rf}   logl_a={la} logl_b={lb}")
    tsv = os.environ.get("RF_LOGL_TSV")
    if tsv:
        newfile = not os.path.exists(tsv)
        with open(tsv, "a") as f:
            if newfile:
                f.write("label\tverdict\trf_dist\tlogl_a\tlogl_b\tlogl_reldiff\tprefixA\tprefixB\n")
            f.write(f"{label}\t{verdict}\t{rf if not err else 'NA'}\t{la}\t{lb}\t{reldiff}\t{pa}\t{pb}\n")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
