"""figdata.py -- the shared analysis layer behind every published panel.

WHY THIS EXISTS. The board requires that a single treatment of the gate-failure cell be applied
identically in every panel, that every fitted exponent carry n and a bootstrap CI, and that the
crossover be quoted with a named estimator. Each of those is a decision, not a detail, and a decision
re-implemented per panel is a decision that will eventually differ between panels. They live here.

THE GATE-FAILURE CELL. `g__Ruminococcoides__T20__l40s_cx` fails the acceptance gate on the SUPPORT
criterion itself (6.0 points against a 5-point tolerance); its topology is identical (RF = 0) and its
log-likelihood agrees to 3.611e-07. The treatment, per the board:

    excluded from every FIT and every SPEEDUP SUMMARY,
    but RETAINED AND LABELLED in every EQUIVALENCE panel.

Excluding it from the equivalence figures would remove the worst case from the figures whose purpose
is to show worst cases, which would be circular. Use `for_fits()` and `for_equivalence()` and the
distinction is made for you. Note also that this cell is the only row whose `peak_ram_mb` is 0 -- a
monitoring failure, NOT a gate input -- which is why it is absent from the host-RAM panel.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import csv, math, os, random, re

GATE_FAILURE_CELL = "g__Ruminococcoides__T20__l40s_cx"
SUPERSEDED_SHA_PREFIX = "07c70cae"
SUPPORT_GATE_TOL = 5.0          # points; matches paired_benchmark.sh:399
LOGL_CONTRACT = 1e-4            # matches paired_benchmark.sh:477


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def num(x):
    try:
        v = float(x)
        return v if v == v else None       # reject NaN
    except (TypeError, ValueError):
        return None


def rate_cats(model):
    """Rate categories from the model string; 4 when heterogeneity is present without a count."""
    m = re.search(r"\+(?:R|G)(\d+)", model or "")
    if m:
        return int(m.group(1))
    return 4 if re.search(r"\+(?:R|G|I)", model or "") else 1


def complexity(r):
    """branches(2n-3) x patterns x rate categories x states^2 -- one pruning pass.

    NOTE the k^2 weighting is a THEORETICAL operation count used as an axis, not a fact these data
    establish: refitting the cost law with k^p prefers p=1 (CPU arm) or p=0 (GPU arm) over p=2. Any
    conclusion drawn on this axis is conditional on this definition and must say so.
    """
    t, p, k = num(r.get("taxa")), num(r.get("patterns")), num(r.get("num_states"))
    if not (t and p and k):
        return None
    return (2 * t - 3) * p * rate_cats(r.get("model")) * k * k


def work_units(r):
    """invocations x per-invocation work (m k^2 r) -- the abscissa of the COST LAW (Fig. 3a).

    NOT the same quantity as `complexity()`. `complexity()` counts one pruning pass over the tree
    (it carries the 2n-3 branch term and no invocation count) and is the abscissa of the DECISION
    figure; `work_units()` multiplies the per-invocation work by how many invocations the search
    actually performed, and is the abscissa of the cost law. Fitting wall time against the two gives
    different exponents, so a number quoted from one must never be attributed to the other -- which
    is exactly the mix-up this shared definition exists to prevent.

    The k^2 weighting is a theoretical operation count used as an axis, NOT a fact these data
    establish; see the k^p comparison in Methods.
    """
    gc, m, k = num(r.get("gpu_calls")), num(r.get("patterns")), num(r.get("num_states"))
    if not (gc and m and k):
        return None
    return gc * m * k * k * rate_cats(r.get("model"))


def load(mod, current_build_only=True):
    """All rows, optionally restricted to the build the manuscript reports."""
    with open(os.path.join(mod, "benchmark", "runs", "RUNS.tsv")) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if current_build_only:
        rows = [r for r in rows
                if not (r.get("binary_sha256") or "").startswith(SUPERSEDED_SHA_PREFIX)]
    for r in rows:
        r["_complexity"] = complexity(r)
        r["_speedup"] = num(r.get("speedup"))
        r["_workflow"] = workflow(r)
        r["_work_units"] = work_units(r)
    return rows


_KEYS = {}


def key_number(mod, key, default=None):
    """Read one value out of KEY_NUMBERS.tsv.

    Panel titles used to hardcode numbers -- "60x", "1.67x", "19x in fp64", "6 runs" -- that no
    table held, so nothing could check them and three of them were wrong. A panel that wants to
    state a number now has to name the key it comes from.
    """
    if not _KEYS:
        p = os.path.join(mod, "benchmark", "runs", "KEY_NUMBERS.tsv")
        with open(p) as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) >= 2:
                    _KEYS[row[0]] = row[1]
    v = _KEYS.get(key, default)
    if v is None:
        raise KeyError(f"KEY_NUMBERS.tsv has no key {key!r}")
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def is_gate_failure(r):
    return r.get("cell") == GATE_FAILURE_CELL


def is_correctness_cell(r):
    """`__mdl_*` cells test model generality on the cheapest alignments; they sit below break-even
    by construction and would bias any speedup distribution downward."""
    return "__mdl_" in (r.get("cell") or "")


def for_fits(rows):
    """Rows admissible to a fit or a speedup summary: gate passed, not a correctness cell."""
    return [r for r in rows if not is_gate_failure(r) and not is_correctness_cell(r)]


# --------------------------------------------------------------------------------------------
# run strata -- defined ONCE, because they were not
# --------------------------------------------------------------------------------------------
# Every `threads == "20"` and `card == "NVIDIA_L40S"` filter used to be written inline in each
# panel generator AND again, independently, in key_numbers.py. The two copies drifted: the
# outer-loop fit was computed over all thread counts for the key numbers (n=91/84) and over the
# 20-thread stratum for the panel (n=78/67), so the manuscript quoted one and the figure showed the
# other. A round-6 review found nine such disagreements. The strata live here now, and both the
# panels and the key numbers call these, so a stratum cannot be changed in one place only.
HEADLINE_CARD = "NVIDIA_L40S"     # the card the headline campaign ran on
HEADLINE_THREADS = "20"           # the thread count the manuscript reports throughout


def at_threads(rows, t=HEADLINE_THREADS):
    """The thread stratum. Default is the 20-thread set the manuscript reports."""
    return [r for r in rows if r.get("threads") == t]


def on_headline_card(rows, card=HEADLINE_CARD):
    return [r for r in rows if r.get("card") == card]


def shared_card_datasets(rows, alphabet="protein", min_cards=6):
    """Genera measured on at least `min_cards` distinct cards, within one alphabet.

    `alphabet=None` pools both, which is what the energy panel did implicitly -- it filtered on
    threads but not on data_type, so it reported 8 genera while the text said four. Callers now
    have to say which they mean.
    """
    by = {}
    for r in at_threads(rows):
        if alphabet and r.get("data_type") != alphabet:
            continue
        by.setdefault(r.get("card"), set()).add(r.get("genus"))
    if not by:
        return set()
    common = set.intersection(*by.values()) if len(by) >= min_cards else set()
    return {g for g in common if g}


def for_equivalence(rows):
    """Rows admissible to an equivalence panel: EVERYTHING, including the gate failure.

    Removing the worst case from a figure of worst cases would be circular -- see the docstring at the top of this file.
    """
    return list(rows)


# --------------------------------------------------------------------------------------------
# workflow labelling -- defined ONCE, because alphabet and model treatment are perfectly aliased
# --------------------------------------------------------------------------------------------
WORKFLOW_AA_MF = "amino acid + ModelFinder"
WORKFLOW_NT_FIXED = "nucleotide, fixed model"


def workflow(r):
    """The analysis workflow a row belongs to -- the label the manuscript uses INSTEAD of alphabet.

    THE CONFOUND IS TOTAL. Every one of the 85 amino-acid runs uses `-m TESTNEW` (full ModelFinder)
    and every one of the 97 nucleotide runs pins a model, with ZERO exceptions either way. Alphabet
    and model treatment are therefore perfectly aliased and no comparison in these data can separate
    them. Consequently: the crossovers are labelled by workflow, the abstract must not attribute the
    difference to alphabet, and the Discussion states that a fixed-model amino-acid analysis is
    expected to behave like the nucleotide series.

    The nucleotide series is additionally simulated and ungapped, which is aliased with it too.
    """
    dt, ms = r.get("data_type"), r.get("model_search")
    if dt == "protein" and ms == "full_ModelFinder_TESTNEW":
        return WORKFLOW_AA_MF
    if dt == "dna" and ms == "fixed_model":
        return WORKFLOW_NT_FIXED
    return f"{dt}, {ms}"          # no such row exists today; would mean the aliasing had broken


def aliasing_holds(rows):
    """-> (bool, message). Guards the claim above so new data cannot silently falsify it."""
    bad = [r["cell"] for r in rows if workflow(r) not in (WORKFLOW_AA_MF, WORKFLOW_NT_FIXED)]
    if bad:
        return False, (f"{len(bad)} row(s) break the alphabet/workflow aliasing, so the two CAN now "
                       f"be separated and the confound statement must be revised: {bad[:5]}")
    return True, "alphabet and model treatment are perfectly aliased in all %d rows" % len(rows)


# --------------------------------------------------------------------------------------------
# estimators -- each returns its n and a bootstrap CI, because the board requires both
# --------------------------------------------------------------------------------------------
def _ols(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return None, None, None
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a0 = my - b * mx
    ss = sum((c - my) ** 2 for c in y)
    rss = sum((c - (a0 + b * a)) ** 2 for a, c in zip(x, y))
    return b, a0, (1 - rss / ss if ss else float("nan"))


def loglog_fit(pts, B=2000, seed=1):
    """Fit log10(y) ~ a + b log10(x). Returns slope, intercept, R^2, n and a 95% CI on the slope
    from resampling the PAIRED observations (the unit of independence is the paired run)."""
    pts = [(a, c) for a, c in pts if a and c and a > 0 and c > 0]
    if len(pts) < 3:
        return None
    x = [math.log10(a) for a, _ in pts]
    y = [math.log10(c) for _, c in pts]
    b, a0, r2 = _ols(x, y)
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        idx = [rng.randrange(len(pts)) for _ in range(len(pts))]
        bb, _, _ = _ols([x[i] for i in idx], [y[i] for i in idx])
        if bb is not None:
            bs.append(bb)
    bs.sort()
    lo, hi = (bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]) if bs else (None, None)
    return dict(slope=b, intercept=a0, r2=r2, n=len(pts), ci=(lo, hi))


def crossover(pts, B=2000, seed=1):
    """Complexity at which the fitted speedup curve crosses parity.

    ESTIMATOR, named because the board requires it: ordinary least squares of log10(speedup) on
    log10(complexity), crossover taken as the abscissa where the fitted line meets speedup = 1, with
    a 95% interval from 2,000 paired bootstrap resamples.

    This replaces a discrete estimator (geometric midpoint of the parity gap) whose point estimate
    sat on its own lower bound -- an artefact of the estimator's granularity, not a property of the
    data.
    """
    pts = [(a, c) for a, c in pts if a and c and a > 0 and c > 0]
    if len(pts) < 5:
        return None
    x = [math.log10(a) for a, _ in pts]
    y = [math.log10(c) for _, c in pts]
    b, a0, r2 = _ols(x, y)
    if not b:
        return None
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        idx = [rng.randrange(len(pts)) for _ in range(len(pts))]
        bb, aa, _ = _ols([x[i] for i in idx], [y[i] for i in idx])
        if bb:
            bs.append(-aa / bb)
    bs.sort()
    lo, hi = (10 ** bs[int(.025 * len(bs))], 10 ** bs[int(.975 * len(bs))]) if bs else (None, None)
    return dict(crossover=10 ** (-a0 / b), ci=(lo, hi), slope=b, r2=r2, n=len(pts),
                n_below=sum(1 for _, c in pts if c < 1))


# --------------------------------------------------------------------------------------------
# split-aware tree comparison -- validated against the harness's own recorded values
# --------------------------------------------------------------------------------------------
def parse_newick(s):
    """-> (taxa, {canonical_split: support}). Iterative.

    A POSITIONAL comparison of `.contree` support labels is WRONG and will silently invent
    differences: two identical topologies may serialise their splits in a different order. Splits are
    canonicalised here as taxon frozensets with a fixed side, matching the harness.
    """
    s = s.strip()
    stack, cur, splits, leaves = [], [], [], []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == '(':
            stack.append(cur); cur = []; i += 1
        elif c == ',':
            i += 1
        elif c == ')':
            i += 1; j = i
            while i < n and s[i] not in '(),;':
                i += 1
            lab = s[j:i].split(':')[0].strip()
            try:
                sup = float(lab) if lab else None
            except ValueError:
                sup = None
            clade = set().union(*cur) if cur else set()
            splits.append((frozenset(clade), sup))
            parent = stack.pop() if stack else []
            parent.append(clade); cur = parent
        elif c == ';':
            break
        else:
            j = i
            while i < n and s[i] not in '(),;':
                i += 1
            nm = s[j:i].split(':')[0].strip()
            if nm:
                leaves.append(nm); cur.append({nm})
    taxa = set(leaves)
    if not taxa:
        return None, None
    first = sorted(taxa)[0]
    out = {}
    for cl, sup in splits:
        if not cl or cl == taxa:
            continue
        side = cl if first not in cl else frozenset(taxa - cl)
        out[frozenset(side)] = sup
    return taxa, out


def support_max_absdiff(path_a, path_b):
    """Max |support difference| over splits shared by two trees; None if not comparable."""
    try:
        ta, sa = parse_newick(open(path_a).read())
        tb, sb = parse_newick(open(path_b).read())
    except OSError:
        return None
    if not ta or not tb or ta != tb:
        return None
    shared = set(sa) & set(sb)
    d = [abs(sa[k] - sb[k]) for k in shared if sa[k] is not None and sb[k] is not None]
    return max(d) if d else 0.0


def quadratic_fit(pts, B=2000, seed=1):
    """Fit log10(y) ~ a + b u + c u^2 with u = log10(x) - mean(log10 x). The exponent is then LOCAL,
    d log y / d log x = b + 2 c u, and the fit reports it at the SMALLEST and the LARGEST x in the
    data, each with a 95% CI from resampling the paired observations, together with the curvature
    c, R^2, n and the coefficients for drawing the curve.

    WHY. The straight log-log fit (loglog_fit) returns one average exponent, but every cost-law
    series is concave-up on the log-log scale: small jobs scale more gently than large ones and the
    local exponent rises towards 1 at the largest workloads. A single number hides that; the pair
    of end exponents is what Figure 3 reports.
    """
    import numpy as np
    pts = [(a, c) for a, c in pts if a and c and a > 0 and c > 0]
    if len(pts) < 5:
        return None
    x = np.log10([a for a, _ in pts]); y = np.log10([c for _, c in pts])
    m = float(x.mean()); u = x - m
    co = np.polyfit(u, y, 2)
    pred = np.polyval(co, u)
    r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    ulo, uhi = float(u.min()), float(u.max())
    sl = lambda c, v: float(2 * c[0] * v + c[1])
    rng = np.random.default_rng(seed)
    lo, hi = [], []
    for _ in range(B):
        i = rng.integers(0, len(u), len(u))
        if len(set(u[i].tolist())) < 3:
            continue
        cb = np.polyfit(u[i], y[i], 2)
        lo.append(sl(cb, ulo)); hi.append(sl(cb, uhi))
    pct = lambda v: tuple(float(q) for q in np.percentile(v, [2.5, 97.5]))
    return dict(slope_lo=sl(co, ulo), slope_hi=sl(co, uhi), ci_lo=pct(lo), ci_hi=pct(hi),
                curvature=float(co[0]), coef=co, mean=m, xmin=float(10 ** x.min()),
                xmax=float(10 ** x.max()), r2=r2, n=len(pts))


def quadratic_curve(f, k=60):
    """(x, y) of a quadratic_fit over its data range, for drawing."""
    import numpy as np
    xr = np.logspace(math.log10(f["xmin"]), math.log10(f["xmax"]), k)
    return xr, 10 ** np.polyval(f["coef"], np.log10(xr) - f["mean"])


def end_slope_difference(pts_a, pts_b, B=3000, seed=1):
    """95% CIs on (local exponent of a - local exponent of b) at the smallest and the largest x,
    when both quadratic fits share the SAME cells. As in slope_difference, the CELLS are resampled
    and both arms refitted on each resample, because the two arms share the run-to-run variation.

    pts_a and pts_b must be aligned: element i of each is the same cell.
    """
    import numpy as np
    assert len(pts_a) == len(pts_b)
    x = np.log10([a for a, _ in pts_a]); ya = np.log10([b for _, b in pts_a]); yb = np.log10([b for _, b in pts_b])
    m = float(x.mean()); u = x - m; ulo, uhi = float(u.min()), float(u.max())
    sl = lambda c, v: float(2 * c[0] * v + c[1])
    def ends(uu, y1, y2):
        c1, c2 = np.polyfit(uu, y1, 2), np.polyfit(uu, y2, 2)
        return sl(c1, ulo) - sl(c2, ulo), sl(c1, uhi) - sl(c2, uhi)
    d_lo, d_hi = ends(u, ya, yb)
    rng = np.random.default_rng(seed); blo, bhi = [], []
    for _ in range(B):
        i = rng.integers(0, len(u), len(u))
        if len(set(u[i].tolist())) < 3:
            continue
        a, b = ends(u[i], ya[i], yb[i]); blo.append(a); bhi.append(b)
    pct = lambda v: tuple(float(q) for q in np.percentile(v, [2.5, 97.5]))
    return dict(diff_lo=d_lo, diff_hi=d_hi, ci_lo=pct(blo), ci_hi=pct(bhi), n=len(u))


def slope_difference(pts_a, pts_b, B=3000, seed=1):
    """95% CI on (slope_a - slope_b) when both fits share the SAME cells.

    Resample the CELLS, then refit both arms on that resample. Comparing two marginal CIs instead
    is the wrong test and materially misleads here: the marginal intervals for the nucleotide CPU
    and GPU exponents overlap, yet the paired difference excludes zero comfortably, because the two
    arms share the run-to-run variation that the marginal intervals each carry.

    pts_a and pts_b must be aligned: element i of each is the same cell.
    """
    assert len(pts_a) == len(pts_b)
    xa = [math.log10(a) for a, _ in pts_a]
    ya = [math.log10(b) for _, b in pts_a]
    yb = [math.log10(b) for _, b in pts_b]
    def _sl(x, y):
        n = len(x); mx = sum(x) / n; my = sum(y) / n
        sxx = sum((v - mx) ** 2 for v in x)
        return None if not sxx else sum((v - mx) * (w - my) for v, w in zip(x, y)) / sxx
    pt = _sl(xa, ya) - _sl(xa, yb)
    rng = random.Random(seed); d = []
    for _ in range(B):
        idx = [rng.randrange(len(xa)) for _ in range(len(xa))]
        xx = [xa[i] for i in idx]
        s1 = _sl(xx, [ya[i] for i in idx]); s2 = _sl(xx, [yb[i] for i in idx])
        if s1 is not None and s2 is not None:
            d.append(s1 - s2)
    d.sort()
    return dict(diff=pt, ci=(d[int(.025 * len(d))], d[int(.975 * len(d))]), n=len(xa))
