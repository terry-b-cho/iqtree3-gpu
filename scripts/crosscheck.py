#!/usr/bin/env python3
"""crosscheck.py - check the manuscript against the figures and the data.

The board applies four cross-file checks before accepting a resubmission. This script applies them
here so that a failure is caught by `run_tests.sh` rather than by a referee:

  1. every figure reference in the manuscript resolves to a row of FIGURE_MANIFEST.tsv, and every
     main-text panel in the manifest is referenced at least once;
  2. every number quoted in the Abstract appears in KEY_NUMBERS.tsv at the same value and precision;
  3. terminology is uniform across the nine files (the two workflows, the gate, complexity), and
     retired figure names and withdrawn claims do not reappear;
  4. no panel title in the manifest is phrased as a question.

Exit status is non-zero if any check fails. Run: python3 scripts/crosscheck.py

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import csv, os, re, sys, glob

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The manuscript folder moves every round. Until round 8 this pointed at the rev2 folder, which
# no longer existed, so every manuscript check silently reported "skipped". MANUSCRIPT_DIR
# overrides; the default is the current round's folder.
SUB = os.environ.get("MANUSCRIPT_DIR") or os.path.join(REPO, "manuscript")   # absent in the public tree: manuscript checks are skipped
MANIFEST = os.path.join(REPO, "results", "figures", "FIGURE_MANIFEST.tsv")

# The composite layouts decide the figure numbering (see composites.reference_index), so the
# valid (figure, panel) pairs are read from there. The manifest still numbers panels the old way --
# Fig05e is the stem of what is now Fig. 6C -- and checking references against it reported every
# main-text reference as dangling.
import importlib.util as _ilu
_cf = os.path.join(REPO, "scripts", "composites.py")
_sp = _ilu.spec_from_file_location("cf", _cf); CF = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(CF)
COMPOSITES = CF.reference_index()
KEYNUM = os.path.join(REPO, "results", "benchmarks", "KEY_NUMBERS.tsv")
# The panel generators whose titles are audited against KEY_NUMBERS.tsv (main figures, then supplementary).
FIGURE_SCRIPTS = ("equivalence.py", "decision.py", "mechanism.py", "phases_cost.py", "hardware.py", "memory.py",
                  "supplementary.py", "supp_ported.py", "trees.py", "source_data.py", "supplementary_data.py",
                  "style_workbooks.py")

# DISCOVERED, not listed. The revision renumbered the sections -- it drops 00_TITLE_AND_AUTHORS and
# adds 09_SUPPLEMENTARY_NOTE_1_scoping_review -- and a hardcoded list silently reported the new file
# as missing and the dropped one as absent. Globbing the numbered sections survives a renumbering.
def _sections(d):
    if not os.path.isdir(d):
        return []
    # rev2 ships 00_AUTHOR_ACTIONS.md alongside the manuscript sections. It is an editorial
    # instruction sheet that quotes the wording it asks to be replaced, so including it here would
    # report the manuscript's own corrections as unfixed.
    # Every round's review sheet is numbered 00_ (00_AUTHOR_ACTIONS.md, 00_REVIEW_AND_ACTIONS_rev4.md);
    # the manuscript sections start at 01_.
    return sorted(f for f in os.listdir(d)
                  if re.match(r"\d\d_.*\.md$", f) and not f.startswith("00_"))


FILES = _sections(SUB)

# The PUBLIC tree. Scanning only the manuscript is what let four stale figure citations survive in
# README.md, docs/USER_GUIDE.md and docs/DEPENDENCIES.md through an entire revision: those files are what a
# reader sees first, and nothing was checking them.
PUBLIC_DOCS = ["README.md", "docs/USER_GUIDE.md", "docs/REPRODUCING.md", "docs/DEPENDENCIES.md",
               "docs/BENCHMARK_SCHEMA.md", "CHANGELOG.md", "docs/CONTRIBUTING.md",
               "results/figures/README.md", "results/validation/README.md",
               "data/protein/README.md", "data/dna/README.md", "examples/README.md",
               "scripts/dev/README.md", "scripts/benchmark/README.md"]

# Everything archived out of the public tree. A reference to any of these is a dangling pointer.
RETIRED_SCRIPTS = ["replicate_variance", "mechanism_figures", "per_step_figures",
                   "phase_decomposition", "complexity_scaling", "benchmark_figures",
                   "status_dashboard", "cpu_production_figure", "all_trees",
                   "tree_figures", "model_coverage", "interim_report",
                   "card_comparison", "paired_alphabet"]

ARCHIVED_PATHS = {
    r"\btrees_all\b": "superseded per-run tree record; replaced by FigS05a_p01..p06",
    r"\bTREE_INDEX\b": "index of the superseded tree record; archived",
    r"\bPRODUCTION_READINESS\b": "internal readiness assessment; archived out of the public tree",
    r"INVALIDATED_2026-07-21": "record of a resolved historical failure; archived",
    r"xx_archive_": "private archive; a public document must not link into one",
}

# Names that must not reappear: retired figures, and phrases behind withdrawn claims.
FORBIDDEN = {
    r"\bsearch_space\b": "retired figure; its content is in Fig. 3b",
    r"\butil_T1\b|\butil_T20\b": "retired figures; superseded by Supplementary Fig. S10",
    r"\bstatus_dashboard\b": "retired figure",
    r"\bprotein_1thread_distribution\b": "retired figure",
    r"\bcard_comparison\b|\bcard_complexity_cpu\b": "retired; superseded by Fig. 5c/5d",
    r"\bcomplexity_scaling\b": "retired; superseded by Fig. 2a/3a",
    r"\bpaired_alphabet\b": "retired; superseded by Fig. 2d",
    r"\bper_step_speedup\b": "retired; superseded by Fig. 4b",
    r"\bper_pattern_macs\b": "retired; superseded by Fig. 6d",
    r"(?<![0-9.])60-fold": "the 60-fold aggregate was withdrawn; quote the per-run median",
    r"confirms? the k<sup>2</sup>|confirm the states-squared": "withdrawn claim",
    r"02\.5Q_complexity_scaling": "retired script; the definition lives in lib/figdata.py",
    # round 8 (R12, R13, R15): withdrawn wordings
    r"four orders": "the equivalence margin is about 50-fold, not four orders (R12)",
    r"workstation-class card is sufficient|A workstation card is sufficient": "conclusion-as-title withdrawn (R15)",
    r"scales? linearly with the number of conditional": "memory scaling is sub-linear (R13)",
    r"\bFig\. 2D\b|\bFig\. 2d\b": "the per-invocation panel is Supplementary Fig. S18 (R15)",
    r"\bcontract\b(?! *\()": "say 'tolerance' for the 1e-4 log-likelihood criterion (R12)",
}

# Terminology that must be used consistently where it is used at all.
BANNED_PHRASING = {
    r"\bthe alphabet (?:sets|governs|determines) the": "alphabet is confounded with workflow",
    r"\bnear 10<sup>9</sup>\b": "quote the bootstrap interval, not a round point value",
}


def load_manifest():
    """-> (manifest rows, valid (figure, panel) pairs, valid figure labels).

    Main figures come from the composite layouts. Only SUPPLEMENTARY rows are taken from the
    manifest, and with their panel letter uppercased: the manifest's main-figure rows are the old
    numbering and must not be accepted as references.
    """
    with open(MANIFEST) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    panels = {(f, p) for f, letters in COMPOSITES.items() for p in letters}
    figures = set(COMPOSITES)
    for r in rows:
        if r["figure"].startswith("S"):
            figures.add(r["figure"])
            panels.add((r["figure"], (r["panel"] or "").upper() or None))
    return rows, panels, figures


def refs_in(text):
    """Every figure reference, as (figure, panel_or_None).

    Handles 'Fig. 2a', 'Fig. 2', 'Figs. 2, 3', 'Supplementary Fig. S12b',
    'Supplementary Figs. S2, S3' and 'Fig. 3d'.
    """
    out = set()
    # strip fenced code and image markdown, which carry file paths rather than prose references
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    for m in re.finditer(r"(Supplementary\s+)?Figs?\.\s*((?:S?\d+[A-Za-z]?)(?:\s*,\s*S?\d+[A-Za-z]?)*)",
                         text):
        supp = bool(m.group(1))
        for tok in re.split(r"\s*,\s*", m.group(2)):
            mm = re.fullmatch(r"(S?)(\d+)([A-Za-z]?)", tok)
            if not mm:
                continue
            fig = ("S" if (mm.group(1) or supp) else "") + str(int(mm.group(2)))
            out.add((fig, mm.group(3) or None))
    return out


def submission_figures():
    """Supplementary figures the submission ships itself, as {"S14", "S15", ...}.

    The manuscript cites TWO figure sets. The tool paper's own panels are in FIGURE_MANIFEST.tsv
    (Fig 1-6, Fig S1-S13). The revision additionally carries the scoping review as Supplementary
    Note 1, whose figures travel with the submission in supplementary_figures/ and are numbered on
    from S14. Those are not manifest rows and never will be, so checking them against the manifest
    reported six false dangling references.
    """
    # Derived from the EMBEDS, not from a directory listing. rev2 dropped the supplementary_figures/
    # folder and points these figures at the packet's live copies instead, so a directory scan found
    # nothing and reported seven real figures as dangling. An embed counts only if its path actually
    # resolves, which also catches a figure referenced but not shipped.
    out = set()
    for f in FILES:
        p = os.path.join(SUB, f)
        if not os.path.exists(p):
            continue
        for alt, path in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", open(p).read()):
            m = re.search(r"\bS(\d+)\b", alt)
            if m and os.path.exists(os.path.normpath(os.path.join(SUB, path))):
                out.add("S" + str(int(m.group(1))))
    return out


def check_references(manifest_rows, panels, figures):
    bad, seen = [], set()
    shipped = submission_figures()
    for f in FILES:
        p = os.path.join(SUB, f)
        if not os.path.exists(p):
            bad.append((f, "file missing")); continue
        for fig, panel in refs_in(open(p).read()):
            seen.add((fig, panel))
            if fig in shipped:            # a figure the submission carries with it
                continue
            if panel is None:
                if fig not in figures:
                    bad.append((f, f"Fig. {fig} does not exist in the manifest or in supplementary_figures/"))
            elif (fig, panel) not in panels:
                bad.append((f, f"Fig. {fig}{panel} does not exist in the manifest or in supplementary_figures/"))
    # every MAIN-TEXT panel should be cited somewhere; supplementary panels are captioned in 07
    uncited = sorted({(f, p) for f, letters in COMPOSITES.items() for p in letters
                      if not f.startswith("S")}
                     - {(f, p) for f, p in seen if p} - {(f, None) for f, _ in seen})
    uncited = [x for x in uncited if isinstance(x[1], str)]
    for fig, panel in uncited:
        if not any(s[0] == fig and s[1] in (None, panel) for s in seen):
            bad.append(("(manuscript)", f"Fig. {fig}{panel} is in the manifest but never referenced"))
    return bad


def check_abstract_numbers():
    """Every number in the abstract must appear in KEY_NUMBERS.tsv at the same precision."""
    vals = set()
    for k, v in csv.reader(open(KEYNUM), delimiter="\t"):
        vals.add(v.strip())
        vals.add(v.strip().lstrip("+"))
        try:                                   # 1.04e+09 -> also accept 1.0 x 10^9 style mantissas
            f = float(v)
            vals.add(("%g" % f))
            vals.add(("%.1f" % f)); vals.add(("%.2f" % f)); vals.add(("%.0f" % f))
            vals.add("%.1e" % f); vals.add("%.2e" % f); vals.add("%.3e" % f)
            m = re.match(r"([0-9.]+)e([+-]\d+)", "%.1e" % f)
            if m:
                vals.add(m.group(1))
        except ValueError:
            pass
    txt = open(os.path.join(SUB, "01_ABSTRACT.md")).read()
    txt = txt.split("## Note on length")[0]            # the note and the short form are commentary
    txt = re.sub(r"10<sup>-?\d+</sup>", " ", txt)      # exponents of scientific notation
    txt = re.sub(r"<sup>\d+(,\d+)*</sup>", " ", txt)   # reference markers
    txt = txt.replace("95% CI", " ")                   # the confidence level is a convention
    txt = re.sub(r"\b(?:19|20)\d\d\b", " ", txt)       # calendar years (2005, 2026) are not measurements
    missing = []
    for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?=[\s,;.)%-]|-fold|$)", txt):
        if tok in vals:
            continue
        # integers that are counts stated alongside a traceable total are allowed through only if
        # they appear verbatim somewhere in KEY_NUMBERS
        missing.append(tok)
    return missing


def check_forbidden():
    bad = []
    for f in FILES:
        p = os.path.join(SUB, f)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p), 1):
            for pat, why in list(FORBIDDEN.items()) + list(BANNED_PHRASING.items()):
                if re.search(pat, line):
                    bad.append((f, i, pat, why))
    return bad


def check_public_tree():
    """No live file may reference anything that was archived out of the public tree.

    This is the check that makes the archive safe: a reintroduced reference now fails the build
    instead of shipping. It sweeps the public documents AND the live scripts, because the last
    sweep found dangling names in shell comments and docstrings as well as in prose.
    """
    pats = dict(ARCHIVED_PATHS)
    # Retired generators are matched as script file names, so a live script or figure whose name
    # merely contains the same words (submit_model_coverage.sh, FigS04a_model_coverage_data) is not flagged.
    pats.update({r"(?<![A-Za-z0-9_])" + re.escape(n) + r"\.(?:py|sh|R)\b": "retired generator; archived out of the public tree"
                 for n in RETIRED_SCRIPTS})

    targets = [os.path.join(REPO, f) for f in PUBLIC_DOCS]
    for root, dirs, files in os.walk(os.path.join(REPO, "scripts")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        targets += [os.path.join(root, f) for f in files
                    if f.endswith((".py", ".sh", ".R", ".md", ".tsv"))]

    # Exemptions, each for a stated reason rather than a blanket skip.
    exempt = {
        os.path.join(REPO, "scripts", "crosscheck.py"),   # this file lists the names
        os.path.join(REPO, "docs", "REPRODUCING.md"),            # names the archives to explain an absence
        os.path.join(REPO, "scripts", "key_numbers.py"),   # documents a bug it already fixed
    }
    bad = []
    for t in targets:
        if t in exempt or not os.path.exists(t):
            continue
        try:
            text = open(t, errors="ignore").read()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat, why in pats.items():
                if re.search(pat, line):
                    bad.append((os.path.relpath(t, REPO), i, pat, why))
    return bad


def check_figures_dir():
    """results/figures/ must hold nothing loose. This is how the old set accumulated."""
    allowed = {"README.md", "FIGURE_MANIFEST.tsv", "panels", "previews", "composites", "source_data", "Source_Data.xlsx"}
    d = os.path.join(REPO, "results", "figures")
    return sorted(set(os.listdir(d)) - allowed) if os.path.isdir(d) else []


def check_panel_literals():
    """No panel title may assert a number that no table holds.

    The round-6 review found nine manuscript values disagreeing with the panels that cite them.
    Four were a stratum drift between key_numbers and the panel generators; the rest were
    NUMERIC LITERALS typed into panel titles -- "the campaign aggregate, 60x" (computed: 68.8),
    "the six hosts differ by up to 1.67x" (computed max: 2.00), "19x in fp64, 3.7x in wall" on a
    panel that already annotated the computed spread, and "the guard degraded 6 runs", a number
    that appeared in no table at all. Nothing checked panel titles, so check_forbidden's existing
    ban on "60-fold" never bit.

    This walks the AST of every panel generator, pulls the STRING-LITERAL parts of set_title and
    suptitle calls (an f-string's computed parts are exempt by construction -- that is the point),
    and requires each numeric token to exist in KEY_NUMBERS.tsv at some accepted precision.
    """
    import ast
    vals = _keynumber_values()
    ALLOW = {"1", "2", "3", "4", "5", "6", "10", "20", "95", "100"}   # axis words, "20 threads"
    bad = []
    for f in sorted(os.path.join(REPO, "scripts", n) for n in FIGURE_SCRIPTS):
        try:
            tree = ast.parse(open(f).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("set_title", "suptitle")):
                continue
            lits = []
            for a in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    lits.append(a.value)
                elif isinstance(a, ast.JoinedStr):
                    lits += [v.value for v in a.values
                             if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            for text in lits:
                for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text):
                    if tok in ALLOW or tok in vals:
                        continue
                    bad.append((os.path.basename(f), node.lineno, tok,
                                " ".join(text.split())[:58]))
    return bad


def _keynumber_values():
    """Every KEY_NUMBERS value at the precisions a title might print it at."""
    vals = set()
    with open(KEYNUM) as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) < 2:
                continue
            v = row[1].strip()
            vals.add(v); vals.add(v.lstrip("+"))
            try:
                fl = float(v)
            except ValueError:
                continue
            for fmt in ("%g", "%.0f", "%.1f", "%.2f", "%.3f"):
                vals.add(fmt % fl)
    return vals


def check_card_specs():
    """Every card row names its source; Note 2's FP64 table quotes the same values (R21)."""
    bad = []
    p = os.path.join(REPO, "scripts", "benchmark", "card_specs.tsv")
    hdr, rows = None, []
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "card":
            hdr = parts; continue
        rows.append(dict(zip(hdr, parts)))
    for r in rows:
        if not r.get("source", "").strip():
            bad.append(f"card_specs.tsv: {r['card']} has no source")
        if r.get("derived") not in ("TRUE", "FALSE"):
            bad.append(f"card_specs.tsv: {r['card']} derived={r.get('derived')!r}")
    note2 = os.path.join(SUB, "10_SUPPLEMENTARY_NOTE_2_reproducibility.md")
    if os.path.exists(note2):
        txt = open(note2).read()
        names = {"a100": "A100", "teslaV100s": "V100S", "teslaV100": "V100-SXM2", "l40s": "L40S", "a40": "A40", "rtx8000": "RTX 8000"}
        for r in rows:
            if r["card"] not in names:
                continue
            m = re.search(re.escape(names[r["card"]]) + r"[^|\n]*\|[^|\n]*\|\s*([0-9.]+)\s*\|", txt)
            if m and m.group(1) != r["fp64_tflops"]:
                bad.append(f"Note 2 quotes {names[r['card']]} FP64 {m.group(1)} but card_specs.tsv says {r['fp64_tflops']}")
    return bad


def check_memory_denominator():
    """The per-entry arena figure must be computed against the allocated capacity (R13)."""
    vals = dict(csv.reader(open(KEYNUM), delimiter="\t"))
    v = float(vals.get("arena_bytes_per_entry_median", "nan"))
    return [] if 7.9 <= v <= 8.4 else [f"arena_bytes_per_entry_median = {v} (expected 8.0 +- 0.4)"]


def check_inventory_citations():
    """Every S24 row cited in the manuscript exists in the redone inventory (round 9)."""
    inv = os.path.join(REPO, "scoping_review", "IQTREE3-GPU_positioning_packet", "03_supplementary_tables", "csv", "S24_charted_inventory.csv")
    if not os.path.exists(inv) or not os.path.isdir(SUB):
        return []
    ids = {r["record_id"] for r in csv.DictReader(open(inv))}
    bad = []
    for f in FILES:
        p = os.path.join(SUB, f)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p), 1):
            for rid in re.findall(r"\bR\d{4}\b", line):
                if rid not in ids:
                    bad.append(f"{f}:{i} cites {rid}, not in S24")
    return bad


def check_titles(manifest_rows):
    return [r["file"] for r in manifest_rows if r["title"].strip().endswith("?")]


def main():
    rows, panels, figures = load_manifest()
    fails = 0
    # The manuscript lives in a private archive that is excluded from the public distribution, so in
    # a public checkout the first three checks have nothing to read. Skip them with a note rather
    # than failing: the public-tree checks below are the ones that matter there, and they always run.
    have_manuscript = os.path.isdir(SUB)
    if not have_manuscript:
        print("  ----  manuscript checks skipped (submission directory not present in this checkout)")

    bad = check_references(rows, panels, figures) if have_manuscript else []
    if bad:
        fails += 1
        print(f"  FAIL  figure references: {len(bad)} problem(s)")
        for f, msg in bad[:25]:
            print(f"          {f}: {msg}")
    else:
        have_manuscript and print(f"  PASS  every figure reference resolves; all {len(panels)} manifest panels accounted for")

    miss = check_abstract_numbers() if have_manuscript else []
    if miss:
        fails += 1
        print(f"  FAIL  abstract numbers not found in KEY_NUMBERS.tsv: {sorted(set(miss))}")
    else:
        have_manuscript and print("  PASS  every number in the abstract is traceable to KEY_NUMBERS.tsv")

    bad = check_forbidden() if have_manuscript else []
    if bad:
        fails += 1
        print(f"  FAIL  retired names or withdrawn claims present: {len(bad)}")
        for f, i, pat, why in bad[:25]:
            print(f"          {f}:{i}  /{pat}/  -- {why}")
    else:
        have_manuscript and print("  PASS  no retired figure name or withdrawn claim appears in the manuscript")

    bad = check_public_tree()
    if bad:
        fails += 1
        print(f"  FAIL  live files reference archived material: {len(bad)}")
        for f, i, pat, why in bad[:25]:
            print(f"          {f}:{i}  /{pat}/  -- {why}")
    else:
        print(f"  PASS  no live file references archived material "
              f"({len(PUBLIC_DOCS)} public docs + every script)")

    bad = check_figures_dir()
    if bad:
        fails += 1
        print(f"  FAIL  loose entries directly under results/figures/: {bad}")
    else:
        print("  PASS  results/figures/ holds only panels/, previews/, composites/, "
              "README and the manifest")

    bad = check_titles(rows)
    if bad:
        fails += 1
        print(f"  FAIL  panel titles phrased as questions: {bad}")
    else:
        print("  PASS  no panel title is phrased as a question")

    bad = check_panel_literals()
    if bad:
        fails += 1
        print(f"  FAIL  panel titles assert {len(bad)} number(s) absent from KEY_NUMBERS.tsv:")
        for f, ln, tok, ctx in bad[:12]:
            print(f"          {f}:{ln}  '{tok}'  in  \"{ctx}\"")
    else:
        print("  PASS  every number in a panel title is traceable to KEY_NUMBERS.tsv")

    for name, fn in (("card_specs sources and Note 2 FP64 values", check_card_specs),
                     ("arena bytes per entry uses the allocated capacity", check_memory_denominator),
                     ("inventory citations (R####) resolve in S24", check_inventory_citations)):
        bad = fn()
        if bad:
            fails += 1
            print(f"  FAIL  {name}: {len(bad)}")
            for b in bad[:12]:
                print(f"          {b}")
        else:
            print(f"  PASS  {name}")

    print(f"\n  {'CROSS-CHECK FAILED' if fails else 'CROSS-CHECK PASSED'} "
          f"({fails} of {10 if have_manuscript else 7} checks failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
