#!/usr/bin/env python3
"""crosscheck.py - cross-file checks on the public tree: figures, docs, scripts and the data.

Six checks, every one of them on files that ship in this repository; a failure is caught by
`run_tests.sh` rather than by a reader:

  1. no public document or live script references material archived out of the public tree;
  2. benchmark/figures/ holds nothing loose beyond the panels, previews, composites, Source Data,
     the README and the manifest;
  3. no panel title in the manifest is phrased as a question;
  4. no panel title asserts a number that KEY_NUMBERS.tsv does not hold;
  5. the per-entry arena figure is computed against the allocated capacity;
  6. the figure and table map in benchmark/README.md is current.

Exit status is non-zero if any check fails. Run: python3 benchmark/scripts/crosscheck.py

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import csv, os, re, sys

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MANIFEST = os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv")
KEYNUM = os.path.join(REPO, "benchmark", "runs", "KEY_NUMBERS.tsv")
# The panel generators whose titles are audited against KEY_NUMBERS.tsv (main figures, then supplementary).
FIGURE_SCRIPTS = ("equivalence.py", "decision.py", "mechanism.py", "phases_cost.py", "hardware.py",
                  "supplementary.py", "supp_ported.py", "trees.py", "source_data.py", "supplementary_data.py",
                  "style_workbooks.py")

# The PUBLIC tree. These files are what a reader sees first, and a stale figure citation or a
# dangling pointer into archived material in any of them ships with the repository.
PUBLIC_DOCS = ["README.md", "docs/USER_GUIDE.md", "docs/DEPENDENCIES.md", "docs/CONTRIBUTING.md",
               "CHANGELOG.md", "examples/README.md", "scripts/dev/README.md", "validation/README.md",
               "benchmark/README.md", "benchmark/REPRODUCING.md", "benchmark/BENCHMARK_SCHEMA.md",
               "benchmark/figures/README.md", "benchmark/FIGURE_MAP.md", "benchmark/data/README.md",
               "benchmark/data/protein/README.md", "benchmark/data/dna/README.md",
               "benchmark/scripts/harness/README.md", "benchmark/scripts/dev/README.md"]

# Everything archived out of the public tree. A reference to any of these is a dangling pointer.
RETIRED_SCRIPTS = ["replicate_variance", "mechanism_figures", "per_step_figures",
                   "phase_decomposition", "complexity_scaling", "benchmark_figures",
                   "status_dashboard", "cpu_production_figure", "all_trees",
                   "tree_figures", "model_coverage", "interim_report",
                   "card_comparison", "paired_alphabet"]

ARCHIVED_PATHS = {
    r"\btrees_all\b": "superseded per-run tree record; replaced by FigS02a_p01..p06",
    r"\bTREE_INDEX\b": "index of the superseded tree record; archived",
    r"\bPRODUCTION_READINESS\b": "internal readiness assessment; archived out of the public tree",
    r"INVALIDATED_2026-07-21": "record of a resolved historical failure; archived",
    r"xx_archive_": "private archive; a public document must not link into one",
}


def load_manifest():
    with open(MANIFEST) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def check_public_tree():
    """No live file may reference anything that was archived out of the public tree.

    This is the check that makes the archive safe: a reintroduced reference now fails the build
    instead of shipping. It sweeps the public documents AND the live scripts, because the last
    sweep found dangling names in shell comments and docstrings as well as in prose.
    """
    pats = dict(ARCHIVED_PATHS)
    # Retired generators are matched as script file names, so a live script or figure whose name
    # merely contains the same words (submit_model_coverage.sh, FigS03a_model_coverage_data) is not flagged.
    pats.update({r"(?<![A-Za-z0-9_])" + re.escape(n) + r"\.(?:py|sh|R)\b": "retired generator; archived out of the public tree"
                 for n in RETIRED_SCRIPTS})

    targets = [os.path.join(REPO, f) for f in PUBLIC_DOCS]
    for top in (os.path.join(REPO, "scripts"), os.path.join(REPO, "benchmark", "scripts")):
        for root, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            targets += [os.path.join(root, f) for f in files
                        if f.endswith((".py", ".sh", ".R", ".md", ".tsv"))]

    # Exemptions, each for a stated reason rather than a blanket skip.
    exempt = {
        os.path.join(REPO, "benchmark", "scripts", "crosscheck.py"),   # this file lists the names
        os.path.join(REPO, "benchmark", "REPRODUCING.md"),   # names the archives to explain an absence
        os.path.join(REPO, "benchmark", "scripts", "key_numbers.py"),   # documents a bug it already fixed
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
    """benchmark/figures/ must hold nothing loose. This is how the old set accumulated."""
    allowed = {"README.md", "FIGURE_MANIFEST.tsv", "panels", "previews", "composites", "source_data", "Source_Data.xlsx"}
    d = os.path.join(REPO, "benchmark", "figures")
    return sorted(set(os.listdir(d)) - allowed) if os.path.isdir(d) else []


def check_panel_literals():
    """No panel title may assert a number that no table holds.

    A review once found nine quoted values disagreeing with the panels that cite them. Four were a
    stratum drift between key_numbers and the panel generators; the rest were NUMERIC LITERALS typed
    into panel titles -- "the campaign aggregate, 60x" (computed: 68.8), "the six hosts differ by up
    to 1.67x" (computed max: 2.00), "19x in fp64, 3.7x in wall" on a panel that already annotated
    the computed spread, and "the guard degraded 6 runs", a number that appeared in no table at all.

    This walks the AST of every panel generator, pulls the STRING-LITERAL parts of set_title and
    suptitle calls (an f-string's computed parts are exempt by construction -- that is the point),
    and requires each numeric token to exist in KEY_NUMBERS.tsv at some accepted precision.
    """
    import ast
    vals = _keynumber_values()
    ALLOW = {"1", "2", "3", "4", "5", "6", "10", "20", "95", "100"}   # axis words, "20 threads"
    bad = []
    for f in sorted(os.path.join(REPO, "benchmark", "scripts", n) for n in FIGURE_SCRIPTS):
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


def check_memory_denominator():
    """The per-entry arena figure must be computed against the allocated capacity (R13)."""
    vals = dict(csv.reader(open(KEYNUM), delimiter="\t"))
    v = float(vals.get("arena_bytes_per_entry_median", "nan"))
    return [] if 7.9 <= v <= 8.4 else [f"arena_bytes_per_entry_median = {v} (expected 8.0 +- 0.4)"]


def check_benchmark_map():
    """benchmark/README.md's figure and table tables are generated by benchmark_map.py; they must be current."""
    import importlib.util as ilu
    p = os.path.join(REPO, "benchmark", "scripts", "benchmark_map.py")
    sp = ilu.spec_from_file_location("benchmark_map", p); m = ilu.module_from_spec(sp); sp.loader.exec_module(m)
    readme = os.path.join(REPO, "benchmark", "README.md")
    if not os.path.exists(readme):
        return ["benchmark/README.md is missing"]
    return m.stale(readme)


def check_titles(manifest_rows):
    return [r["file"] for r in manifest_rows if r["title"].strip().endswith("?")]


def main():
    rows = load_manifest()
    fails = 0

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
        print(f"  FAIL  loose entries directly under benchmark/figures/: {bad}")
    else:
        print("  PASS  benchmark/figures/ holds only panels/, previews/, composites/, "
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

    for name, fn in (("arena bytes per entry uses the allocated capacity", check_memory_denominator),
                     ("benchmark/README.md figure and table map is current", check_benchmark_map)):
        bad = fn()
        if bad:
            fails += 1
            print(f"  FAIL  {name}: {len(bad)}")
            for b in bad[:12]:
                print(f"          {b}")
        else:
            print(f"  PASS  {name}")

    print(f"\n  {'CROSS-CHECK FAILED' if fails else 'CROSS-CHECK PASSED'} ({fails} of 6 checks failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
