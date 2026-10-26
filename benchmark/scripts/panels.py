#!/usr/bin/env python3
"""panels.py - enforce the figure contract instead of asserting it.

WHY THIS EXISTS. After the previous pass I reported that every text element was 6 pt and the theme
was uniform. Both were false: I had verified the embedded font FAMILY in the PDFs and never the
SIZE, and 102 explicit `fontsize=` arguments and 16 `.grid(True)` calls were quietly overriding the
shared style. rcParams set defaults; they do not bind explicit arguments. The only way to know is to
read the emitted files, which is what this does.

Checks, each of which fails the build:
  1. every panel's Tf font sizes lie in the permitted set (mathtext sub/superscripts allowed)
  2. no panel embeds DejaVu (the silent fallback when Arial/Helvetica/Nimbus are absent)
  3. every emitted file matches the naming contract Fig{NN}{p}_{slug} / FigS{NN}{p}_{slug}
  4. every panel has BOTH a .pdf and a .png
  5. every panel appears in FIGURE_MANIFEST.tsv, and the manifest has no orphan rows

Exit code 0 if all pass, 1 otherwise.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, re, sys, csv, glob, zlib, collections, importlib.util as ilu

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_p = os.path.join(REPO, "benchmark", "scripts", "lib", "figstyle.py")
_s = ilu.spec_from_file_location("figstyle", _p); FS = ilu.module_from_spec(_s)
_s.loader.exec_module(FS)

PANELS = os.path.join(REPO, "benchmark", "figures", "panels")
MANIFEST = os.path.join(REPO, "benchmark", "figures", "FIGURE_MANIFEST.tsv")
NAME = re.compile(r"^Fig(S?)(\d{2})([a-z])_([a-z0-9_]+?)(_p\d{2})?$")


def pdf_font_sizes(path):
    d = open(path, "rb").read()
    out = set()
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", d, re.S):
        try:
            t = zlib.decompress(m.group(1))
        except Exception:
            t = m.group(1)
        for g in re.finditer(rb"/F\d+\s+([\d.]+)\s+Tf", t):
            out.add(round(float(g.group(1)), 2))
    return out


def pdf_fonts(path):
    d = open(path, "rb").read()
    return set(re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", d))


def main():
    fails, checked = [], 0
    if not os.path.isdir(PANELS):
        print("  no panels directory yet - nothing to audit"); return 0
    perm = set(FS.permitted_sizes())
    stems = sorted({os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{PANELS}/*")})
    for stem in stems:
        checked += 1
        if not NAME.match(stem):
            fails.append(f"naming: {stem} does not match Fig{{NN}}{{p}}_{{slug}}")
        pdf, png = f"{PANELS}/{stem}.pdf", f"{PANELS}/{stem}.png"
        if not (os.path.exists(pdf) and os.path.exists(png)):
            fails.append(f"format: {stem} is missing a .pdf or .png")
            continue
        off = sorted(s for s in pdf_font_sizes(pdf) if s not in perm)
        if off:
            fails.append(f"type: {stem} uses {off} pt, outside {sorted(perm)}")
        dv = [f.decode() for f in pdf_fonts(pdf) if b"DejaVu" in f]
        if dv:
            fails.append(f"font: {stem} embeds {dv} - the silent fallback")
    # manifest agreement
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            listed = {r["file"] for r in csv.DictReader(fh, delimiter="\t")}
        for s in stems:
            if s not in listed:
                fails.append(f"manifest: {s} is not listed")
        for s in listed - set(stems):
            fails.append(f"manifest: {s} is listed but no file exists")
    else:
        fails.append("manifest: FIGURE_MANIFEST.tsv missing")
    # source-side: the overrides that defeat the shared style
    for f in [f"{REPO}/benchmark/scripts/{n}" for n in ("equivalence.py", "decision.py", "mechanism.py", "phases_cost.py",
                                               "hardware.py", "memory.py", "supplementary.py")]:
        src = open(f).read()
        for pat, msg in ((r"fontsize\s*=\s*[0-9]", "literal fontsize="),
                         (r"\.grid\(True", ".grid(True)")):
            if re.search(pat, src):
                fails.append(f"source: {os.path.basename(f)} contains {msg} - style belongs in figstyle")
    print(f"  audited {checked} panel(s)")
    for f in fails:
        print(f"    FAIL  {f}")
    print(f"  {'PASS' if not fails else 'FAILED'}: {len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
