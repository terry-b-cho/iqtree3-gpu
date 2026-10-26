#!/usr/bin/env python3
"""portability.py - prove the repository runs from any directory.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.

WHY THIS EXISTS. `run_tests.sh` used to assert portability with two greps that were far narrower
than they looked: they globbed only `scripts/*.sh` and `scripts/*/*.sh`, matched only a bare
`VAR=/n/scratch` at line start, and scanned no Python, no R, no Markdown and nothing at the repository
root. They passed vacuously while five real portability bugs sat in the tree -- including three
Python scripts in `scripts/dev/` that derived the repository root one directory too shallow and would
have written into `scripts/examples/`.

Three checks:
  1. NO MACHINE-SPECIFIC ABSOLUTE PATH in any shipped script, document or config.
  2. CORRECT ROOT DEPTH in every Python and R file that derives the repository root from its own
     location -- counted against where the file actually sits.
  3. RELOCATION: copy the machinery to a temporary directory and confirm the repository root resolves
     to the COPY, with every environment variable unset and the working directory elsewhere. This
     is the only check that actually exercises what "runs from any directory" means.

Exit status is non-zero if any check fails.
"""
import os, re, shutil, subprocess, sys, tempfile

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directories that are not part of the distributed tool.
PRUNE = {".git", ".iqtree3_src", "zenodo_upload_bundle", "__pycache__", "node_modules",
         "runs", "figures", "profiling", "data", "logs",
         "iqtree3_gpu_build", "iqtree3_build", "build_graphs", "CMakeFiles"}   # build output is not shipped
PRUNE_PREFIX = ("xx_archive",)

# Roots that belong to one machine. `/usr`, `/opt`, `/etc`, `/tmp`, `/var` and `/dev` are normal in
# a shebang, a container path or an Lmod probe, so they are not listed.
BAD_ROOT = re.compile(r"(?<![\w.])/(?:n/scratch|n/app|home|net|cluster|scratch)(?=/)")

# Files that may legitimately contain the patterns: the guards themselves.
EXEMPT = {
    os.path.join("scripts", "portability.py"),
    os.path.join("benchmark", "scripts", "crosscheck.py"),
    os.path.join("scripts", "run_tests.sh"),
}

SCAN_EXT = (".sh", ".py", ".R", ".yml", ".yaml", ".md", ".cff", ".example", ".def", ".toml")


def shipped_files():
    for root, dirs, files in os.walk(REPO):
        # any CMake build tree (a user may build into ./build or a name of their own) is not shipped either
        dirs[:] = [d for d in dirs
                   if d not in PRUNE and not d.startswith(PRUNE_PREFIX)
                   and not os.path.exists(os.path.join(root, d, "CMakeCache.txt"))]
        for f in files:
            if f.endswith(SCAN_EXT) or f == ".gitignore":
                p = os.path.join(root, f)
                yield p, os.path.relpath(p, REPO)


def check_absolute_paths():
    bad = []
    for p, rel in shipped_files():
        if rel in EXEMPT:
            continue
        try:
            text = open(p, errors="ignore").read()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = BAD_ROOT.search(line)
            if m:
                bad.append((rel, i, line.strip()[:100]))
    return bad


def check_sbatch_directives():
    """#SBATCH -o/-e must be relative AND literal: SLURM expands neither variables nor ~."""
    bad = []
    for p, rel in shipped_files():
        if not rel.endswith(".sh"):
            continue
        for i, line in enumerate(open(p, errors="ignore").read().splitlines(), 1):
            m = re.match(r"^#SBATCH\s+(?:-[oe]|--(?:output|error))[=\s]+(\S+)", line)
            if m and re.match(r"^[/~]|\$", m.group(1)):
                bad.append((rel, i, m.group(1)))
    return bad


def check_root_depth():
    """A file at scripts/x.py needs two dirname levels; benchmark/scripts/x.py needs three.

    The Python idiom starts from the FILE (`abspath(__file__)`), so it needs one `dirname` per path
    component including the file itself. The R idiom starts from the file's DIRECTORY
    (`dirname(--file=)`), so it needs one `".."` fewer.
    """
    bad = []
    for p, rel in shipped_files():
        depth = len(rel.split(os.sep))              # dirname() calls needed from the file itself
        if rel.endswith(".py"):
            for i, line in enumerate(open(p, errors="ignore").read().splitlines(), 1):
                if "abspath(__file__)" not in line or "dirname" not in line:
                    continue
                if not re.search(r"\b(REPO|MODULE_ROOT|ROOT)\b\s*=", line):
                    continue
                got = line.count("os.path.dirname(")
                if got != depth:
                    bad.append((rel, i, f"{got} dirname level(s), needs {depth}"))
        elif rel.endswith(".R"):
            for i, line in enumerate(open(p, errors="ignore").read().splitlines(), 1):
                if re.search(r"\bMOD\b\s*<-", line) and "--file=" in line:
                    got = line.count('".."')
                    if got != depth - 1:            # R starts from the directory, not the file
                        bad.append((rel, i, f'{got} ".." level(s), needs {depth - 1}'))
    return bad


def check_relocation():
    """The real proof: resolve the repository root from a COPY, from `/`, with the environment bare."""
    tmp = tempfile.mkdtemp(prefix="iqtree_gpu_reloc_")
    try:
        dest = os.path.join(tmp, "relocated-clone")
        os.makedirs(dest)
        shutil.copy2(os.path.join(REPO, "build_iqtree_gpu.sh"), dest)   # the second root marker
        shutil.copytree(os.path.join(REPO, "scripts"), os.path.join(dest, "scripts"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(REPO, "benchmark", "scripts"), os.path.join(dest, "benchmark", "scripts"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(REPO, "config"), os.path.join(dest, "config"))
        fails = []

        env = {k: v for k, v in os.environ.items()
               if k not in ("REPO_DIR", "SLURM_SUBMIT_DIR", "LOGDIR", "IQTREE_BUILD_ROOT")}
        r = subprocess.run(["bash", "-c", f'. "{dest}/scripts/lib/paths.sh" && printf "%s" "$REPO"'],
                           cwd="/", env=env, capture_output=True, text=True)
        if r.stdout.strip() != dest:
            fails.append(f"paths.sh resolved to {r.stdout.strip()!r}, expected {dest!r}"
                         + (f" (stderr: {r.stderr.strip()[:120]})" if r.stderr.strip() else ""))

        # every Python entry point must agree, from the copy, with REPO_DIR unset
        for name in ("key_numbers.py", "aggregate_runs.py", "crosscheck.py"):
            f = os.path.join(dest, "benchmark", "scripts", name)
            if not os.path.exists(f):
                continue
            code = ("import os,re,sys\n"
                    "src=open(sys.argv[1]).read()\n"
                    "for l in src.split(chr(10)):\n"
                    "    s=l.strip()\n"
                    "    if s.startswith(('REPO =','REPO=')) and 'abspath(__file__)' in s:\n"
                    "        ns={'os':os,'__file__':sys.argv[1]}; exec(s,ns); print(ns['REPO']); break\n"
                    "    if s.startswith(('_HERE =','_HERE=')) and 'abspath(__file__)' in s:\n"
                    "        ns={'os':os,'__file__':sys.argv[1]}; exec(s,ns)\n"
                    "        for l2 in src.split(chr(10)):\n"
                    "            s2=l2.strip()\n"
                    "            if s2.startswith(('REPO =','REPO=')) and '_HERE' in s2:\n"
                    "                exec(s2,ns); print(ns['REPO']); break\n"
                    "        break\n")
            rr = subprocess.run([sys.executable, "-c", code, f], cwd="/", env=env,
                                capture_output=True, text=True)
            got = rr.stdout.strip()
            if got and got != dest:
                fails.append(f"{name} resolved to {got!r}, expected {dest!r}")
        return fails
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    fails = 0

    bad = check_absolute_paths()
    if bad:
        fails += 1
        print(f"  FAIL  machine-specific absolute path in {len(bad)} line(s)")
        for rel, i, line in bad[:15]:
            print(f"          {rel}:{i}  {line}")
    else:
        print("  PASS  no machine-specific absolute path in any shipped script, doc or config")

    bad = check_sbatch_directives()
    if bad:
        fails += 1
        print(f"  FAIL  #SBATCH log path is absolute or contains a variable: {len(bad)}")
        for rel, i, v in bad[:10]:
            print(f"          {rel}:{i}  {v}")
    else:
        print("  PASS  every #SBATCH log path is relative and literal")

    bad = check_root_depth()
    if bad:
        fails += 1
        print(f"  FAIL  wrong repository-root depth in {len(bad)} file(s)")
        for rel, i, why in bad:
            print(f"          {rel}:{i}  {why}")
    else:
        print("  PASS  every script derives the repository root at the correct depth")

    bad = check_relocation()
    if bad:
        fails += 1
        print(f"  FAIL  relocation: {len(bad)} problem(s)")
        for b in bad:
            print(f"          {b}")
    else:
        print("  PASS  the repository resolves its own root after relocation, from / with a bare environment")

    print(f"\n  {'PORTABILITY FAILED' if fails else 'PORTABILITY PASSED'} ({fails} of 4 checks failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
