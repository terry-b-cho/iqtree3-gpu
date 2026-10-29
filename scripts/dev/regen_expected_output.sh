#!/bin/bash
#SBATCH --job-name=regen_example
#SBATCH -p gpu_requeue
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: a LITERAL value on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$SLURM_GRES" here is passed to the scheduler verbatim and the allocation fails).
# This is the default for a bare `sbatch scripts/...`; the submitters pass --gres="$SLURM_GRES" on
# the sbatch COMMAND LINE, where the shell does expand it, and a command-line flag overrides the
# directive. Set SLURM_GRES in config/site.env to change it everywhere the submitters are used.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/regen_example_%j.out
#SBATCH -e logs/regen_example_%j.err
# One-off: regenerate examples/expected_output with the current binary. The reference a user
# compares against must come from the shipped engine. Correctness-only -> gpu_requeue is fine.
set -uo pipefail
[ -r /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh 2>/dev/null || true
i3g_load_modules || exit 1
# --- locate the repository, then the binaries ------------------------------------------------------
# Try each candidate source in turn and take the FIRST that actually contains scripts/lib/paths.sh.
# We do NOT special-case the SLURM spool path: SLURM execs a COPY of a batch script (commonly under
# /var/spool/slurmd, but the location is site-configurable), so "where am I" is simply not a reliable
# answer inside a job. Probing in order and validating each candidate works wherever the spool lives.
_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") \
  || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "${SLURM_SUBMIT_DIR:-}") \
  || _d=$(_i3g_try "$PWD") \
  || { echo "FATAL: cannot locate the repository root. Submit with REPO_DIR=/path/to/iqtree3-gpu sbatch ...," >&2
       echo "       or run sbatch from the repository root so SLURM_SUBMIT_DIR points at it." >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"
BIN=$IQTREE_BUILD_ROOT/iqtree3_gpu_build/iqtree3
EXP=$REPO/examples/expected_output
W=/tmp/regen_example_$SLURM_JOB_ID; mkdir -p "$W"

echo "== node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) =="
echo "== binary sha256=$(sha256sum "$BIN" | cut -c1-16) =="

# Run the example exactly as a user would, but into a scratch dir (OUT is respected via env? no --
# run_example writes into examples/output; run the commands directly, mirroring run_example.sh).
ALN=$REPO/examples/data/example_CAG-269.faa
"$BIN" -s "$ALN" -m LG+G4 -B 1000 -T 1 -gpu -seed 1 --prefix "$W/gpu" -redo > "$W/gpu.stdout" 2>&1
grep -q "GPU acceleration ACTIVE" "$W/gpu.stdout" || { echo "FAIL: GPU not engaged"; exit 1; }
"$BIN" -s "$ALN" -m LG+G4 -B 1000 -T 1 -seed 1 --prefix "$W/cpu" -redo > "$W/cpu.stdout" 2>&1
VER=$(python3 "$REPO/scripts/rf_logl_check.py" "$W/cpu" "$W/gpu" example 2>&1 | grep "VERDICT")
echo "  $VER"
echo "$VER" | grep -q "PASS" || { echo "FAIL: verdict not PASS"; exit 1; }

# Rebuild the four reference artifacts in the same format as before.
cp "$W/gpu.treefile" "$EXP/gpu.treefile"
grep -E "Number of constant sites|Number of invariant|parsimony informative|Model of substitution|Log-likelihood of the tree" \
  "$W/gpu.iqtree" | head -5 > "$EXP/gpu_key_lines.txt"
{
  echo "Reference result for examples/run_example.sh"
  echo "($(nvidia-smi --query-gpu=name --format=csv,noheader|head -1), CUDA 12.8, gcc 14.2.0, IQ-TREE 3.1.3 + GPU backend, -m LG+G4 -B 1000 -T 1 -seed 1)"
  echo
  echo "$VER"
  echo
  echo "Your run should reproduce rf_dist=0 and the same log-likelihood. Small differences in wall time are"
  echo "normal; the tree and log-likelihood should match."
} > "$EXP/verdict.txt"
# tree figure via the same renderer as the benchmark trees (Bio.Phylo, subs/site scale bar)
python3 - "$W/gpu.treefile" "$EXP/example_tree" <<'PY'
import sys, os
os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import Phylo
t = Phylo.read(sys.argv[1], "newick"); t.ladderize()
fig, ax = plt.subplots(figsize=(5,4))
Phylo.draw(t, axes=ax, do_show=False, show_confidence=False)
for c in ax.collections: c.set_color("#0072B2"); c.set_linewidth(1.1)
for l in ax.get_lines(): l.set_color("#0072B2"); l.set_linewidth(1.1)
ax.set_xlabel("substitutions/site"); ax.set_ylabel("")
ax.set_title("Example: GPU ML tree (LG+G4, identical to CPU, RF=0)", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2]+".png", dpi=160, bbox_inches="tight")
fig.savefig(sys.argv[2]+".pdf", bbox_inches="tight")
PY
echo "== expected_output regenerated =="
ls -la "$EXP" | sed 's/^/  /'
