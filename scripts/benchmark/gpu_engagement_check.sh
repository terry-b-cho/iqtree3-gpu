#!/bin/bash
#SBATCH --job-name=gpuchk
#SBATCH -p gpu
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: a LITERAL value on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$SLURM_GRES" here is passed to the scheduler verbatim and the allocation fails).
# This is the default for a bare `sbatch scripts/...`; the submitters pass --gres="$SLURM_GRES" on
# the sbatch COMMAND LINE, where the shell does expand it, and a command-line flag overrides the
# directive. Set SLURM_GRES in config/site.env to change it everywhere the submitters are used.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 20
#SBATCH --mem=48G
#SBATCH -t 0:30:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/gpuchk_%j.out
#SBATCH -e logs/gpuchk_%j.err
# =============================================================================
#  Is the GPU likelihood kernel ACTUALLY engaged in the benchmarked binary?
# =============================================================================
#  Evidence prompting this check, from a live gpu_T20 A100 run:
#    - "[IQ-TREE3-GPU] N CUDA device(s) visible"   printed 1166 times
#    - "[IQ-TREE3-GPU] GPU acceleration ACTIVE"    printed 0 times
#    - GPU utilisation 0% across 120 samples over 60 s (SM clock idle at 765 MHz)
#  Those printfs are consecutive and unconditional inside the pointer-swap block of
#  setLikelihoodKernel(). The device-count report sits EARLIER in the same function.
#  If only the earlier one prints, control never reaches the swap -> the GPU kernel is
#  never installed and every "GPU" run silently used the CPU kernel.
#  The benchmarked binary was built Jul 13; phylotreesse.cpp was edited Jul 17.
# =============================================================================
set -uo pipefail
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
ALN=$REPO/examples/data/example_CAG-269.faa
OUT=$REPO/results/benchmarks/gpu_engagement
mkdir -p "$OUT"; cd "$OUT" || exit 1

echo "== node $(hostname); binary built: $(stat -c %y $BIN) =="
nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/  card: /'

# background utilisation sampler
( for i in $(seq 1 400); do
    nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader,nounits
    sleep 0.25
  done ) > "$OUT/util_samples.txt" 2>/dev/null &
SAMP=$!

echo; echo "--- running: iqtree3 -gpu -T 20 -m LG+G4 (full stdout captured) ---"
"$BIN" -s "$ALN" -m LG+G4 -T 20 -gpu -seed 1 --prefix "$OUT/chk" -redo > "$OUT/chk.stdout" 2>&1
echo "  exit=$?"
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null

echo; echo "=============== VERDICT ==============="
for p in "CUDA device(s) visible" "GPU acceleration ACTIVE" "multi-thread site parallelism" "accelerated: amino-acid" "selftest"; do
  printf "  %-34s : %s\n" "$p" "$(grep -c "$p" "$OUT/chk.stdout")"
done
echo
awk -F, 'NF>=2 && $1+0==$1 {n++; s+=$1; if($1>m)m=$1; if($1>0)b++} END{
  if(n) printf "  GPU utilisation over %d samples: mean %.1f%%  max %d%%  nonzero %d (%.0f%%)\n", n, s/n, m, b, 100*b/n;
  else print "  no utilisation samples"}' "$OUT/util_samples.txt"
echo
if grep -q "GPU acceleration ACTIVE" "$OUT/chk.stdout"; then
  echo "  => swap block REACHED: GPU kernel installed."
else
  echo "  => swap block NEVER REACHED: the -gpu runs used the CPU kernel."
  echo "     Every GPU-vs-CPU timing collected with this binary is invalid."
fi
