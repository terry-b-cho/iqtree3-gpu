#!/bin/bash
#SBATCH --job-name=gpuvalid
#SBATCH -p gpu
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: a LITERAL value on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$SLURM_GRES" here is passed to the scheduler verbatim and the allocation fails).
# This is the default for a bare `sbatch scripts/...`; the submitters pass --gres="$SLURM_GRES" on
# the sbatch COMMAND LINE, where the shell does expand it, and a command-line flag overrides the
# directive. Set SLURM_GRES in config/site.env to change it everywhere the submitters are used.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH -t 4:00:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/gpuvalid_%j.out
#SBATCH -e logs/gpuvalid_%j.err
# =============================================================================
#  IQ-TREE3-GPU - validate the GPU backend now that the kernel swap actually applies.
# =============================================================================
#  Until the setLikelihoodKernel wrapper fix, the GPU pointer swap was unreachable dead
#  code: every `-gpu` run silently used the CPU AVX+FMA kernel. So the previous "7/7 gates
#  pass, logL reldiff exactly 0.0" was vacuous -- it compared the CPU kernel to itself, and
#  the multi-threaded GPU path (per-thread CUDA streams over disjoint site ranges) has never
#  once executed. Race conditions there would have been invisible.
#
#  This script therefore checks TWO things, in order:
#    (A) ENGAGEMENT - the GPU is genuinely doing the work (banner + non-zero utilisation)
#    (B) EXACTNESS  - GPU == CPU (RF=0, matching logL) at -T 1, -T 4, -T 20
#  (A) must hold or (B) is meaningless. Both arms run on THIS node, same binary, so the
#  CPU-hardware confound that invalidated the earlier matrix cannot recur.
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
CHECK=$REPO/scripts/rf_logl_check.py
ALN=${ALN:-$REPO/data/protein/g__CAG-269/concat_alignment.faa}
OUT=$REPO/results/benchmarks/gpu_validation
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT" || exit 1

echo "== node $(hostname) =="
echo "== binary $(stat -c %y $BIN) =="
nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/   card: /'
grep -m1 "model name" /proc/cpuinfo | sed 's/^/   cpu : /'
echo "== alignment: $ALN ($(grep -c '^>' $ALN) taxa) =="

MODEL=LG+G4
run () { # tag  extra_flags...
  local tag=$1; shift
  echo; echo "--- run: $tag ---"
  ( for i in $(seq 1 2400); do
      nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 0.25
    done ) > "$OUT/$tag.util" 2>/dev/null & local S=$!
  local t0=$SECONDS
  "$BIN" -s "$ALN" -m $MODEL -seed 1 --prefix "$OUT/$tag" -redo "$@" > "$OUT/$tag.stdout" 2>&1
  local rc=$? w=$((SECONDS-t0))
  kill $S 2>/dev/null; wait $S 2>/dev/null
  local act util
  act=$(grep -c "GPU acceleration ACTIVE" "$OUT/$tag.stdout")
  util=$(awk 'NF&&$1+0==$1{n++;s+=$1;if($1>m)m=$1}END{if(n)printf "mean %.1f%% max %d%%",s/n,m; else printf "n/a"}' "$OUT/$tag.util")
  printf "    rc=%s wall=%ss  ACTIVE-banner=%s  GPU-util: %s\n" "$rc" "$w" "$act" "$util"
  echo "$w" > "$OUT/$tag.wall"
}

# ---------------- (A) engagement + (B) exactness ----------------
run cpu_T20        -T 20
run gpu_T1  -gpu   -T 1
run gpu_T4  -gpu   -T 4
run gpu_T20 -gpu   -T 20

# +R arm. FreeRate drives _pattern_lh_cat, which the GPU must write in IQ-TREE's
# vector_size-interleaved layout for transformPatternLhCat() to de-interleave correctly.
# If that layout is wrong the tree log-likelihood still looks right while +R/+I EM
# (ratefree.cpp) and empirical-Bayes site rates go subtly wrong -- so it needs its own check.
MODEL=LG+R4
run cpuR_T20       -T 20
run gpuR_T20 -gpu  -T 20
MODEL=LG+G4

echo; echo "=============== (A) ENGAGEMENT ==============="
ENG=1
for t in gpu_T1 gpu_T4 gpu_T20 gpuR_T20; do
  a=$(grep -c "GPU acceleration ACTIVE" "$OUT/$t.stdout")
  u=$(awk 'NF&&$1+0==$1&&$1>0{c++}END{print c+0}' "$OUT/$t.util")
  if [ "$a" -ge 1 ] && [ "$u" -ge 1 ]; then echo "  PASS $t: swap applied, $u non-zero-utilisation samples"
  else echo "  FAIL $t: banner=$a nonzero_util_samples=$u"; ENG=0; fi
done
[ "$ENG" = 1 ] || { echo; echo "  *** GPU NOT ENGAGED - exactness results below are meaningless ***"; }

echo; echo "=============== (B) EXACTNESS (vs CPU) ==============="
for t in gpu_T1 gpu_T4 gpu_T20; do python3 "$CHECK" "$OUT/cpu_T20" "$OUT/$t" "$t"; done
echo "  --- +R4 (exercises _pattern_lh_cat interleaving) ---"
python3 "$CHECK" "$OUT/cpuR_T20" "$OUT/gpuR_T20" "LG+R4_T20"
echo; echo "  --- GPU thread-scaling self-consistency (must be identical to each other) ---"
python3 "$CHECK" "$OUT/gpu_T1" "$OUT/gpu_T20" "gpuT1_vs_gpuT20"

echo; echo "=============== TIMING (same node, same binary) ==============="
for t in cpu_T20 gpu_T1 gpu_T4 gpu_T20 cpuR_T20 gpuR_T20; do printf "  %-10s %ss\n" "$t" "$(cat $OUT/$t.wall)"; done
echo "  (fixed model LG+G4, no ModelFinder -- indicative only, not the headline benchmark)"
