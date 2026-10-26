#!/bin/bash
# Submit the paired benchmark. Two MATCHED sets, one per thread budget, same 14 genera in each:
#   20-thread set : full genera            -> production scale (CPU-20T vs GPU-20T)
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#   1-thread  set : 1Thread_g__*_n6 subsets -> 1T is ~19x slower in wall time, so full genera
#                   would take days-weeks on CPU-1T and time out; the n=6 subsets keep the
#                   1T comparison obtainable with the SAME number of datapoints (n) as the 20T set.
# Both arms of a cell always run in ONE job on ONE node (see paired_benchmark.sh).
# Card default is a100: the kernel is fp64 compute-bound and the L40S runs fp64 at 1/64 rate.
set -uo pipefail
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
S=$REPO/benchmark/scripts/harness/paired_benchmark.sh
MODE=${1:-20t}; CARD=${CARD:-a100}
# BENCHMARK SET: 12 genera at 12 DISTINCT tree sizes, n = 4..30 (benchmark/scripts/dev/build_capped_genomes.py).
#
# WHY THIS SHAPE. This is a tool-development benchmark. It has to demonstrate (a) the GPU returns the
# same tree as CPU IQ-TREE and (b) it is faster, ACROSS problem sizes -- cheaply enough to re-run on
# every engine change. Evenly spaced sizes let speedup-vs-taxa be FITTED and extrapolated; an earlier
# version capped everything at 30 and produced six datasets at the same n, from which no scaling
# behaviour can be inferred. Alignment columns also vary (99k..233k), so both dimensions move.
#
# This constrains only the BENCHMARK, never the tool. The device arena is sized from IQ-TREE's own
# max_lh_slots and falls back to the CPU kernel if it will not fit in VRAM, so users can analyse trees
# of any size and any number of taxa. The ORIGINAL uncapped genera remain in benchmark/data/protein/g__* and can be
# run individually:  GENUS=g__Colicola THREADS=20 sbatch benchmark/scripts/harness/paired_benchmark.sh
FULL=(n04_g__Brotaphodocola n06_g__CAG-269 n08_g__Hominicoprocola n10_g__Coprovivens \
      n12_g__Caccovivens n14_g__Nanosyncoccus n17_g__Lachnospira n20_g__Coprococcus_C \
      n23_g__Onthovivens n26_g__Ruminococcus n28_g__Roseburia n30_g__Enterousia)
EXCLUDED=(g__Colicola g__Ruminococcoides)   # not needed for the size sweep; runnable individually
# 1-thread set: the existing n=6 subsets, already far below the cap (built by build_1thread_genomes.py)
ONET=(); for g in "${FULL[@]}"; do ONET+=("1Thread_${g#n??_}_n6"); done
# Card -> partition map (verified with sinfo). Both non-preemptible, which timing runs require:
#   gpu_quad : a100 (1 real node, 4 GPUs; a100.mig is MIG-partitioned and unusable here),
#              teslaV100s (5 nodes, 20 GPUs, 32 GB)
#   gpu      : teslaV100 (2 nodes, 8 GPUs) -- but only 16 GB VRAM, too small for the largest genera
# teslaV100s is the pragmatic primary: 8.2 vs 9.7 TFLOPS fp64 (within 15%) but 5x the capacity.
case "$CARD" in
  # Card-to-partition mapping is site knowledge (on O2 these cards live in gpu_quad).
  # Override per invocation with PART=..., or set SLURM_PARTITION in config/site.env.
  a100|teslaV100s) PART=${PART:-gpu_quad} ;;
  teslaV100)       PART=gpu ;;
  *)               PART=gpu ;;
esac
mkdir -p "$LOGDIR"
sub () { # genus threads time
  sbatch --job-name="pr_${1#g__}_T$2_$CARD" --chdir="$REPO" -o "$LOGDIR/pr_${1#g__}_T$2_${CARD}_%j.out" -e "$LOGDIR/pr_${1#g__}_T$2_${CARD}_%j.err" --partition=$PART --gres=gpu:${CARD}:1 \
         --time="$3" --export=ALL,REPO_DIR="$REPO",GENUS="$1",THREADS="$2",CARD="$CARD" "$S" \
    && echo "  submitted $1 -T $2 @$CARD -> $PART ($3)"
}
case "$MODE" in
  20t) echo "=== paired CPU-20T vs GPU-20T, ${#FULL[@]} full genera @$CARD ==="
       for g in "${FULL[@]}"; do sub "$g" 20 2-00:00:00; done ;;
  1t)  echo "=== paired CPU-1T vs GPU-1T, ${#ONET[@]} n=6 subsets @$CARD ==="
       for g in "${ONET[@]}"; do sub "$g" 1 3-00:00:00; done ;;
  pilot) echo "=== PILOT: one small genus, both thread budgets, @$CARD ==="
       sub g__CAG-269 20 1-00:00:00; sub 1Thread_g__CAG-269_n6 1 1-00:00:00 ;;
  *) echo "usage: $0 {pilot|20t|1t}   (env: CARD=a100|l40s|...)"; exit 1 ;;
esac
