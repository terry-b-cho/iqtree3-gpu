#!/usr/bin/env bash
# =============================================================================
#  submit_ultra_complexity.sh - tree building BEYOND the previous complexity ceiling.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY. Reviewer feedback: we needed measurements past 1e10 and past our own maximum, for BOTH
#  alphabets. Before this, protein topped out at 2.59e10 (n=28) and DNA at 1.4e10.
#
#  A. PROTEIN - no simulation needed, the data was already staged. The n=4..30 ladder used
#     SIZE-CAPPED subsets; the FULL genera were never benchmarked. Patterns counted directly from
#     each alignment (never inferred):
#       g__Onthovivens     n=34   98,012 patterns -> 2.04e10
#       g__Enterousia      n=54   68,166 patterns -> 2.29e10
#       g__Ruminococcus    n=38  140,491 patterns -> 3.28e10
#       g__Roseburia       n=41  170,890 patterns -> 4.32e10
#       g__Colicola        n=65  156,621 patterns -> 6.37e10
#       g__Ruminococcoides n=66  169,798 patterns -> 7.01e10   (2.7x the previous protein maximum)
#
#  B. DNA - two new AliSim sets sized to clear the protein ceiling:
#       dna_ux_n60_L6M    n=60  4,212,714 patterns -> 3.15e10
#       dna_ux_n100_L4M   n=100 3,300,495 patterns -> 4.16e10
#
#  NO-OVERWRITE: every cell is namespaced __l40s_cx, so each is a NEW RUNS.tsv row. The headline
#  figures filter on cells ending __l40s and are untouched; the complexity figures already plot __l40s_cx.
#
#  MEMORY FROM COUNTED PATTERNS, NEVER FLAT. host_RAM ~= patterns x taxa x C, C measured per
#  alphabet: 600 B for DNA (18 cells) and 2600 B for protein (8 cells, range 1574-2524). Requests
#  therefore span 12 G (protein n=34) to 228 G (DNA n=100) - large only where the tree is large.
#  The DNA cells will run at ~85% of their request; the l40s node has 500 G.
#
#  Usage: submit_ultra_complexity.sh [dry|go] [aa|dna|all]     (default dry all)
# =============================================================================
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
MODE=${1:-dry}; WHAT=${2:-all}
PART=${PART:-${SLURM_PARTITION:-gpu}}
CARD=l40s_cx

# genus  mem_GB  timelimit   (mem = counted_patterns x taxa x 2600 B, rounded up)
AA_CELLS=(
  "g__Onthovivens      12 24:00:00"
  "g__Enterousia       13 24:00:00"
  "g__Ruminococcus     18 36:00:00"
  "g__Roseburia        24 48:00:00"
  "g__Colicola         34 96:00:00"
  "g__Ruminococcoides  36 96:00:00"
)
# tag  mem_GB  timelimit   (mem = counted_patterns x taxa x 600 B, +15% margin)
DNA_CELLS=(
  "dna_ux_n60_L6M   175 24:00:00"
  "dna_ux_n100_L4M  228 30:00:00"
)

mkdir -p "$LOGDIR"
sub () {   # kind  tag  mem  time
  local kind=$1 tag=$2 mem=$3 tl=$4 aln
  if [ "$kind" = aa ]; then aln=$REPO/benchmark/data/protein/$tag/concat_alignment.faa
  else aln=$REPO/benchmark/data/dna/$tag/aln.phy; fi
  [ -s "$aln" ] || { echo "  MISSING $aln"; return; }
  printf "  %-4s %-20s mem=%-5s t=%s\n" "$kind" "$tag" "${mem}G" "$tl"
  [ "$MODE" = go ] || return
  local extra=""; [ "$kind" = dna ] && extra=",MODEL=GTR+F+G4"
  sbatch --job-name="ux_${tag}" --chdir="$REPO" -o "$LOGDIR/ux_${tag}_%j.out" -e "$LOGDIR/ux_${tag}_%j.err" --partition="$PART" --gres=$SLURM_GRES \
         -c 20 --mem=${mem}G --time="$tl" \
         --export=ALL,REPO_DIR="$REPO",GENUS="$tag",THREADS=20,CARD="$CARD",ALN_OVERRIDE="$aln"$extra \
         "$S" >/dev/null || echo "    SUBMIT FAILED $tag"
}

echo "== ultra-complexity cells (mode=$MODE, what=$WHAT, partition $PART) =="
if [ "$WHAT" = all ] || [ "$WHAT" = aa ]; then
  for c in "${AA_CELLS[@]}"; do sub aa $c; done
fi
if [ "$WHAT" = all ] || [ "$WHAT" = dna ]; then
  for c in "${DNA_CELLS[@]}"; do sub dna $c; done
fi
echo "== $([ "$MODE" = go ] && echo submitted || echo 'preview only - pass "go" to submit') =="
