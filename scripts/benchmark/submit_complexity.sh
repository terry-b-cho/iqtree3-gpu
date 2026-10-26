#!/usr/bin/env bash
# =============================================================================
#  submit_complexity.sh - submit the COMPLEXITY-GAP cells (additive; nothing is overwritten).
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  Closes the measured gap where DNA (5.9e4..4.8e8) and protein (2.2e8..2.6e10) barely overlapped:
#    A. dna_pt_n<NN>   DNA simulated on the PROTEIN run's own inferred topology (n=14..30), length
#                      chosen so DNA complexity matches that protein cell -> the paired points
#    B. dna_hc_*       high-complexity DNA (taxa > 30 and sites > 233k, never explored before)
#    C. p_lc_*         low-complexity protein (column-subsampled real alignments, never run before)
#
#  NO-OVERWRITE RULE: every cell uses CARD=l40s_cx, so its directory is <genus>__T<threads>__l40s_cx
#  and its RUNS.tsv row is NEW. The headline figures/tables filter on cells ending in `__l40s`
#  (the retired validation scripts), so the 124 existing measurements and every headline curve are untouched.
#
#  MEMORY IS SIZED PER CELL FROM MEASUREMENT, not requested flat. Research Computing flagged that our
#  jobs used <half of the RAM they reserved (measured across 124 runs: 1.13 GB mean, 6.99 GB max
#  against a 96 GB request), which cost us scheduler priority.
#
#  SIZING RULE (recalibrated 2026-08-31 after three OOM failures). The first model sized the DEVICE
#  arena (patterns x 128 B x 1.3 x taxa) and under-predicted badly. The lesson from the failures:
#  **the CPU arm, not the GPU arm, sets the host-memory requirement.** All three OOM cells finished
#  their GPU arm normally (which keeps partials in VRAM) and then died in the CPU arm, which holds
#  every partial in host RAM. The empirical rule, from 18 measured DNA cells, is
#
#      host_RAM  ~=  patterns x taxa x C,     C = 256-697 B  (350-500 B for the large cells)
#
#  so we size at C = 600 B, above every measured cell but far below a flat reservation. `patterns`
#  is COUNTED from the alignment (see the counter in build_complexity_datasets.sh), never inferred from a pattern/site
#  ratio - that ratio rises with taxa (0.52 at n=30, 0.76 at n=60, 0.85 at n=100) and guessing it is
#  what caused the original under-request.
#
#  Usage: submit_complexity.sh [all|pt|hc|lc]     (default all)
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
S=$REPO/scripts/benchmark/paired_benchmark.sh
WHAT=${1:-all}
PART=${PART:-${SLURM_PARTITION:-gpu}}                 # non-preemptible (PreemptMode=OFF); timings must not be requeued
CARD=l40s_cx                      # cell namespace for the complexity extension (hardware is l40s)

# submit CELL_DIR TAXA THREADS MEM_GB TIME  -- DNA (fixed model, PHYLIP under data/dna/)
mkdir -p "$LOGDIR"
sub_dna () {
  local tag=$1 thr=$2 mem=$3 tl=$4
  local aln=$REPO/data/dna/$tag/aln.phy
  [ -s "$aln" ] || { echo "  MISSING $aln"; return; }
  local cpus=20; [ "$thr" = 1 ] && cpus=4
  sbatch --job-name="cx_${tag}_T${thr}" --chdir="$REPO" -o "$LOGDIR/cx_${tag}_T${thr}_%j.out" -e "$LOGDIR/cx_${tag}_T${thr}_%j.err" --partition="$PART" --gres=$SLURM_GRES \
         -c $cpus --mem=${mem}G --time="$tl" \
         --export=ALL,REPO_DIR="$REPO",GENUS="$tag",THREADS="$thr",CARD="$CARD",ALN_OVERRIDE="$aln",MODEL="GTR+F+G4" \
         "$S" >/dev/null && echo "  submitted $tag T$thr (mem=${mem}G t=$tl)"
}
# submit protein cell -- full ModelFinder, FASTA under data/protein/
sub_aa () {
  local tag=$1 thr=$2 mem=$3 tl=$4
  local aln=$REPO/data/protein/$tag/concat_alignment.faa
  [ -s "$aln" ] || { echo "  MISSING $aln"; return; }
  local cpus=20; [ "$thr" = 1 ] && cpus=4
  sbatch --job-name="cx_${tag}_T${thr}" --chdir="$REPO" -o "$LOGDIR/cx_${tag}_T${thr}_%j.out" -e "$LOGDIR/cx_${tag}_T${thr}_%j.err" --partition="$PART" --gres=$SLURM_GRES \
         -c $cpus --mem=${mem}G --time="$tl" \
         --export=ALL,REPO_DIR="$REPO",GENUS="$tag",THREADS="$thr",CARD="$CARD",ALN_OVERRIDE="$aln" \
         "$S" >/dev/null && echo "  submitted $tag T$thr (mem=${mem}G t=$tl)"
}

# ---- A. paired DNA on protein topologies (20T). mem/time from the predicted arena + 2-4x margin ----
do_pt () {
  sub_dna dna_pt_n14  20  8 04:00:00
  sub_dna dna_pt_n17  20 32 12:00:00
  sub_dna dna_pt_n20  20 32 12:00:00
  sub_dna dna_pt_n23  20 24 08:00:00
  sub_dna dna_pt_n26  20 40 12:00:00
  sub_dna dna_pt_n28  20 88 16:00:00   # 4,836,971 patterns x 28 -> ~81 G needed
  sub_dna dna_pt_n30  20 24 08:00:00
}
# ---- B. high-complexity DNA ladder (20T; the smallest also at 1T to extend the 1-thread curve) ----
do_hc () {
  sub_dna dna_hc_n30_L1M   20  8 04:00:00
  sub_dna dna_hc_n30_L3M   20 16 06:00:00
  sub_dna dna_hc_n60_L2M   20 24 08:00:00
  sub_dna dna_hc_n60_L4M   20 118 16:00:00  # 3,028,907 patterns x 60 -> ~109 G needed
  sub_dna dna_hc_n100_L2M  20 110 16:00:00  # 1,691,383 patterns x 100 -> ~101 G needed
  sub_dna dna_hc_n30_L1M    1  8 12:00:00
}
# ---- C. low-complexity protein (20T + 1T; tiny, so the floor request of 8G is already generous) ----
do_lc () {
  for tag in p_lc_n04_L200 p_lc_n04_L2k p_lc_n04_L20k p_lc_n06_L20k p_lc_n08_L50k; do
    sub_aa "$tag" 20 8 04:00:00
    sub_aa "$tag"  1 8 08:00:00
  done
}

echo "== submitting complexity-gap cells (namespace __${CARD}, partition $PART) =="
case "$WHAT" in
  pt) do_pt ;; hc) do_hc ;; lc) do_lc ;;
  all) do_pt; do_hc; do_lc ;;
  *) echo "usage: $0 [all|pt|hc|lc]"; exit 1 ;;
esac
echo "== queued: $(squeue -u "$USER" -h -o %j | grep -c '^cx_') complexity cells =="
