#!/usr/bin/env bash
# =============================================================================
#  build_dna_datasets.sh - reproducible DNA benchmark set, mirroring the AA sweep.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY SIMULATED. This repository's staged data is amino acid (concat_alignment.faa); there are no
#  nucleotide alignments, and back-translating the AA alignments would need CDS we do not have.
#  IQ-TREE ships AliSim, so we generate DNA directly. That is arguably BETTER for a tool benchmark:
#    * fully reproducible from a seed and a model string -- anyone can regenerate byte-identical input
#    * sizes chosen deliberately rather than dictated by what happened to be staged
#    * the TRUE model is known, so model recovery is checkable, not just self-consistency
#  These are benchmark inputs for a likelihood engine, not data for a biological claim, so simulated
#  sequence is entirely appropriate here.
#
#  The size sweep mirrors the AA set (n = 4..30 at 12 distinct sizes) so protein and DNA speedups are
#  compared at matched tree sizes rather than across different shapes.
#
#  Usage: build_dna_datasets.sh [outdir]
# =============================================================================
set -uo pipefail
[ -r /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh 2>/dev/null || true
[ "${NO_LMOD:-0}" = "1" ] || { command -v module >/dev/null 2>&1 && \
  module load $SLURM_MODULES >/dev/null 2>&1; } || true

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
BIN=${IQTREE_BIN:-$IQTREE_BUILD_ROOT/iqtree3_gpu_build/iqtree3}
OUT=${1:-$REPO/data/dna}
mkdir -p "$OUT"

# Matched to the AA sweep: 12 distinct tree sizes.
SIZES=(4 6 8 10 12 14 17 20 23 26 28 30)
# Alignment lengths mirroring the AA sweep's range (~99k-235k columns), varied independently of n so
# the taxa and sites axes can be separated in the scaling fit rather than moving together.
LENS=(195456 143055 169245 122439 114107 99220 205341 206383 134971 212930 233159 109330)

# A general-purpose DNA model with all the machinery the GPU kernel must handle:
# GTR (full 6-rate reversible), empirical-style base frequencies, and rate heterogeneity.
# Explicit parameters keep the simulation reproducible and the true values known.
MODEL='GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}'

echo "== building DNA benchmark set in $OUT =="
echo "   model: $MODEL"
built=0
for i in "${!SIZES[@]}"; do
  n=${SIZES[$i]}; L=${LENS[$i]}
  tag=$(printf "dna_n%02d" "$n")
  d="$OUT/$tag"; mkdir -p "$d"
  # -seed derived from n so each dataset is independently reproducible, and a rerun of one dataset
  # cannot perturb the others.
  "$BIN" --alisim "$d/aln" -t "RANDOM{yh/$n}" -m "$MODEL" --length "$L" \
         -seed $((1000 + n)) -redo > "$d/alisim.log" 2>&1
  if [ -f "$d/aln.phy" ]; then
    got_n=$(head -1 "$d/aln.phy" | awk '{print $1}')
    got_L=$(head -1 "$d/aln.phy" | awk '{print $2}')
    printf "   %-10s n=%-3s len=%-7s  (requested n=%s len=%s)\n" "$tag" "$got_n" "$got_L" "$n" "$L"
    built=$((built+1))
  else
    echo "   FAILED $tag"; tail -3 "$d/alisim.log" | sed 's/^/      /'
  fi
done
echo "== built $built of ${#SIZES[@]} DNA datasets =="
echo "   true model recorded for each: $MODEL"
echo "$MODEL" > "$OUT/TRUE_MODEL.txt"
