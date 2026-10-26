#!/bin/bash
#SBATCH --job-name=equiv
#SBATCH -p gpu_requeue
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: a LITERAL value on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$SLURM_GRES" here is passed to the scheduler verbatim and the allocation fails).
# This is the default for a bare `sbatch scripts/...`; the submitters pass --gres="$SLURM_GRES" on
# the sbatch COMMAND LINE, where the shell does expand it, and a command-line flag overrides the
# directive. Set SLURM_GRES in config/site.env to change it everywhere the submitters are used.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH -t 8:00:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/equiv_%j.out
#SBATCH -e logs/equiv_%j.err
# =============================================================================
#  IQ-TREE3-GPU - EQUIVALENCE MATRIX (Phase 1 science gate)
# =============================================================================
#  Gate per cell:  logL within 1e-4  AND  RF = 0  AND  GPU engagement proven.
#  NOT bit-identity -- CPU IQ-TREE is not even bit-reproducible against itself across thread
#  counts (computeLikelihoodDerv accumulates under an OpenMP reduction).
#
#  MODEL COVERAGE IS DATA-DRIVEN, not guessed. Across 66 archived real runs on this dataset
#  ModelFinder selected: VT/LG/WAG/PMB/Q.PFAM matrices, +F on every single one, +I on roughly
#  half, and FreeRate +R2..+R10 -- with +G4 chosen ZERO times. So the +R path is the normal path
#  here, not an edge case, and it gets the most cells. +G4 is kept only as a control.
#
#  This is correctness-only, so gpu_requeue (preemptible) is fine: a preempted job is requeued and
#  re-run, which costs time but cannot corrupt a PASS/FAIL verdict. TIMING runs must never use it.
#
#  Usage:  GENUS=g__CAG-269 sbatch equivalence_matrix.sh
#          THREADS="1 4 20"  MODELS="LG+G4 LG+F+R4"  to narrow it.
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
GENUS=${GENUS:-g__CAG-269}
ALN=$REPO/benchmark/data/protein/$GENUS/concat_alignment.faa
OUT=$REPO/validation/equivalence/$GENUS
TSV=$REPO/validation/equivalence_matrix.tsv
mkdir -p "$OUT" "$(dirname "$TSV")"

[ -f "$ALN" ] || { echo "ERROR: no alignment $ALN"; exit 1; }
head -1 "$ALN" | grep -q "^version https://git-lfs" && { echo "ERROR: $ALN is an unresolved Git LFS pointer"; exit 1; }

CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
NTAX=$(grep -c '^>' "$ALN")
echo "== equivalence matrix: $GENUS ($NTAX taxa) on $(hostname), $CARD =="
[ -s "$TSV" ] || echo -e "genus\ttaxa\tmodel\tthreads\textra\tcard\tengaged\tverdict\trf\tlogl_cpu\tlogl_gpu\treldiff" > "$TSV"

MODELS=${MODELS:-"LG+F+R4 VT+F+I+R5 WAG+F+I+R3 PMB+F+R6 Q.PFAM+F+I+R7 LG+F+I+R10 LG+G4"}
THREADS=${THREADS:-"1 2 4 8 20"}   # 2 and 8 added: packets = threads*2, and only 1/4/20 had ever been validated
fail=0

cell () {  # model threads extra_label extra_flags...
  local model=$1 thr=$2 lab=$3; shift 3
  local tag; tag=$(echo "${model}_T${thr}_${lab}" | tr '+ ' '__')
  echo; echo "--- $model  -T $thr  [$lab] ---"
  for arm in cpu gpu; do
    local g=""; [ "$arm" = gpu ] && g="-gpu"
    "$BIN" -s "$ALN" -m "$model" -T "$thr" $g -seed 1 "$@" \
           --prefix "$OUT/${tag}_${arm}" -redo > "$OUT/${tag}_${arm}.stdout" 2>&1
  done
  # Engagement first: without it both arms are the same CPU kernel and any agreement is vacuous.
  local eng=INVALID
  grep -q "GPU acceleration ACTIVE" "$OUT/${tag}_gpu.stdout" && eng=OK
  local out v rf lc lg rd
  out=$(python3 "$CHECK" "$OUT/${tag}_cpu" "$OUT/${tag}_gpu" "$tag" 2>&1)
  echo "   engagement=$eng"; echo "   $out"
  v=$(echo "$out"  | grep -oP 'VERDICT\[[^]]*\]: \K\w+')
  rf=$(echo "$out" | grep -oP 'rf_dist=\K\S+')
  lc=$(echo "$out" | grep -oP 'logl_a=\K\S+')
  lg=$(echo "$out" | grep -oP 'logl_b=\K\S+')
  rd=$(echo "$out" | grep -oP 'logl_reldiff=\K\S+')
  [ "$eng" = OK ] && [ "$v" = PASS ] || fail=1
  echo -e "$GENUS\t$NTAX\t$model\t$thr\t$lab\t$CARD\t$eng\t${v:-NA}\t${rf:-NA}\t${lc:-NA}\t${lg:-NA}\t${rd:-NA}" >> "$TSV"
}

for m in $MODELS; do
  for t in $THREADS; do cell "$m" "$t" plain; done
done
# Support values (UFBoot + SH-aLRT) are the third of the gate that logL and RF do not cover.
cell "LG+F+R4" 20 support --ufboot 1000 --alrt 1000
# SAFE_LH is a different scaling code path (per-pattern-per-category counters, rescale test inside
# the category loop). Our genera are far below the 2000-taxon auto-trigger, so it is only reachable
# via -safe -- and therefore only ever tested if we test it explicitly.
cell "LG+F+R4" 20 safe -safe

echo
echo "=============== SUMMARY ($GENUS) ==============="
column -t "$TSV" | tail -n +1 | grep -E "^genus|$GENUS" | tail -20
echo
[ "$fail" -eq 0 ] && echo "EQUIVALENCE: ALL CELLS PASS" || echo "EQUIVALENCE: FAILURES PRESENT"
exit $fail
