#!/usr/bin/env bash
# =============================================================================
#  submit_model_coverage.sh - MODEL-COVERAGE cells (additive; nothing is overwritten).
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY. The equivalence guarantee (RF=0, logL within 1e-4, GPU genuinely engaged) is currently
#  demonstrated on 24 distinct PROTEIN models -- ModelFinder picks a different one per genus -- but
#  on exactly ONE DNA model: every DNA row in RUNS.tsv is GTR+F+G4. Partitioned models have never
#  been run at all (mf_flags is "none" in every row). This closes both gaps, converting the
#  supported-model matrix in docs/DEPENDENCIES.md from a claim into a measurement.
#
#  This is a CORRECTNESS-GENERALITY test, not a speed test. The cells are deliberately the cheapest
#  measured alignments (dna_n06/n08/n10: 2-12 s CPU wall, 340-554 MB peak RAM), because the question
#  is "does the GPU path stay exact under a different rate model", not "how fast is it".
#
#  NO-OVERWRITE RULE. The cell directory is ${GENUS}__T${THREADS}__${CARD}, so the MODEL must be
#  encoded in CARD or four models on one alignment would collide in a single directory. Namespaces
#  here are `mdl_*`, which deliberately do NOT end in `__l40s*`: every headline figure filters on an
#  exact `__l40s` / `__l40s_cx` / `__l40s(q|_r\d+)` suffix (the figure scripts,
#  fresh_build_accept.sh:63), so these rows cannot perturb any published curve. CARD is a namespace only -- the
#  `card` column is filled from nvidia-smi (paired_benchmark.sh:89), so the hardware is still recorded truthfully.
#
#  Usage: submit_model_coverage.sh [dry|go]        (default dry)
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
MODE=${1:-dry}
PART=${PART:-${SLURM_PARTITION:-gpu_quad}}          # non-preemptible; gpu_qos caps this user at 34 CPUs and is busy
RUNS=$REPO/benchmark/runs/RUNS.tsv

# tag  genus      model          namespace   -- namespace encodes the model, genus the alignment
CELLS=(
  "dna_n06  HKY+F+G4     mdl_hky"    # simpler rate matrix (transition/transversion only)
  "dna_n06  GTR+F+R4     mdl_r4"     # FREE-RATE on DNA: a rate-heterogeneity path never exercised
  "dna_n06  GTR+F+I+G4   mdl_ig"     # invariant sites + gamma
  "dna_n06  JC           mdl_jc"     # degenerate edge case: no rate matrix, no base frequencies
  "dna_n10  HKY+F+G4     mdl_hky"    # confirm the result holds at a second tree size
  "dna_n10  GTR+F+R4     mdl_r4"
)

mkdir -p "$LOGDIR"
sub () {
  local genus=$1 model=$2 card=$3
  local aln=$REPO/benchmark/data/dna/$genus/aln.phy
  local cell="${genus}__T20__${card}"
  [ -s "$aln" ] || { echo "  MISSING $aln"; return 1; }
  # refuse to touch an existing successful measurement
  if awk -F'\t' -v c="$cell" 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next} $1==c && $(h["gate_pass"])=="TRUE"{f=1} END{exit !f}' "$RUNS"; then
    echo "  SKIP (already measured, gate_pass=TRUE): $cell"; return 0
  fi
  printf "  %-14s %-12s -> %-28s" "$genus" "$model" "$cell"
  if [ "$MODE" != go ]; then echo "   [dry]"; return 0; fi
  local jid
  jid=$(sbatch --parsable --job-name="mdl_${genus}_${card}" --chdir="$REPO" -o "$LOGDIR/mdl_${genus}_${card}_%j.out" -e "$LOGDIR/mdl_${genus}_${card}_%j.err" --partition="$PART" --gres=$SLURM_GRES \
        -c 20 --mem=4G --time=1:00:00 \
        --export=ALL,REPO_DIR="$REPO",GENUS="$genus",THREADS=20,CARD="$card",ALN_OVERRIDE="$aln",MODEL="$model" \
        "$S") && echo "   job=$jid" || echo "   SUBMIT FAILED"
}

echo "== model-coverage cells (mode=$MODE, partition=$PART) =="
for c in "${CELLS[@]}"; do sub $c; done
echo "== $([ "$MODE" = go ] && echo submitted || echo 'preview only - pass "go" to submit') =="
