#!/usr/bin/env bash
# =============================================================================
#  submit_protein_replicates.sh - protein-20T replicate runs (schema group K).
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY. Every protein-20T size had exactly ONE L40S measurement, so no error bar existed - and a
#  single-replicate timing is not publication-grade. The variance is demonstrably material: where a
#  second L40S-class context already exists (__l40sq), speedup differs by +12% to +28%
#  (n6 2.35->2.81, n14 3.04->3.39, n20 4.23->5.23, n30 3.80->4.85). The DNA study found the same
#  shape - same-node run-to-run CV 3-4%, cross-context CV 34-43% - which is exactly what turned the
#  early DNA-20T ">1x" readings into a transient-node artifact. This fills the protein half so
#   Replicate spread is reported by benchmark/scripts/supp_ported.py (FigS02a/FigS03a) and tabulated
#   into benchmark/runs/REPLICATES_{protein,dna}20T.tsv.
#
#  TIERED SCOPE (cost-aware): the 9 sizes under 6 h get +2 runs (-> n=3); n17/n26/n28, which cost
#  15.9/15.5/12.9 h each, get +1 run (-> n=2). 21 jobs, ~73 h, instead of 24 jobs / ~117 h.
#
#  MEMORY IS READ FROM THE MEASURED RUN, NOT GUESSED OR FLAT. Research Computing deprioritised this
#  account for reserving far more RAM than jobs used, so each job here requests
#      ceil(1.8 x peak_ram_mb of that exact cell), floor 4 G
#  taken live from benchmark/runs/RUNS.tsv. Measured protein-20T peaks span 0.56 G (n=4) to
#  6.99 G (n=28), so requests span 4 G to 14 G - only the large/complex trees get large requests.
#  The 1.8x margin covers ModelFinder landing on a slightly different model on a re-run. (Contrast
#  the old flat 96 G, which ran at 7.3% utilisation, and the DNA complexity cells, where the CPU arm
#  genuinely needs 88-118 G - see submit_complexity.sh's sizing note.)
#
#  Usage: submit_protein_replicates.sh [dry|go]      (default dry - prints, submits nothing)
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
RUNS=$REPO/benchmark/runs/RUNS.tsv
MODE=${1:-dry}
PART=${PART:-${SLURM_PARTITION:-gpu}}                       # non-preemptible; a requeued job's wall time is meaningless

# size -> genus, and the wall-time budget (~2.2x the measured cpu+gpu wall of the headline run)
declare -A GEN=( [4]=n04_g__Brotaphodocola [6]=n06_g__CAG-269 [8]=n08_g__Hominicoprocola
                 [10]=n10_g__Coprovivens [12]=n12_g__Caccovivens [14]=n14_g__Nanosyncoccus
                 [17]=n17_g__Lachnospira [20]=n20_g__Coprococcus_C [23]=n23_g__Onthovivens
                 [26]=n26_g__Ruminococcus [28]=n28_g__Roseburia [30]=n30_g__Enterousia )
declare -A TL=( [4]=02:00:00 [6]=02:00:00 [8]=03:00:00 [10]=02:00:00 [12]=05:00:00 [14]=02:00:00
                [17]=36:00:00 [20]=12:00:00 [23]=06:00:00 [26]=36:00:00 [28]=30:00:00 [30]=07:00:00 )
# tier: 2 extra replicates for the cheap sizes, 1 for the three expensive ones
CHEAP=(4 6 8 10 12 14 20 23 30)
PRICEY=(17 26 28)

# mem_for GENUS -> requested GB = ceil(1.8 x measured peak_ram_mb), floor 4
mem_for () {
  awk -F'\t' -v cell="$1__T20__l40s" '
    NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
    $(h["cell"])==cell {mb=$(h["peak_ram_mb"])+0}
    END{ g=int(1.8*mb/1024)+1; if(g<4)g=4; print g }' "$RUNS"
}

mkdir -p "$LOGDIR"
submit () {   # taxa  replicate_suffix
  local n=$1 rep=$2 g=${GEN[$1]} tl=${TL[$1]} mem
  mem=$(mem_for "$g")
  local aln=$REPO/benchmark/data/protein/$g/concat_alignment.faa
  [ -s "$aln" ] || { echo "  MISSING $aln"; return; }
  printf "  n=%-3s %-26s %-3s mem=%-4s t=%s\n" "$n" "$g" "$rep" "${mem}G" "$tl"
  [ "$MODE" = go ] || return
  sbatch --job-name="rep_${g}_${rep}" --chdir="$REPO" -o "$LOGDIR/rep_${g}_${rep}_%j.out" -e "$LOGDIR/rep_${g}_${rep}_%j.err" --partition="$PART" --gres=$SLURM_GRES \
         -c 20 --mem=${mem}G --time="$tl" \
         --export=ALL,REPO_DIR="$REPO",GENUS="$g",THREADS=20,CARD="l40s_$rep",ALN_OVERRIDE="$aln" \
         "$S" >/dev/null || echo "    SUBMIT FAILED n=$n $rep"
}

echo "== protein-20T replicates (mode=$MODE, partition $PART) =="
echo "   memory per job = ceil(1.8 x that cell's MEASURED peak_ram_mb), floor 4 G"
for n in "${CHEAP[@]}"; do submit "$n" r2; submit "$n" r3; done
for n in "${PRICEY[@]}"; do submit "$n" r2; done
tot=$(( ${#CHEAP[@]} * 2 + ${#PRICEY[@]} ))
echo "== $tot jobs $([ "$MODE" = go ] && echo submitted || echo 'previewed (pass "go" to submit)') =="
