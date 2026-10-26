#!/usr/bin/env bash
# =============================================================================
#  submit_binary_rerun.sh -- re-measure the cells recorded with the SUPERSEDED binary.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY. `binary_sha256` splits RUNS.tsv into two builds: 138 rows on edf783c3... (Aug-Sep) and 44 on
#  07c70cae... (July). Both are IQ-TREE 3.1.3 + this repo's GPU patch -- the same tool, two builds of
#  our own code ~10 days apart -- but they do not behave identically. On identical DNA alignments the
#  July build issues ~2.1-2.5x FEWER GPU calls and is correspondingly faster at every size
#  (dna_n28: 1.82x / 318,563 calls vs 0.91-0.95x / 776,801 calls). `dna_n17__T20__l40s_n202chk` is a
#  same-node control on the newer build, so this is the binary, not the node.
#
#  All 44 old-build cells sit in the `__l40s` HEADLINE namespace, which is exactly what the figure
#  generators filter on -- so the published ladders are currently drawn from the superseded build.
#
#  SAFETY. Everything here is ADDITIVE:
#    * a new namespace, CARD=l40s_b2, so cell dirs are <genus>__T<thr>__l40s_b2 and the 44 existing
#      `__l40s` cells are never written to. No figure changes until its filter is repointed.
#    * paired_benchmark.sh independently refuses to rm -rf a cell whose RESULT.tsv has
#      gate_pass=TRUE unless FORCE=1, so even a mistyped namespace cannot destroy a measurement.
#    * the binary is asserted before anything is submitted (see EXPECT_SHA): a silently rebuilt
#      binary would otherwise introduce a THIRD build mid-campaign.
#    * memory and walltime come from the matching `__l40s` row's own measured peak_ram_mb and walls,
#      not from guesses -- Research Computing penalises over-reservation.
#    * non-preemptible partitions only; a requeued job would silently corrupt a wall measurement.
#
#  Usage: submit_binary_rerun.sh [dry|go]     (default dry)
# =============================================================================
set -uo pipefail
_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "${SLURM_SUBMIT_DIR:-}") || _d=$(_i3g_try "$PWD") \
  || { echo "FATAL: cannot locate the repository root; set REPO_DIR" >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"

MODE=${1:-dry}
CARD=l40s_b2                       # the new namespace: every row is an addition
OLD_SHA_PREFIX=07c70cae            # the superseded build we are replacing
EXPECT_SHA=edf783c3                # the build every re-run must use
PART=${PART:-${SLURM_PARTITION:-gpu_quad}}
RUNS="$REPO/benchmark/runs/RUNS.tsv"
S="$REPO/benchmark/scripts/harness/paired_benchmark.sh"

# ---- assert the binary BEFORE doing anything ------------------------------------------------
[ -x "$IQTREE_GPU_BIN" ] || { echo "FATAL: GPU binary not found: $IQTREE_GPU_BIN" >&2; exit 1; }
ACTUAL_SHA=$(sha256sum "$IQTREE_GPU_BIN" | cut -c1-8)
echo "== binary check =="
echo "   path   : $IQTREE_GPU_BIN"
echo "   sha256 : ${ACTUAL_SHA}...  (expected ${EXPECT_SHA}...)"
if [ "$ACTUAL_SHA" != "$EXPECT_SHA" ]; then
  echo "   ABORT: the binary on disk is neither the superseded build nor the one these re-runs are" >&2
  echo "          meant to characterise. Submitting now would create a THIRD build in the record." >&2
  echo "          Rebuild to ${EXPECT_SHA} or update EXPECT_SHA deliberately." >&2
  exit 1
fi
echo "   OK: matches the build used by the other 138 rows."
echo

mkdir -p "$LOGDIR"
echo "== cells recorded on the superseded build (namespace -> $CARD) =="
printf "   %-38s %-8s %-4s %-9s %-7s %s\n" CELL TYPE THR MEM TIME "old speedup"

submitted=0; skipped=0; missing=0
while IFS=$'\t' read -r cell dtype thr genus aln mem_mb cpu_s gpu_s old_sp; do
  [ -z "$cell" ] && continue
  new_cell="${genus}__T${thr}__${CARD}"

  # never touch an existing successful measurement in the target namespace either
  if awk -F'\t' -v c="$new_cell" 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}
       $1==c && $(h["gate_pass"])=="TRUE"{f=1} END{exit !f}' "$RUNS"; then
    printf "   %-38s SKIP (already measured on this build)\n" "$new_cell"; skipped=$((skipped+1)); continue
  fi
  if [ ! -s "$REPO/$aln" ]; then
    printf "   %-38s MISSING ALIGNMENT %s\n" "$new_cell" "$aln"; missing=$((missing+1)); continue
  fi

  # memory: 1.8x the measured peak of the matching __l40s row, floor 4 G
  mem_g=$(awk -v m="$mem_mb" 'BEGIN{g=int(1.8*m/1024)+1; if(g<4)g=4; print g}')
  # walltime: 3x the measured combined wall (both arms run serially in one job), floor 1 h, cap 4 d
  tl=$(awk -v c="$cpu_s" -v g="$gpu_s" 'BEGIN{
        h=3*(c+g)/3600; if(h<1)h=1; if(h>96)h=96;
        d=int(h/24); r=h-24*d; hh=int(r); mm=int((r-hh)*60);
        if(d>0) printf "%d-%02d:%02d:00", d, hh, mm; else printf "%02d:%02d:00", hh, mm}')
  model=$([ "$dtype" = dna ] && echo "GTR+F+G4" || echo "TESTNEW")
  cpus=20; [ "$thr" = 1 ] && cpus=4

  printf "   %-38s %-8s %-4s %-9s %-7s %s" "$new_cell" "$dtype" "$thr" "${mem_g}G" "$tl" "$old_sp"
  if [ "$MODE" != go ]; then echo "   [dry]"; continue; fi
  jid=$(sbatch --parsable --job-name="b2_${genus}_T${thr}" --partition="$PART" --gres=$SLURM_GRES \
        --chdir="$REPO" -o "$LOGDIR/b2_${genus}_T${thr}_%j.out" -e "$LOGDIR/b2_${genus}_T${thr}_%j.err" \
        -c "$cpus" --mem="${mem_g}G" --time="$tl" \
        --export=ALL,REPO_DIR="$REPO",GENUS="$genus",THREADS="$thr",CARD="$CARD",ALN_OVERRIDE="$REPO/$aln",MODEL="$model" \
        "$S") && { echo "   job=$jid"; submitted=$((submitted+1)); } || echo "   SUBMIT FAILED"
done < <(
  awk -F'\t' -v old="$OLD_SHA_PREFIX" -v mod="$REPO/" '
    NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
    substr($(h["binary_sha256"]),1,8)==old {
      cmd=$(h["command_line_cpu"]); aln="";
      if (match(cmd, /-s [^ ]+/)) { aln=substr(cmd, RSTART+3, RLENGTH-3); sub(mod, "", aln) }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        $1, $(h["data_type"]), $(h["threads"]), $(h["genus"]), aln,
        $(h["peak_ram_mb"]), $(h["cpu_wall_s"]), $(h["gpu_wall_s"]), $(h["speedup"])
    }' "$RUNS" | sort
)
echo
echo "== $submitted submitted, $skipped already done, $missing missing input =="
[ "$MODE" = go ] || echo "   preview only - pass \"go\" to submit"
