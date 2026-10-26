#!/usr/bin/env bash
# =============================================================================
#  refresh_all.sh - regenerate every table, figure, workbook and the figure map from RUNS.tsv, in dependency order.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  This is the sequence to run after any batch of jobs lands. Order matters: the aggregator must
#  publish RUNS.tsv first, the replicate means feed the DNA/protein 20T figures, and the interim
#  report embeds figures produced by everything before it.
#
#  Nothing here overwrites a successful measurement: aggregate_runs.py upserts by cell key, snapshots the table
#  before writing, and ABORTS if a successful row would vanish or change (see its guard).
#
#  Usage: bash benchmark/scripts/refresh_all.sh
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
cd "$REPO" || exit 1

run () { printf "  %-28s " "$1"; python3 "benchmark/scripts/$1.py" "${@:2}" >/tmp/_i3g.$$ 2>&1 \
         && tail -1 /tmp/_i3g.$$ || { echo "FAILED"; tail -3 /tmp/_i3g.$$ | sed 's/^/      /'; }; }

echo "== 1. aggregate (publishes RUNS.tsv/.csv + RUNS_summary.*) =="
run aggregate_runs
echo "== 1b. key numbers (single source of truth for every number quoted in the docs) =="
run key_numbers
echo "== 2. per-step wall + CPU time from each per-run IQ-TREE logs -> PHASE_STEPS.tsv =="
run phase_extract
echo "== 3. main-text figure panels (benchmark/figures/panels/) =="
for s in equivalence decision mechanism \
         phases_cost hardware memory; do run "$s"; done
echo "== 4. supplementary panels + replicate tables =="
for s in supp_ported supplementary trees; do run "$s"; done
echo "== 4b. assembled previews + greyscale proof =="
run assemble
echo "== 4c. lettered composite figures (one file per figure, as the journal sets them) =="
run composites
echo "== 5. enforce the figure contract =="
run panels
echo "== 6. workbooks: Source Data (per panel) + Supplementary Data (the tables), styled =="
if python3 -c "import pandas, openpyxl" 2>/dev/null; then
  run source_data
  run supplementary_data
  run style_workbooks benchmark/figures/Source_Data.xlsx benchmark/runs/Supplementary_Data_benchmark.xlsx
else
  echo "  (skipped: pandas + openpyxl not installed; the committed workbooks are left as they are)"
fi
echo "== 7. figure and table map in benchmark/README.md =="
run benchmark_map
rm -f /tmp/_i3g.$$
echo "== done: $(( $(wc -l < benchmark/runs/RUNS.tsv) - 1 )) rows in RUNS.tsv =="
