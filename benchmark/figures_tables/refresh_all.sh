#!/usr/bin/env bash
# =============================================================================
#  refresh_all.sh - regenerate every table, number, figure, workbook and the figure map from the
#  measured record, in dependency order. No IQ-TREE, no GPU: about 4 minutes.
# =============================================================================
#  Two tiers. The benchmark HARNESS (benchmark/scripts/: the SLURM harness, aggregate_runs.py,
#  phase_extract.py) produced and records the measurements; this directory (benchmark/figures_tables/)
#  turns the record into every figure and table. Steps 1 and 2 call the record builders; everything
#  after that is figures.py and tables.py.
#
#  Nothing here overwrites a successful measurement: aggregate_runs.py upserts by cell key, snapshots
#  the table before writing, and ABORTS if a successful row would vanish or change (see its guard).
#
#  Usage: bash benchmark/figures_tables/refresh_all.sh
#  Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# =============================================================================
set -uo pipefail
_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") \
  || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "${SLURM_SUBMIT_DIR:-}") \
  || _d=$(_i3g_try "$PWD") \
  || { echo "FATAL: cannot locate the repository root. Run from the repository, or set REPO_DIR=/path/to/iqtree3-gpu." >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"
cd "$REPO" || exit 1

FT=benchmark/figures_tables
run () { printf "  %-28s " "$1 ${2:-}"; python3 "$@" >/tmp/_i3g.$$ 2>&1 \
         && tail -1 /tmp/_i3g.$$ || { echo "FAILED"; tail -3 /tmp/_i3g.$$ | sed 's/^/      /'; }; }

echo "== 1. aggregate (publishes RUNS.tsv/.csv + RUNS_summary.*) =="
run benchmark/scripts/aggregate_runs.py
echo "== 2. per-step wall + CPU time from each per-run IQ-TREE log -> PHASE_STEPS.tsv =="
run benchmark/scripts/phase_extract.py
echo "== 3. key numbers (single source of truth for every number quoted in the docs) =="
run $FT/tables.py key_numbers
echo "== 4. figure panels (benchmark/figures/panels/) and the replicate table =="
for s in equivalence decision mechanism phases_cost hardware supp_ported supplementary trees; do run $FT/figures.py "$s"; done
echo "== 5. lettered composite figures (one file per figure, as the journal sets them) =="
run $FT/figures.py composites
echo "== 6. enforce the figure contract =="
run $FT/figures.py audit
echo "== 7. workbooks: Source Data (per panel) + Supplementary Data (the tables), styled =="
if python3 -c "import pandas, openpyxl" 2>/dev/null; then
  run $FT/tables.py source_data
  run $FT/tables.py supplementary_data
  run $FT/tables.py style_workbooks benchmark/figures/Source_Data.xlsx benchmark/runs/Supplementary_Data_benchmark.xlsx
else
  echo "  (skipped: pandas + openpyxl not installed; the committed workbooks are left as they are)"
fi
echo "== 8. figure and table map in benchmark/README.md =="
run $FT/tables.py benchmark_map
rm -f /tmp/_i3g.$$
echo "== done: $(( $(wc -l < benchmark/runs/RUNS.tsv) - 1 )) rows in RUNS.tsv =="
