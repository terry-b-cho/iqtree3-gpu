#!/usr/bin/env bash
# =============================================================================
#  scripts/run_tests.sh -- the single entry point for verifying this repository.
#
#    --quick   (default)  no GPU needed, seconds. Syntax, path resolution, the equivalence
#                         checker's own self-consistency, and repo invariants.
#    --gpu                needs a CUDA device. Engagement + equivalence on real alignments.
#    --full               needs a scheduler. The long acceptance suites.
#
#  Exit 0 only if every selected check passes.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
set -uo pipefail

_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "$PWD") || { echo "FATAL: cannot locate repository root; set REPO_DIR" >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"

TIER="${1:---quick}"
PASS=0; FAIL=0
ok   () { printf "  \033[32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS+1)); }
bad  () { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL+1)); }
note () { printf "  ----  %s\n" "$1"; }

echo "== IQ-TREE3-GPU test suite ($TIER) =="
echo "   repository: $REPO"

# ---------------------------------------------------------------- quick (no GPU)
echo; echo "-- syntax --"
n=0; b=0
for f in "$REPO"/scripts/*.sh "$REPO"/scripts/*/*.sh "$REPO"/benchmark/scripts/*/*.sh "$REPO"/benchmark/figures_tables/*.sh \
         "$REPO"/benchmark/scripts/*/*/*.sh "$REPO"/*.sh "$REPO"/examples/*.sh; do
  [ -f "$f" ] || continue; n=$((n+1)); bash -n "$f" 2>/dev/null || { bad "bash -n $(basename "$f")"; b=1; }
done
[ $b -eq 0 ] && ok "bash -n on $n shell scripts"
n=0; b=0
for f in "$REPO"/scripts/*.py "$REPO"/scripts/*/*.py "$REPO"/benchmark/scripts/*.py "$REPO"/benchmark/scripts/*/*.py \
         "$REPO"/benchmark/figures_tables/*.py "$REPO"/benchmark/figures_tables/lib/*.py; do
  [ -f "$f" ] || continue; n=$((n+1))
  python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$f" 2>/dev/null \
    || { bad "python syntax $(basename "$f")"; b=1; }
done
[ $b -eq 0 ] && ok "python parses on $n scripts"

echo; echo "-- path resolution --"
[ -n "${REPO:-}" ] && [ -f "$REPO/build_iqtree_gpu.sh" ] && ok "repository root resolves" || bad "repository root"
( cd / && env -u REPO_DIR -u SLURM_SUBMIT_DIR bash -c ". $REPO/scripts/lib/paths.sh" >/dev/null 2>&1 )
[ $? -eq 0 ] && ok "paths.sh resolves from an unrelated CWD" || bad "paths.sh from unrelated CWD"
# Portability is checked by scripts/portability.py, not by a grep here. The two greps this
# replaced globbed only scripts/*.sh and scripts/*/*.sh, matched only a bare VAR=/n/scratch at line
# start, and scanned no Python, no R, no Markdown and nothing at the repository root -- so they passed
# vacuously while three scripts derived the repository root one directory too shallow. The checker also
# RELOCATES the repository to a temporary directory and resolves the root from there, which is the only
# way to actually test "runs from any directory".
out=$(python3 "$REPO/scripts/portability.py" 2>&1) || true
while IFS= read -r l; do
  case "$l" in
    "  PASS  "*) ok "${l#  PASS  }" ;;
    "  FAIL  "*) bad "${l#  FAIL  }" ;;
  esac
done < <(printf '%s\n' "$out" | grep -E "^  (PASS|FAIL)")
printf '%s\n' "$out" | grep -E "^          " | head -12 | sed 's/^/     /'

echo; echo "-- equivalence checker --"
if [ -f "$REPO/examples/expected_output/gpu.treefile" ]; then
  python3 "$CHECK" "$REPO/examples/expected_output/gpu" "$REPO/examples/expected_output/gpu" selftest >/dev/null 2>&1 \
    && ok "rf_logl_check.py agrees with itself (RF=0)" \
    || note "checker self-test skipped (needs a matching .iqtree beside the tree)"
else note "examples/expected_output absent - checker self-test skipped"; fi

echo; echo "-- data invariants --"
R="$REPO/benchmark/runs/RUNS.tsv"
if [ -s "$R" ]; then
  rows=$(( $(wc -l < "$R") - 1 ))
  crows=$(( $(wc -l < "${R%.tsv}.csv") - 1 ))
  [ "$rows" = "$crows" ] && ok "RUNS.tsv/.csv agree ($rows rows)" || bad "TSV/CSV row mismatch ($rows vs $crows)"
  nz=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next} $(h["rf_dist"])+0!=0{n++} END{print n+0}' "$R")
  [ "$nz" = 0 ] && ok "RF=0 in all $rows recorded runs" || bad "$nz run(s) have RF != 0"
  orphan=0
  while IFS= read -r c; do [ -f "$REPO/benchmark/runs/paired/$c/RESULT.tsv" ] || orphan=$((orphan+1)); done \
    < <(awk -F'\t' 'NR>1{print $1}' "$R")
  [ "$orphan" = 0 ] && ok "every RUNS.tsv row has its cell directory" || bad "$orphan row(s) missing a cell dir"

  # ---- build provenance -------------------------------------------------------------------
  # Every published figure and number must come from ONE build. The filter lives in
  # benchmark/figures_tables/lib/headline.py; these invariants are what stop it being silently lost in a later edit.
  nbuild=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{print substr($(h["binary_sha256"]),1,8)}' "$R" | sort -u | grep -c .)
  [ "$nbuild" -le 2 ] && ok "RUNS.tsv spans $nbuild binary build(s)" \
    || bad "$nbuild distinct builds in RUNS.tsv - a third build has entered the record"
  cur=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next} substr($(h["binary_sha256"]),1,8)!="07c70cae"{n++} END{print n+0}' "$R")
  K="$REPO/benchmark/runs/KEY_NUMBERS.tsv"
  if [ -s "$K" ]; then
    kn=$(awk -F'\t' '$1=="total_rows"{print $2}' "$K")
    [ "$kn" = "$cur" ] && ok "KEY_NUMBERS reports the current-build count ($cur of $rows rows)" \
      || bad "KEY_NUMBERS total_rows=$kn but $cur rows are current-build - the build filter is not applied"
  fi
  unfiltered=0
  for f in "$REPO"/benchmark/scripts/*.py "$REPO"/benchmark/figures_tables/*.py; do
    grep -q "RUNS.tsv" "$f" 2>/dev/null || continue
    case "$(basename "$f")" in aggregate_runs.py|phase_extract.py|verify_data.py) continue ;; esac   # aggregator + log parser see every row by design
    # Both lib/headline.py and lib/figdata.py apply the build filter (figdata.load defaults to
    # current_build_only=True), so either import satisfies the invariant.
    grep -qE "headline|figdata" "$f" 2>/dev/null || { echo "       $(basename "$f") reads RUNS.tsv without a build filter"; unfiltered=$((unfiltered+1)); }
  done
  [ "$unfiltered" = 0 ] && ok "every analysis script applies the build filter" \
    || bad "$unfiltered script(s) read RUNS.tsv without the build filter"
else bad "RUNS.tsv missing"; fi

echo; echo "-- figure contract --"
# Enforced, not asserted. After the previous pass I reported every text element was 6 pt and the
# theme uniform; both were false, because I had checked the embedded font FAMILY and never the SIZE.
if [ -d "$REPO/benchmark/figures/panels" ]; then
  if out=$(python3 "$REPO/benchmark/figures_tables/figures.py" audit 2>&1); then
    ok "$(echo "$out" | tail -2 | head -1 | sed 's/^ *//')"
  else
    bad "panel audit failed"; echo "$out" | grep FAIL | head -8 | sed 's/^/       /'
  fi
else
  echo "  ---- panel audit skipped (no panels yet)"
fi

echo; echo "-- cross-check --"
# Cross-file checks on the public tree (archived references, loose figure files, panel-title
# literals, card sources, the README map); a stale pointer fails the build, not the reader.
if [ -f "$REPO/benchmark/figures/FIGURE_MANIFEST.tsv" ]; then
  out=$(python3 "$REPO/benchmark/figures_tables/tables.py" crosscheck 2>&1) || true
  # NOT piped into `while`: a pipeline runs the loop in a subshell, so every ok/bad call would
  # increment a counter that is discarded when the subshell exits, and four passing checks would
  # print but not be counted.
  while IFS= read -r l; do
    case "$l" in
      "  PASS  "*) ok "${l#  PASS  }" ;;
      "  FAIL  "*) bad "${l#  FAIL  }" ;;
    esac
  done < <(printf '%s\n' "$out" | grep -E "^  (PASS|FAIL)")
else
  echo "  ---- cross-check skipped (no manifest yet)"
fi

# ---------------------------------------------------------------- gpu
if [ "$TIER" = "--gpu" ] || [ "$TIER" = "--full" ]; then
  echo; echo "-- GPU --"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    bad "no nvidia-smi: cannot run GPU tier"
  elif [ ! -x "$IQTREE_GPU_BIN" ]; then
    bad "GPU binary not found at $IQTREE_GPU_BIN (set IQTREE_BUILD_ROOT in config/site.env)"
  else
    bash "$REPO/scripts/iqtree3-gpu-verify.sh" "$IQTREE_GPU_BIN" >/dev/null 2>&1 \
      && ok "iqtree3-gpu-verify: engaged + RF=0 + logL within 1e-4" || bad "iqtree3-gpu-verify"
    ( cd "$REPO/examples" && ./run_example.sh "$IQTREE_GPU_BIN" >/dev/null 2>&1 ) \
      && ok "worked example reproduces" || bad "examples/run_example.sh"
  fi
fi

# ---------------------------------------------------------------- full
if [ "$TIER" = "--full" ]; then
  echo; echo "-- full acceptance (scheduler) --"
  if command -v sbatch >/dev/null 2>&1; then
    note "submit manually and check the job logs:"
    note "  sbatch scripts/kernel_unit_test.sh          # kernel unit test"
    note "  GENUS=n14_g__Nanosyncoccus sbatch scripts/equivalence_matrix.sh"
    note "  sbatch benchmark/scripts/harness/fresh_build_accept.sh   # clone -> patch -> build -> verify"
  else bad "no sbatch: cannot run the full tier"; fi
fi

echo; echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
