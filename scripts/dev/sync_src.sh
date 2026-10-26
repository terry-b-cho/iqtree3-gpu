#!/usr/bin/env bash
# =============================================================================
#  sync_src.sh - keep the shippable src/ in step with the working build tree.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY THIS EXISTS.
#  build_iqtree_gpu.sh builds a fresh clone from src/iqtree3_gpu.patch + src/gpu_new_files/.
#  Those are SEPARATE COPIES of the sources we actually develop and test in
#  gpu_build/iqtree3_src/tree/. Nothing keeps them in step automatically, so every edit to the
#  kernel silently makes the public repo ship different code from the code we validated.
#
#  This is not hypothetical. On 2026-07-21 src/ was four days stale and a fresh clone would have
#  rebuilt the defective kernel in which the GPU pointer swap was dead code. It was fixed, and then
#  went stale AGAIN within hours as soon as the next kernel change landed.
#
#  So: run `--check` in CI / before any release or benchmark, and `--sync` after any kernel change.
#
#  Usage:
#    sync_src.sh --check    # exit 1 if repo src/ differs from the build tree (no writes)
#    sync_src.sh --sync     # copy sources across, regenerate the patch, verify it applies
# =============================================================================
set -uo pipefail

# --- locate the repository ------------------------------------------------------------------------
# Paths: resolved by scripts/lib/paths.sh; nothing here is hardcoded. Try each candidate source in
# turn and take the FIRST that actually contains scripts/lib/paths.sh. We do NOT special-case the
# SLURM spool path: SLURM execs a COPY of a batch script, so "where am I" is not reliable inside a
# job. Probing in order and validating each candidate works wherever the spool lives.
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

MODE="${1:---check}"
HERE="$REPO"
BUILD_SRC="${IQTREE_SRC:-$IQTREE_BUILD_ROOT/iqtree3_src}"
PIN=6401e06905f65a11ccd6e45f369a1987d657d600
NEW_FILES=(gpu_kernels.cu gpu_kernels.h phylotreegpu.cpp gpu_kernels_test.cu)

[ -d "$BUILD_SRC/tree" ] || { echo "ERROR: build tree not found: $BUILD_SRC"; exit 2; }

# ---- 1. the drop-in source files -------------------------------------------------------------
drift=0
for f in "${NEW_FILES[@]}"; do
  a="$HERE/src/gpu_new_files/$f"; b="$BUILD_SRC/tree/$f"
  if [ ! -f "$b" ]; then echo "  WARN  $f missing from build tree"; continue; fi
  if [ ! -f "$a" ]; then
    echo "  DRIFT $f (absent from src/gpu_new_files)"; drift=1
    [ "$MODE" = --sync ] && cp -p "$b" "$a"
  elif ! cmp -s "$a" "$b"; then
    echo "  DRIFT $f ($(stat -c%s "$a") vs $(stat -c%s "$b") bytes)"; drift=1
    [ "$MODE" = --sync ] && cp -p "$b" "$a"
  else
    echo "  ok    $f"
  fi
done

# ---- 2. the patch against pristine upstream ---------------------------------------------------
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git -C "$BUILD_SRC" diff > "$TMP/regen.patch" 2>/dev/null
if ! cmp -s "$TMP/regen.patch" "$HERE/src/iqtree3_gpu.patch"; then
  echo "  DRIFT iqtree3_gpu.patch ($(wc -l < "$HERE/src/iqtree3_gpu.patch") vs $(wc -l < "$TMP/regen.patch") lines)"
  drift=1
  [ "$MODE" = --sync ] && cp "$TMP/regen.patch" "$HERE/src/iqtree3_gpu.patch"
else
  echo "  ok    iqtree3_gpu.patch"
fi

if [ "$MODE" = --check ]; then
  if [ "$drift" -ne 0 ]; then
    echo
    echo "OUT OF SYNC: the public repo would ship different code than the build tree you tested."
    echo "Fix with:  scripts/sync_src.sh --sync"
    exit 1
  fi
  echo "IN SYNC"
  exit 0
fi

# ---- 3. after syncing, PROVE a pristine clone reconstructs the tested tree ---------------------
echo
echo "== verifying the patch applies to pristine $PIN and reproduces the build tree =="
git clone -q --no-checkout --shared "$BUILD_SRC" "$TMP/iq" 2>/dev/null || { echo "  ERROR: clone failed"; exit 2; }
git -C "$TMP/iq" checkout -q "$PIN" || { echo "  ERROR: cannot check out pin"; exit 2; }
git -C "$TMP/iq" apply --whitespace=nowarn "$HERE/src/iqtree3_gpu.patch" || { echo "  ERROR: patch does NOT apply"; exit 1; }
for f in "${NEW_FILES[@]}"; do
  [ -f "$HERE/src/gpu_new_files/$f" ] && cp "$HERE/src/gpu_new_files/$f" "$TMP/iq/tree/"
done
bad=0
for f in "${NEW_FILES[@]}" phylotreesse.cpp phylotree.h phylotree.cpp; do
  [ -f "$BUILD_SRC/tree/$f" ] || continue
  if cmp -s "$TMP/iq/tree/$f" "$BUILD_SRC/tree/$f"; then echo "  ok    $f identical after patch"
  else echo "  FAIL  $f differs after patch"; bad=1; fi
done
# the swap must be reachable -- the exact defect this guard exists to prevent
if grep -q "GPU acceleration ACTIVE" "$TMP/iq/tree/phylotreesse.cpp" && \
   grep -q "setLikelihoodKernelCPU" "$TMP/iq/tree/phylotreesse.cpp"; then
  echo "  ok    GPU kernel swap is present and reachable in the reconstructed tree"
else
  echo "  FAIL  reconstructed tree lacks the swap wrapper"; bad=1
fi
[ "$bad" -eq 0 ] && echo "SYNCED AND VERIFIED" || { echo "SYNC VERIFICATION FAILED"; exit 1; }
