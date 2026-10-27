#!/usr/bin/env bash
# =============================================================================
#  IQ-TREE3-GPU — worked example  (takes about a minute)
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  What this does, in order:
#    1. builds a tree from the bundled 8-taxon alignment ON THE GPU  (-gpu)
#    2. builds the SAME tree on the CPU (stock IQ-TREE code path)
#    3. proves the two are identical  (Robinson-Foulds distance = 0)
#    4. (optional) draws the tree as a PDF/PNG if R + ape are installed
#
#  Output is streamed live so you can watch ModelFinder, the tree search and the
#  bootstrap run — this is exactly what you'll see on your own data.
#
#  Usage:   ./run_example.sh                          # from the examples/ directory
#           ./run_example.sh /path/to/iqtree3         # explicit binary
#           ./run_example.sh "" data/example_Nanosyncoccus.faa   # a different bundled dataset
#
#  Three small datasets from different genera are bundled (see data/PROVENANCE.md) so you can try
#  more than one tree; they are all equivalent for the purpose of checking your installation.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ALN="${2:-$HERE/data/example_CAG-269.faa}"
[ -f "$ALN" ] || ALN="$HERE/${2:-data/example_CAG-269.faa}"
OUT="$HERE/out"
CHECK="$REPO/scripts/rf_logl_check.py"
PLOT="$REPO/scripts/dev/plot_trees.R"

# --- locate the iqtree3 binary -------------------------------------------------
# priority: explicit argument > in-repo build (build_iqtree_gpu.sh) > $PATH
BIN="${1:-}"
[ -z "$BIN" ] && [ -x "$REPO/iqtree3_gpu_build/iqtree3" ] && BIN="$REPO/iqtree3_gpu_build/iqtree3"
[ -z "$BIN" ] && command -v iqtree3 >/dev/null 2>&1 && BIN="$(command -v iqtree3)"
if [ -z "$BIN" ] || { [ ! -x "$BIN" ] && ! command -v "$BIN" >/dev/null 2>&1; }; then
  cat <<EOF
ERROR: could not find the iqtree3 binary.
       Build it first from the repository root:
           ./build_iqtree_gpu.sh ./iqtree3_gpu_build 89     # 89=L40S 86=A6000 80=A100 70=V100
       ...or pass the path explicitly:   ./run_example.sh /path/to/iqtree3
EOF
  exit 1
fi

# Model choice: LG+G4 is fixed (fast) so the example finishes in about a minute. On real data you would
# normally let ModelFinder choose the model with -m TESTNEW (see the note printed at the end).
MODEL="LG+G4"
BOOT="-B 1000"          # ultrafast bootstrap, 1000 replicates
mkdir -p "$OUT"

echo "==============================================================="
echo " IQ-TREE3-GPU worked example"
echo "==============================================================="
echo "  binary     : $BIN"
echo "  alignment  : ${ALN#$REPO/}   ($(grep -c '^>' "$ALN") taxa)"
echo "  model      : $MODEL   (+ ultrafast bootstrap x1000)"
echo "  output dir : ${OUT#$REPO/}"
echo

# --- 1. GPU run (live output) --------------------------------------------------
echo "---------------------------------------------------------------"
echo " [1/4] GPU run   ->  iqtree3 -gpu -m $MODEL"
echo "       Watch for: '[IQ-TREE3-GPU] GPU acceleration ACTIVE'  (the GPU kernel installed)"
echo "                  '[IQ-TREE3-GPU selftest] ... rel-vs-CPUrecompute=0.000e+00'"
echo "---------------------------------------------------------------"
t0=$SECONDS
"$BIN" -s "$ALN" -m "$MODEL" $BOOT -T 1 -gpu -seed 1 --prefix "$OUT/gpu" -redo 2>&1 \
  | tee "$OUT/gpu.stdout" | grep -vE "^$|^NOTE: |^WARNING: " || true
gpu_t=$((SECONDS - t0))
echo "  -> GPU run finished in ${gpu_t}s"

# Did the GPU kernel actually get installed? The device-count line and the self-test both appear
# even when it did NOT: the self-test calls the CUDA path directly, bypassing the likelihood
# function pointers. Only "GPU acceleration ACTIVE" is printed from inside the pointer-swap block.
if grep -q "GPU acceleration ACTIVE" "$OUT/gpu.stdout"; then
  echo "  -> GPU kernel engaged (verified)"
else
  echo "  !! WARNING: the GPU kernel was NOT engaged - this run used the CPU."
  if grep -q "no CUDA device is visible" "$OUT/gpu.stdout"; then
    echo "     No CUDA device is visible to this process (no GPU here, or none allocated)."
    echo "     That is fine for checking the software runs, but it is not a GPU test."
  else
    echo "     A CUDA device was found but the kernel swap did not apply - please report this."
  fi
  echo "     The rf_dist=0 check below will still pass, because both runs used the CPU kernel."
fi
echo

# --- 2. CPU run (live output) --------------------------------------------------
echo "---------------------------------------------------------------"
echo " [2/4] CPU run   ->  iqtree3 -m $MODEL      (no -gpu: stock code path)"
echo "---------------------------------------------------------------"
t0=$SECONDS
"$BIN" -s "$ALN" -m "$MODEL" $BOOT -T 1 -seed 1 --prefix "$OUT/cpu" -redo 2>&1 \
  | grep -vE "^$|^NOTE: |^WARNING: " || true
cpu_t=$((SECONDS - t0))
echo "  -> CPU run finished in ${cpu_t}s"
echo

# --- 3. the guarantee: identical result ---------------------------------------
echo "---------------------------------------------------------------"
echo " [3/4] Exactness check  (GPU vs CPU: topology + log-likelihood)"
echo "---------------------------------------------------------------"
if command -v python3 >/dev/null 2>&1; then
  python3 "$CHECK" "$OUT/cpu" "$OUT/gpu" example || true
else
  echo "  (python3 not found - skipping automated check)"
fi
echo
echo "  A PASS with rf_dist=0 means the GPU produced EXACTLY the same tree as the CPU."
echo

# --- 4. optional: draw the tree ------------------------------------------------
echo "---------------------------------------------------------------"
echo " [4/4] Tree figure"
echo "---------------------------------------------------------------"
if command -v Rscript >/dev/null 2>&1 && [ -f "$PLOT" ]; then
  Rscript "$PLOT" --single "$OUT/gpu.treefile" "$OUT/example_tree" 2>&1 | sed 's/^/  /' || true
  [ -f "$OUT/example_tree.png" ] && echo "  -> wrote ${OUT#$REPO/}/example_tree.png (and .pdf)"
else
  echo "  Rscript or the plotting script not available - skipping the figure."
  echo "  (optional) install R + the 'ape' package to draw trees, then re-run."
  echo "  A pre-rendered reference is committed at examples/expected_output/example_tree.png"
fi
echo

# --- summary -------------------------------------------------------------------
echo "==============================================================="
echo " Done.  GPU ${gpu_t}s | CPU ${cpu_t}s"
echo "==============================================================="
cat <<EOF
 Your results are in ${OUT#$REPO/}/ :
   gpu.treefile   the maximum-likelihood tree (Newick) built on the GPU
   gpu.iqtree     the full report: model, log-likelihood, branch supports
   gpu.log        the complete run log
   cpu.*          the same files from the CPU run (identical tree)
   example_tree.* the tree drawn as PDF/PNG (if R + ape were available)

 Compare against examples/expected_output/ to confirm your build reproduces ours.

 To use the tool on YOUR data, take your normal IQ-TREE command and add -gpu:
   iqtree3 -gpu -T 20 -s your_alignment.faa -m TESTNEW --ufboot 1000 --alrt 1000 --prefix myrun
   (-m TESTNEW lets ModelFinder pick the best model; -T sets the thread count.
    -T never changes the result, but it is NOT symmetric across the two arms: on the CPU
    it runs N workers over disjoint pattern ranges, while on the GPU a single launch
    computes all patterns per node and -T drives host-side work only.
    See docs/USER_GUIDE.md, "-T N is not symmetric between the two arms".)
EOF
