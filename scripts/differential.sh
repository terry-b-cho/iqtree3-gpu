#!/bin/bash
#SBATCH --job-name=gpudiff
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
#SBATCH -t 2:00:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/gpudiff_%j.out
#SBATCH -e logs/gpudiff_%j.err
# =============================================================================
#  IQ-TREE3-GPU - DIFFERENTIAL harness: compare INTERMEDIATE quantities, not just the final tree.
# =============================================================================
#  WHY. An end-to-end "RF = 0 and logL matches" result is a weak test: it is the LAST thing the
#  pipeline produces, so a defect in an intermediate buffer can be masked by the optimiser
#  converging to the same answer anyway. That is how this project shipped, for weeks, a GPU backend
#  whose kernel was never installed at all.
#
#  This harness compares per-site quantities on a FIXED topology, so search divergence cannot
#  confound the comparison and any difference is purely numerical:
#
#    -wsl    per-site log-likelihood            -> exercises _pattern_lh
#    -wslr   per-site logL PER RATE CATEGORY    -> exercises _pattern_lh_cat
#
#  _pattern_lh_cat is the one flagged as most likely to be silently wrong: if the GPU writes it in
#  the wrong interleaving, the tree log-likelihood still comes out right while +R/+I EM and
#  empirical-Bayes site rates go quietly wrong. ModelFinder picked +R in 66 of 66 real runs on this
#  data, so that path is not an edge case -- it is the normal path.
#
#  Deliberately uses NO debug hooks and NO code changes: everything here is stock IQ-TREE output,
#  so the harness cannot itself perturb what it measures.
#
#  Usage: GENUS=g__CAG-269 MODEL=LG+R4 sbatch differential.sh
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
GENUS=${GENUS:-g__CAG-269}
ALN=$REPO/data/protein/$GENUS/concat_alignment.faa
OUT=$REPO/results/validation/differential/${GENUS}
rm -rf "$OUT"; mkdir -p "$OUT"

echo "== differential harness: $GENUS on $(hostname) =="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/   card: /'
[ -f "$ALN" ] || { echo "ERROR: no alignment $ALN"; exit 1; }
head -1 "$ALN" | grep -q "^version https://git-lfs" && { echo "ERROR: $ALN is an unresolved Git LFS pointer"; exit 1; }

# A fixed starting topology, so both arms evaluate the SAME tree and only arithmetic differs.
echo "-- building a fixed reference topology (CPU, parsimony start) --"
"$BIN" -s "$ALN" -m LG -T 20 -n 0 --prefix "$OUT/start" -redo > "$OUT/start.stdout" 2>&1
TREE="$OUT/start.treefile"
[ -f "$TREE" ] || { echo "ERROR: could not build a starting tree"; tail -5 "$OUT/start.stdout"; exit 1; }

fail=0
for MODEL in ${MODELS:-"LG+G4" "LG+R4" "LG+F+I+R5" "WAG+F+R3"}; do
  tag=$(echo "$MODEL" | tr '+' '_')
  echo
  echo "=============== model $MODEL ==============="

  # STEP 1 -- converge the model ONCE on the CPU, purely to obtain its fitted parameters.
  # Fixing the topology with -te is NOT sufficient to make this a kernel comparison: IQ-TREE still
  # optimises model parameters, so the two arms land on slightly different points and we would be
  # measuring optimiser divergence. Observed on 2026-07-21: per-site differences up to 4.6e-02 for
  # +R models while the TOTAL logL agreed to 6.5e-08, with the arms differing in the 4th decimal of
  # a category proportion. At an optimum dlogL/dtheta = 0, so a parameter nudge moves the total by
  # O(dtheta^2) but each site by O(dtheta) -- exactly the pattern seen.
  "$BIN" -s "$ALN" -te "$TREE" -m "$MODEL" -T 20 -seed 1 \
         --prefix "$OUT/${tag}_fit" -redo > "$OUT/${tag}_fit.stdout" 2>&1
  PINNED=$(python3 "$REPO/scripts/fix_model.py" "$OUT/${tag}_fit.iqtree" "$MODEL" 2>/dev/null)
  [ -n "$PINNED" ] || PINNED="$MODEL"
  echo "   pinned model: $PINNED"

  # STEP 2 -- both arms now evaluate the SAME fully-specified model on the SAME topology, so any
  # remaining per-site difference is kernel arithmetic and nothing else.
  for arm in cpu gpu; do
    g=""; [ "$arm" = gpu ] && g="-gpu"
    "$BIN" -s "$ALN" -te "$TREE" -m "$PINNED" -T 20 $g -seed 1 -wslr \
           --prefix "$OUT/${tag}_${arm}" -redo > "$OUT/${tag}_${arm}.stdout" 2>&1
    echo "   $arm rc=$? "
  done
  # Guard: if the pin did not take, the comparison is not a kernel test. Detect it rather than
  # silently reporting a misleading PASS or FAIL.
  for arm in cpu gpu; do
    if grep -qE "Optimizing (free )?rate|Optimizing model parameters" "$OUT/${tag}_${arm}.stdout" 2>/dev/null; then
      echo "   WARNING: $arm still optimised parameters despite the pin -- result is not a pure kernel comparison"
    fi
  done
  # engagement is a precondition: without it both arms are the same CPU kernel and agree trivially
  if ! grep -q "GPU acceleration ACTIVE" "$OUT/${tag}_gpu.stdout"; then
    echo "   *** GPU NOT ENGAGED for $MODEL - comparison below is meaningless ***"; fail=1
  fi
  python3 "$REPO/scripts/compare_sitelh.py" \
          "$OUT/${tag}_cpu.sitelh" "$OUT/${tag}_gpu.sitelh" "$MODEL" || fail=1
done

echo
if [ "$fail" -eq 0 ]; then echo "DIFFERENTIAL: ALL MODELS PASS"; else echo "DIFFERENTIAL: FAILURES PRESENT"; fi
exit $fail
