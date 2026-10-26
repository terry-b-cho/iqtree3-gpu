#!/bin/bash
#SBATCH --job-name=gpurebuild
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 2:00:00
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/gpurebuild_%j.out
#SBATCH -e logs/gpurebuild_%j.err
# Rebuild the GPU binary after the setLikelihoodKernel swap-hook fix (IQ-TREE3-GPU).
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
cd $IQTREE_BUILD_ROOT/iqtree3_gpu_build || exit 1
echo "== rebuilding in $(pwd) on $(hostname) =="
cmake --build . -j 16 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "== build rc=$rc =="
[ $rc -eq 0 ] || exit $rc
ls -l ./iqtree3
echo "== does the rebuilt binary still contain the banner strings? =="
strings ./iqtree3 | grep -c "GPU acceleration ACTIVE"
