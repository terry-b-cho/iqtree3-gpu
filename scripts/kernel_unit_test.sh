#!/bin/bash
#SBATCH --job-name=kerntest
#SBATCH -p gpu_requeue
# NOTE: a LITERAL value on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$SLURM_GRES" here is passed to the scheduler verbatim and the allocation fails).
# This is the default for a bare `sbatch scripts/...`; the submitters pass --gres="$SLURM_GRES" on
# the sbatch COMMAND LINE, where the shell does expand it, and a command-line flag overrides the
# directive. Set SLURM_GRES in config/site.env to change it everywhere the submitters are used.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:30:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/kerntest_%j.out
#SBATCH -e logs/kerntest_%j.err
# First execution of the rewritten eigenspace GPU kernels. Validates partials, theta, logL,
# df and ddf against an in-file CPU reference at rel tol 1e-12 (rungs 1-3 of the ladder).
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
i3g_load_modules || exit 1
GB=$IQTREE_BUILD_ROOT/iqtree3_src/tree
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
echo "== node $(hostname) =="; nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/   card: /'
echo "== compiling =="
# -lcublas: the MCMCTree Hessian is a single cublasDsyrk over the device-resident G matrix.
# nvcc links cudart implicitly but never cuBLAS, so omitting this fails only at link time.
nvcc -std=c++17 -arch=sm_89 -fmad=false -O2 -o "$W/kt" "$GB/gpu_kernels.cu" "$GB/gpu_kernels_test.cu" -lcublas 2>&1 | tail -20
[ -x "$W/kt" ] || { echo "COMPILE FAILED"; exit 1; }
echo "== running =="
"$W/kt"; echo "== test exit=$? =="
