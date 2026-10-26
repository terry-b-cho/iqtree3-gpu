# shellcheck shell=bash
# =============================================================================
#  scripts/lib/paths.sh -- resolve the repository root and the tool binaries.
#  SOURCE this file; never execute it.  . "<repo>/scripts/lib/paths.sh"
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY THIS IS NOT A ONE-LINER
#
#  The obvious idiom -- walk up from "${BASH_SOURCE[0]}" -- is CORRECT for a script you run
#  directly and SILENTLY WRONG for a script you sbatch. SLURM copies the batch script to the
#  compute node and execs the copy, so inside a job BASH_SOURCE is
#      /var/spool/slurmd/job<ID>/slurm_script
#  and a walk-up from there resolves the repository root to the node's spool directory. The benchmark
#  then computes OUT=$REPO/benchmark/runs/paired/<cell>, `rm -rf`s it and writes a measurement
#  into node-local scratch that disappears when the job ends. That is the exact failure this file
#  exists to prevent, so the rule here is: NEVER GUESS. Resolve from an authoritative source or
#  exit loudly.
#
#  RESOLUTION ORDER (first hit wins)
#    1. $REPO_DIR                -- explicit. The ONLY reliable source inside a SLURM job; the
#                                  submitters export it via --export=ALL,REPO_DIR=...
#    2. this file's own location -- ../.. from scripts/lib/. Fixed, known depth, so no caller has
#                                  to count "..". Rejected if it points into /var/spool.
#    3. $SLURM_SUBMIT_DIR       -- walked upward to the marker. Covers `cd <repo> && sbatch ...`
#    4. $PWD                    -- walked upward to the marker. Covers running from a subdirectory.
#    5. hard exit 78            -- with a message telling the caller to set REPO_DIR.
#
#  EXPORTS: REPO, REPO_DIR, LOGDIR, IQTREE_GPU_BIN, IQTREE_CPU_BIN, GPUBIN, STOCK, CHECK, AGG
# =============================================================================

_i3g_is_root() {            # $1 is the repository root iff both markers are present
  [ -n "${1:-}" ] && [ -f "$1/scripts/lib/paths.sh" ] && [ -f "$1/build_iqtree_gpu.sh" ]
}

_i3g_walk_up() {            # print the nearest ancestor of $1 that is the repository root
  local d="${1:-}"
  [ -n "$d" ] || return 1
  d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do
    _i3g_is_root "$d" && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d")
  done
  return 1
}

_i3g_resolve() {
  local cand=""
  # 1. explicit
  if [ -n "${REPO_DIR:-}" ] && _i3g_is_root "$REPO_DIR"; then printf '%s' "$REPO_DIR"; return 0; fi
  # 2. this file's own location. No spool-path special case: the candidate is VALIDATED with
  #    _i3g_is_root, so a spool copy simply fails the test and we fall through. That works wherever
  #    a given site configures its spool, instead of assuming /var/spool.
  local self="${BASH_SOURCE[0]:-}"
  if [ -n "$self" ]; then
    cand=$(cd "$(dirname "$self")/../.." 2>/dev/null && pwd) || cand=""
    _i3g_is_root "$cand" && { printf '%s' "$cand"; return 0; }
  fi
  # 3. the directory sbatch was invoked from
  cand=$(_i3g_walk_up "${SLURM_SUBMIT_DIR:-}") && { printf '%s' "$cand"; return 0; }
  # 4. the current directory
  cand=$(_i3g_walk_up "$PWD") && { printf '%s' "$cand"; return 0; }
  return 1
}

if ! REPO=$(_i3g_resolve); then
  {
    echo "FATAL: cannot locate the repository root."
    echo "  Inside a SLURM job the script runs from a spool copy and cannot find itself, so the"
    echo "  repository root must be passed in explicitly. Either:"
    echo "     REPO_DIR=/path/to/iqtree3-gpu sbatch <script>"
    echo "  or submit from the repository root so SLURM_SUBMIT_DIR points at it:"
    echo "     cd /path/to/iqtree3-gpu && sbatch scripts/..."
  } >&2
  exit 78
fi
export REPO REPO_DIR="$REPO"
LOGDIR="${LOGDIR:-$REPO/logs}"
export LOGDIR
# Create it here, once, rather than in each submitter. Note this only helps a job whose -o/-e are
# passed on the sbatch COMMAND LINE (as the submitters do): for a bare `sbatch scripts/...` slurmd
# opens the relative log path before this file is ever sourced, which is why logs/.gitkeep is tracked.
mkdir -p "$LOGDIR" 2>/dev/null || true

# ---- site configuration ---------------------------------------------------------------------
# config/site.env is gitignored and per-user; config/site.env.example is the tracked, documented
# fallback whose defaults reproduce the environment this campaign was measured on, so an
# unconfigured checkout behaves exactly as before.
for _i3g_f in "${SITE_ENV:-}" "$REPO/config/site.env" "$REPO/config/site.env.example"; do
  if [ -n "$_i3g_f" ] && [ -r "$_i3g_f" ]; then . "$_i3g_f"; break; fi
done
unset _i3g_f

# ---- scheduler and toolchain ------------------------------------------------------------------
# These four were documented in site.env for a long time and read by NOTHING: every script
# hardcoded `module load gcc/14.2.0 cuda/12.8`, `--gres=gpu:l40s:1` and a partition name, so a user
# on another cluster who configured them as instructed saw no effect. They are exported here and
# consumed by the benchmark and dev scripts, with the values this campaign ran on as the defaults,
# so behaviour on O2 is unchanged.
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
SLURM_GRES="${SLURM_GRES:-gpu:l40s:1}"
SLURM_MODULES="${SLURM_MODULES:-gcc/14.2.0 cuda/12.8}"
NO_LMOD="${NO_LMOD:-0}"
NCU_BIN="${NCU_BIN:-}"
NSYS_BIN="${NSYS_BIN:-}"
export SLURM_PARTITION SLURM_GRES SLURM_MODULES NO_LMOD NCU_BIN NSYS_BIN

# i3g_load_modules -- the ONE place `module load` happens. Honours NO_LMOD, which is what its
# documentation has always promised and what makes a workstation (no Lmod) usable.
i3g_load_modules () {
  [ "${NO_LMOD:-0}" = "1" ] && return 0
  [ -r /etc/profile.d/lmod.sh ] && . /etc/profile.d/lmod.sh 2>/dev/null || true
  command -v module >/dev/null 2>&1 || {
    echo "ERROR: Lmod unavailable on $(hostname). Set NO_LMOD=1 in config/site.env if this" >&2
    echo "       machine has no module system and the toolchain is already on PATH." >&2
    return 1; }
  # shellcheck disable=SC2086
  module load $SLURM_MODULES || { echo "ERROR: module load $SLURM_MODULES failed" >&2; return 1; }
}

# ---- binaries -------------------------------------------------------------------------------
# ONE variable, ONE default, NO search order. A "prefer the in-repo build, else the external one"
# fallback would silently swap which binary produced a measurement the first time somebody ran
# build_iqtree_gpu.sh, making new rows incomparable with the recorded ones.
IQTREE_GPU_BIN="${GPUBIN_OVERRIDE:-${IQTREE_GPU_BIN:-$IQTREE_BUILD_ROOT/iqtree3_gpu_build/iqtree3}}"
IQTREE_CPU_BIN="${IQTREE_CPU_BIN:-$IQTREE_BUILD_ROOT/iqtree3_build/iqtree3}"
GPUBIN="$IQTREE_GPU_BIN"      # legacy names kept so existing scripts need no edit
STOCK="$IQTREE_CPU_BIN"
CHECK="$REPO/scripts/rf_logl_check.py"
AGG="$REPO/benchmark/scripts/aggregate_runs.py"
export IQTREE_GPU_BIN IQTREE_CPU_BIN GPUBIN STOCK CHECK AGG

# ---- introspection ----------------------------------------------------------------------------
# PATH_DEBUG=1 prints what resolved and exits 0, without running any payload. This is how a
# refactor is proved path-identical: capture before, capture after, diff.
if [ -n "${PATH_DEBUG:-}" ]; then
  for _i3g_v in REPO LOGDIR IQTREE_BUILD_ROOT IQTREE_GPU_BIN IQTREE_CPU_BIN GPUBIN STOCK CHECK AGG \
                SLURM_PARTITION SLURM_GRES SLURM_MODULES NO_LMOD NCU_BIN NSYS_BIN \
                EIGEN3_INCLUDE_DIR Boost_INCLUDE_DIR; do
    eval "printf '%s=%s\n' \"\$_i3g_v\" \"\${$_i3g_v-}\""
  done
  unset _i3g_v
  return 0 2>/dev/null || exit 0
fi
