#!/bin/bash
#SBATCH --job-name=launchlat
#SBATCH -p gpu
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: literal values on purpose; SLURM does not expand shell variables in #SBATCH directives.
# Submit from the repository root, or with REPO_DIR=/path/to/iqtree3-gpu.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 0:30:00
#SBATCH -o logs/launchlat_%j.out
#SBATCH -e logs/launchlat_%j.err
# =============================================================================
#  Empty-kernel launch latency on the L40S (author action R14, round 8).
# =============================================================================
#  Compiles scripts/benchmark/launch_latency.cu with the campaign toolchain and runs it once on one
#  L40S. Writes results/profiling/launch_latency_l40s.txt (full stdout, with the device header) and
#  results/profiling/launch_latency_l40s.tsv (the two data rows), which key_numbers.py reads
#  into launch_latency_us_l40s_median and companions. Not a benchmark timing; any node will do.
# =============================================================================
set -uo pipefail
_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") \
  || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "${SLURM_SUBMIT_DIR:-}") \
  || _d=$(_i3g_try "$PWD") \
  || { echo "FATAL: cannot locate the repository root. Submit from the repository root or set REPO_DIR." >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"
i3g_load_modules || exit 1
OUT=$REPO/results/profiling; mkdir -p "$OUT"
W=/tmp/launchlat_${SLURM_JOB_ID:-$$}; mkdir -p "$W"
echo "== node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) nvcc=$(nvcc --version | tail -1) =="
nvcc -O2 -arch=sm_89 -o "$W/launch_latency" "$REPO/scripts/benchmark/launch_latency.cu" || exit 1
"$W/launch_latency" 100000 | tee "$OUT/launch_latency_l40s.txt"
grep -v '^#' "$OUT/launch_latency_l40s.txt" > "$OUT/launch_latency_l40s.tsv"
echo "== wrote $OUT/launch_latency_l40s.{txt,tsv} =="
rm -rf "$W"
