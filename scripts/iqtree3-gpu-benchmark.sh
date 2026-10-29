#!/usr/bin/env bash
# iqtree3-gpu-benchmark.sh -- OPTIONAL: run the same analysis twice, once on the CPU and once on
# the GPU, and report the speedup and whether the two agree.
#
# A normal `-gpu` run does NOT do any of this: it runs once, on the GPU, and writes the usual IQ-TREE
# outputs. Use this wrapper only when you want the comparison itself -- a pipeline sanity check, a
# speedup number for your own data, or a line for a methods section. It costs one extra full
# analysis on the CPU.
#
# Usage:  iqtree3-gpu-benchmark.sh <iqtree3> <outprefix> <iqtree arguments...>
#         iqtree3-gpu-benchmark.sh ./iqtree3_gpu_build/iqtree3 mytree -s aln.faa -m LG+F+R4 -T 20 -seed 1
#   Give the analysis exactly as you would to iqtree3, WITHOUT -gpu, --gpu-strict or --prefix: the
#   wrapper adds them. Outputs land in <outprefix>_cpu.* and <outprefix>_gpu.*.
# Requires python3 (standard library only) for the agreement verdict.
# Paths: the repository root is resolved from this file's own location. Nothing is hardcoded.
set -uo pipefail
if [ $# -lt 3 ]; then
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2
fi
BIN="$1"; PREFIX="$2"; shift 2
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$DIR/rf_logl_check.py"
PY="${IQTREE_GPU_PY:-python3}"
for a in "$@"; do case "$a" in
    -gpu|--gpu-lik|--gpu-strict|-gpu-strict|--gpu-newton|-gpu-newton) echo "ERROR: do not pass $a; the wrapper adds the GPU options itself"; exit 2;;
    --prefix|-pre) echo "ERROR: do not pass $a; give the output prefix as the second argument"; exit 2;;
esac; done
command -v "$BIN" >/dev/null 2>&1 || { echo "ERROR: iqtree3 not found: $BIN"; exit 1; }
[ -f "$CHECK" ] || { echo "ERROR: $CHECK not found"; exit 1; }
# -seed is required for the two arms to follow the same search; add one if the user gave none.
seed=""; for a in "$@"; do [ "$a" = -seed ] || [ "$a" = --seed ] && seed=given; done
[ -n "$seed" ] || set -- "$@" -seed 1

echo "==============================================================="
echo " IQ-TREE3-GPU benchmark: CPU arm vs GPU arm (same command, same seed)"
echo "==============================================================="
echo " binary : $(command -v "$BIN")"; "$BIN" --version 2>&1 | head -1 | sed 's/^/ version: /'
echo " args   : $*"
echo "---------------------------------------------------------------"
echo " [1/3] CPU arm  -> ${PREFIX}_cpu.*"
t0=$(date +%s.%N); "$BIN" "$@" --prefix "${PREFIX}_cpu" -redo > "${PREFIX}_cpu.stdout" 2>&1; cpu_rc=$?
cpu_t=$(echo "$(date +%s.%N) - $t0" | bc)
echo "       exit $cpu_rc, wall $(printf '%.1f' "$cpu_t") s"
echo " [2/3] GPU arm  -> ${PREFIX}_gpu.*   (--gpu-strict: any CPU fallback is an error, so the"
echo "       timing below is a GPU measurement or nothing)"
t0=$(date +%s.%N); "$BIN" "$@" --gpu-strict --prefix "${PREFIX}_gpu" -redo > "${PREFIX}_gpu.stdout" 2>&1; gpu_rc=$?
gpu_t=$(echo "$(date +%s.%N) - $t0" | bc)
echo "       exit $gpu_rc, wall $(printf '%.1f' "$gpu_t") s"
grep -E "GPU acceleration ACTIVE|branch-length Newton loop|GPUSTATS gpu_calls|--gpu-strict" "${PREFIX}_gpu.stdout" | sed 's/^/       /' | tail -4
echo "---------------------------------------------------------------"
echo " [3/3] agreement"
"$PY" "$CHECK" "${PREFIX}_cpu" "${PREFIX}_gpu" "$(basename "$PREFIX")"; agree=$?
echo "---------------------------------------------------------------"
if [ "$cpu_rc" -eq 0 ] && [ "$gpu_rc" -eq 0 ]; then
    printf " wall time: CPU %.1f s   GPU %.1f s   speedup %.2fx (CPU wall / GPU wall)\n" "$cpu_t" "$gpu_t" "$(echo "$cpu_t / $gpu_t" | bc -l)"
    echo " (Speedup depends on tree complexity and thread count; the published medians are in README.md.)"
else
    echo " one arm failed (cpu exit $cpu_rc, gpu exit $gpu_rc); see ${PREFIX}_cpu.stdout / ${PREFIX}_gpu.stdout"
fi
[ "$cpu_rc" -eq 0 ] && [ "$gpu_rc" -eq 0 ] && [ "$agree" -eq 0 ]
