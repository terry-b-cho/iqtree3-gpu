#!/bin/bash
#SBATCH --job-name=nsyskt
#SBATCH -p gpu_requeue
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: a LITERAL value on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$SLURM_GRES" here is passed to the scheduler verbatim and the allocation fails).
# This is the default for a bare `sbatch scripts/...`; the submitters pass --gres="$SLURM_GRES" on
# the sbatch COMMAND LINE, where the shell does expand it, and a command-line flag overrides the
# directive. Set SLURM_GRES in config/site.env to change it everywhere the submitters are used.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 4:00:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/nsyskt_%j.out
#SBATCH -e logs/nsyskt_%j.err
# =============================================================================
#  Nsight SYSTEMS kernel-time attribution (group L fallback) - works WITHOUT GPU perf counters.
# =============================================================================
#  ncu (HW-counter roofline) fails on this cluster (0 kernels profiled, no error text -- counters
#  likely admin-restricted). nsys only reads the CUDA timeline, so it needs no counters, and gives
#  the number the card comparison needs most: PURE DEVICE KERNEL TIME per run (gpu_kernel_time_s),
#  separating device work from host/launch overhead. kernel_share = kernel_time / gpu_wall directly
#  tests "DNA is launch/host-bound at 20T" and de-confounds the cross-card ranking.
#  Profiling overhead is small but nonzero -> these walls are NOT benchmark timings (gpu_requeue OK).
#  Output: benchmark/profiling/nsys_kernel_summary.tsv (+ per-case .sqlite kept for audit)
# =============================================================================
set -uo pipefail
[ -r /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh 2>/dev/null || true
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
# Resolved from PATH, overridable via NSYS_BIN in config/site.env. A hardcoded cluster path used
# to sit here, which on any other machine exec'd a file that does not exist instead of failing.
NSYS="${NSYS_BIN:-$(command -v nsys || echo nsys)}"
OUT=$REPO/benchmark/profiling; mkdir -p "$OUT"
W=/tmp/nsyskt_$SLURM_JOB_ID; mkdir -p "$W"
echo "== node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) nsys=$($NSYS --version 2>/dev/null | head -1) =="

echo -e "case\ttotal_wall_s\tkernel_time_s\tkernel_share\tmemcpy_time_s\tnum_kernels\ttop_kernel\ttop_kernel_share" > "$OUT/nsys_kernel_summary.tsv"
profile () { # tag aln model
  local tag=$1 aln=$2 model=$3
  echo "--- $tag ($model) ---"
  local t0=$SECONDS
  "$NSYS" profile -t cuda -o "$W/$tag" --force-overwrite true \
    "$BIN" -s "$aln" -m "$model" -T 1 -gpu -seed 1 --prefix "$W/${tag}_run" -redo \
    > "$W/${tag}.app.log" 2> "$OUT/nsys_stderr_${tag}.log" || { echo "  nsys rc=$?"; tail -3 "$OUT/nsys_stderr_${tag}.log" | sed 's/^/  /'; return; }
  local wall=$((SECONDS - t0))
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv "$W/$tag.nsys-rep" -o "$W/${tag}_kern" >/dev/null 2>&1
  "$NSYS" stats --report cuda_gpu_mem_time_sum --format csv "$W/$tag.nsys-rep" -o "$W/${tag}_mem" >/dev/null 2>&1
  python3 - "$W" "$tag" "$wall" "$OUT" <<'PY'
import csv, sys, glob, os
W, tag, wall, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
def read(pat):
    fs = glob.glob(f"{W}/{tag}{pat}*.csv")
    return list(csv.DictReader(open(fs[0]))) if fs else []
kern = read("_kern")
ktot = sum(float(r.get("Total Time (ns)","0") or 0) for r in kern)/1e9
nk   = sum(int(float(r.get("Instances","0") or 0)) for r in kern)
top  = max(kern, key=lambda r: float(r.get("Total Time (ns)","0") or 0), default=None)
mem  = read("_mem")
mtot = sum(float(r.get("Total Time (ns)","0") or 0) for r in mem)/1e9
tname = (top.get("Name","?")[:60] if top else "?")
tshare = (float(top.get("Total Time (ns)","0") or 0)/1e9/ktot if top and ktot else 0)
with open(f"{OUT}/nsys_kernel_summary.tsv","a") as fh:
    fh.write(f"{tag}\t{wall:.1f}\t{ktot:.2f}\t{ktot/wall:.3f}\t{mtot:.2f}\t{nk}\t{tname}\t{tshare:.2f}\n")
print(f"  wall={wall:.0f}s kernel={ktot:.1f}s share={ktot/wall:.1%} kernels={nk} memcpy={mtot:.2f}s")
PY
  cp "$W/$tag.nsys-rep" "$OUT/nsys_${tag}.nsys-rep" 2>/dev/null || true
}
profile protein "$REPO/benchmark/data/protein/n14_g__Nanosyncoccus/concat_alignment.faa" "LG+G4"
profile dna     "$REPO/benchmark/data/dna/dna_n14/aln.phy"                       "GTR+F+G4"
echo "== summary =="; column -t -s$'\t' "$OUT/nsys_kernel_summary.tsv"
