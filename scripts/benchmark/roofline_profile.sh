#!/bin/bash
#SBATCH --job-name=roofline
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
#SBATCH -o logs/roofline_%j.out
#SBATCH -e logs/roofline_%j.err
# =============================================================================
#  Nsight Compute roofline profiling (schema group L) - the compute- vs memory-bound VERDICT.
# =============================================================================
#  Measures, per GPU kernel, on one protein and one DNA case:
#    achieved_dram_gbs      = dram__bytes.sum / gpu__time_duration.sum
#    achieved_fp64_gflops   = (dadd + dmul + 2*dfma) / gpu__time_duration.sum
#    arithmetic intensity   = fp64 FLOPs / DRAM bytes
#    roofline_bound         = compute if AI > ridge (fp64_peak/bandwidth), else memory
#  PROFILING IS NOT TIMING: ncu replays kernels, so wall time here is meaningless by design ->
#  gpu_requeue is acceptable. Steady-state sampling: skip the first 200 launches, profile 60.
#  Output: results/profiling/nsight_raw_{protein,dna}.csv + nsight_summary.tsv
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
# Resolved from PATH, overridable via NCU_BIN in config/site.env. A hardcoded cluster path used
# to sit here, which on any other machine exec'd a file that does not exist instead of failing.
NCU="${NCU_BIN:-$(command -v ncu || echo ncu)}"
OUT=$REPO/results/profiling; mkdir -p "$OUT"
W=/tmp/roofline_$SLURM_JOB_ID; mkdir -p "$W"
METRICS="gpu__time_duration.sum,dram__bytes.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"

echo "== node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) ncu=$($NCU --version | tail -1) =="

# ---- PROBE 1: are the metric names valid on this ncu? ----
echo "--- metric-name probe ---"
for m in gpu__time_duration dram__bytes smsp__sass_thread_inst_executed_op_dfma_pred_on; do
  $NCU --query-metrics 2>/dev/null | grep -qw "$m" && echo "  OK $m" || echo "  MISSING metric: $m"
done
# ---- PROBE 2: counter permission (ERR_NVGPUCTRPERM shows here if profiling is admin-restricted) ----
echo "--- permission probe (1 kernel, default metrics) ---"
$NCU --launch-count 1 --target-processes all "$BIN" -s "$REPO/examples/data/example_CAG-269.faa"   -m LG+G4 -T 1 -gpu -seed 1 --prefix "$W/probe" -redo > "$W/probe.out" 2> "$OUT/nsight_probe_stderr.log" || true
grep -iE "ERR_NVGPUCTRPERM|permission|not supported|Failed" "$OUT/nsight_probe_stderr.log" | head -3 | sed "s/^/  /"
grep -c "==PROF== Profiling" "$W/probe.out" 2>/dev/null | sed "s/^/  profiled-kernel markers: /"

profile () { # tag aln model
  local tag=$1 aln=$2 model=$3
  echo "--- profiling $tag ($model) ---"
  "$NCU" --csv --metrics "$METRICS" --launch-skip 200 --launch-count 60 \
         --target-processes all --profile-from-start on \
         "$BIN" -s "$aln" -m "$model" -T 1 -gpu -seed 1 --prefix "$W/$tag" -redo \
         > "$OUT/nsight_raw_${tag}.csv" 2> "$OUT/nsight_stderr_${tag}.log" || \
         { echo "  ncu rc=$?"; tail -5 "$OUT/nsight_stderr_${tag}.log" | sed "s/^/  ncu-err: /"; }
  grep -c '^"' "$OUT/nsight_raw_${tag}.csv" 2>/dev/null | sed 's/^/  csv rows: /'
}
profile protein "$REPO/data/protein/n14_g__Nanosyncoccus/concat_alignment.faa" "LG+G4"
profile dna     "$REPO/data/dna/dna_n14/aln.phy"                       "GTR+F+G4"

# ---- reduce per-kernel: FLOPs, bytes, AI, achieved rates; classify vs the CARD's ridge ----
python3 - "$OUT" <<'PY'
import csv, sys, collections, os
OUT = sys.argv[1]
# L40S: fp64 peak ~1.4e12 FLOP/s (1/64 rate), bandwidth 864e9 B/s -> ridge = 1.62 FLOP/B
FP64_PEAK, BW = 1.4e12, 864e9
rows_out = [["case","kernel","launches","time_ms","dram_gb","fp64_gflop","ai_flop_per_byte",
             "achieved_dram_gbs","achieved_fp64_gflops","pct_fp64_peak","pct_bw","roofline_bound"]]
for case in ("protein","dna"):
    p = f"{OUT}/nsight_raw_{case}.csv"
    if not os.path.exists(p): continue
    lines = [l for l in open(p, errors="ignore") if l.startswith('"')]
    rd = list(csv.DictReader(lines))
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    cnt = collections.Counter()
    for r in rd:
        k = r.get("Kernel Name","?").split("(")[0][:60]
        m = r.get("Metric Name",""); v = r.get("Metric Value","0").replace(",","")
        try: agg[k][m] += float(v)
        except ValueError: pass
        if m == "gpu__time_duration.sum": cnt[k] += 1
    for k, m in agg.items():
        t = m.get("gpu__time_duration.sum",0) * 1e-9   # ns -> s (ncu reports ns for .sum)
        if t <= 0: continue
        fl = m.get("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",0) + \
             m.get("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",0) + \
             2*m.get("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",0)
        by = m.get("dram__bytes.sum",0)
        ai = fl/by if by else 0
        bound = "compute" if ai > FP64_PEAK/BW else "memory"
        rows_out.append([case,k,cnt[k],f"{t*1e3:.2f}",f"{by/1e9:.3f}",f"{fl/1e9:.3f}",f"{ai:.3f}",
                         f"{by/t/1e9:.1f}",f"{fl/t/1e9:.1f}",
                         f"{fl/t/FP64_PEAK*100:.1f}",f"{by/t/BW*100:.1f}",bound])
with open(f"{OUT}/nsight_summary.tsv","w",newline="") as fh:
    csv.writer(fh,delimiter="\t").writerows(rows_out)
print(f"nsight_summary.tsv: {len(rows_out)-1} kernel rows")
for r in rows_out[1:]:
    print(f"  {r[0]:8s} {r[1][:44]:44s} AI={r[6]:>8s} {r[11]}")
PY
echo "== done =="
