#!/bin/bash
# =============================================================================
#  IQ-TREE3-GPU - GPU-vs-GPU CARD SWEEP (schema group J) - SEPARATE from the headline benchmark.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  Same genus, same thread count, same seed, same binary; ONLY the GPU model changes. When the
#  card is the variable, a reader needs (schema benchmark/BENCHMARK_SCHEMA.md section J):
#    (a) the card's STATIC intrinsic specs, to explain WHY it wins/loses - protein tracks
#        fp64_tflops (compute-bound), DNA tracks mem_bandwidth_gbs (memory-bound);
#    (b) proof it ran at FULL clocks (sustained sm/mem clock, temp, throttle flag) - a throttled
#        card is not being measured at spec;
#    (c) control of the HOST CONFOUND - a different GPU usually means a different node and thus a
#        different host CPU, which changes the host-side portion of the run.
#  This harness adds exactly those group-J fields on top of the core benchmark row that
#  paired_benchmark.sh already produces. It does NOT re-implement paired_benchmark.sh: it RUNS paired_benchmark.sh for
#  the card (same problem, same seed, --gpu-strict, engagement-gated) and then joins (i) the static
#  specs from card_specs.tsv and (ii) runtime nvidia-smi samples taken during the GPU arm.
#
#  TWO MODES (one file, so the batch job that samples the GPU is the same code that is documented):
#    submit (default) : sbatch THIS script once per card, pinned to a NON-preemptible partition.
#    run  (CS_RUN=1)  : execute on the compute node -> run paired_benchmark.sh inline, sample group J, emit row.
#
#  PARTITION CHOICE IS A CORRECTNESS ISSUE HERE, not convenience. This sweep measures TIME, and
#  gpu_requeue has PreemptMode=REQUEUE -- a preempted job is requeued and RESTARTED, so its wall
#  time is meaningless. Every card below is pinned to a NON-preemptible partition (gpu_quad and gpu
#  are both PreemptMode=OFF), verified with sinfo:
#    gpu_quad : a100, teslaV100s, a40, rtx8000, l40s
#    gpu      : l40s, teslaV100
#  a6000 is DELIBERATELY EXCLUDED from the sweep loop: it exists only in gpu_requeue on this
#  cluster, so it cannot be timed without preemption risk. It is kept in card_specs.tsv as a
#  reference row only. The fp64-strong (a100 9.7, teslaV100s 8.2) vs fp64-weak (l40s 1.4, a40 0.58,
#  rtx8000 0.51) contrast is fully covered without it.
# =============================================================================
# Resource header applies to the RUN-mode job (paired_benchmark.sh's own #SBATCH lines are inert when it is
# invoked with `bash`, so THIS header is what the node actually allocates). Matches paired_benchmark.sh.
#SBATCH --job-name=cardsweep
#SBATCH -c 20
# MEMORY IS RIGHT-SIZED FROM MEASUREMENT, not guessed. Across all 124 recorded runs the peak
# resident set was 1.13 GB mean / 6.99 GB max (logged per row as peak_ram_mb), so a 96G request
# wasted >90% of the reservation and depressed our scheduler priority. 16G keeps a >2x margin
# over the largest headline cell; the complexity-extension cells pass a larger --mem on the
# sbatch command line (see benchmark/scripts/harness/submit_complexity.sh), which overrides this.
#SBATCH --mem=16G
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/cardsweep_%j.out
#SBATCH -e logs/cardsweep_%j.err
set -uo pipefail

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
S=$REPO/benchmark/scripts/harness/paired_benchmark.sh              # the benchmark we reuse verbatim
SPECS=$REPO/benchmark/scripts/harness/card_specs.tsv                     # static intrinsic specs (group J), joined by card
SELF="${BASH_SOURCE[0]}"
GENUS=${GENUS:-n14_g__Nanosyncoccus}; THREADS=${THREADS:-20}

# spec_get CARD COLUMN  ->  value from card_specs.tsv for that card, or NA (group J static join).
# Skips '#' comment lines, reads the first non-comment line as the header, matches col 1 == card.
mkdir -p "$LOGDIR"
spec_get () {
  awk -v card="$1" -v col="$2" '
    /^[[:space:]]*#/ {next}
    /^[[:space:]]*$/ {next}
    !hdr { for(i=1;i<=NF;i++) idx[$i]=i; hdr=1; next }
    $1==card { c=idx[col]; if(c){print $c} else {print "NA"}; found=1; exit }
    END { if(!found) print "NA" }
  ' FS='\t' "$SPECS"
}

# ============================================================================================
#  RUN MODE - executes on the compute node with the requested --gres card.
# ============================================================================================
run_mode () {
  CARD=${CARD:?RUN mode needs CARD}
  echo "== CARD SWEEP (group J) card=$CARD genus=$GENUS -T $THREADS node=$(hostname -s) =="

  # ---- static intrinsic specs, joined from card_specs.tsv by the CARD key (group J) ----
  local sp_fp64 sp_fp32 sp_bw sp_memtype sp_bus sp_vram sp_sm sp_arch sp_tdp sp_pcie sp_fp64t sp_msrp
  sp_fp64=$(spec_get "$CARD" fp64_tflops)              # group J: gpu_fp64_tflops (protein prediction)
  sp_fp32=$(spec_get "$CARD" fp32_tflops)              # group J: gpu_fp32_tflops
  sp_bw=$(spec_get "$CARD" mem_bandwidth_gbs)          # group J: gpu_mem_bandwidth_gbs (DNA prediction)
  sp_memtype=$(spec_get "$CARD" mem_type)              # group J: gpu_mem_type (HBM vs GDDR)
  sp_bus=$(spec_get "$CARD" bus_width_bit)             # group J: bus_width
  sp_vram=$(spec_get "$CARD" vram_gb)                  # group J: vram_gb (spec)
  sp_sm=$(spec_get "$CARD" sm_count)                   # group J: gpu_sm_count (spec)
  sp_arch=$(spec_get "$CARD" arch)                     # group J: gpu_arch
  sp_tdp=$(spec_get "$CARD" tdp_w)                     # group J: board TDP (context for power cap)
  sp_pcie=$(spec_get "$CARD" pcie_gen)                 # group J: pcie gen (context for h2d/d2h)
  sp_fp64t=$(spec_get "$CARD" fp64_tensor)             # group J: DMMA (fp64 tensor) availability
  sp_msrp=$(spec_get "$CARD" msrp_usd)                 # group J: card MSRP -> perf_per_usd
  if [ "$sp_fp64" = NA ]; then
    echo "   *** WARNING: card '$CARD' not found in $SPECS - static group-J spec columns will be NA ***"
  fi

  # ---- one-shot device attributes (group J: things that are constant across the run) ----
  local drv cc gpuname vramtot powcap migmode eccmode hostcpu
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)          # group J: cuda_driver (per card; held constant or documented)
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)              # group J: gpu_arch / compute capability
  gpuname=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
  vramtot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) # group J: vram_gb (measured, MB)
  powcap=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1)   # group J: gpu_power_cap_w (was it power-capped?)
  migmode=$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | head -1)     # group J: mig_mode (a MIG slice != a full card)
  eccmode=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | head -1)     # group J: ecc_on (ECC cuts effective bandwidth)
  hostcpu=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//;s/ /_/g')               # group J: host_cpu_model (the HOST CONFOUND)
  drv=${drv:-NA}; cc=${cc:-NA}; vramtot=${vramtot:-NA}; powcap=${powcap:-NA}
  migmode=${migmode:-NA}; eccmode=${eccmode:-NA}; hostcpu=${hostcpu:-NA}

  # Decide the throttle-reason field once. Newer drivers rename clocks_throttle_reasons.* to
  # clocks_event_reasons.*; if neither is queryable we FALL BACK to sm-vs-max-sm clock comparison
  # (schema J: "query clocks_throttle_reasons.active if available, else compare sm_clock to
  # clocks.max.sm"). QTHROTTLE is empty when unavailable.
  local QTHROTTLE=""
  if nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader >/dev/null 2>&1; then
    QTHROTTLE="clocks_throttle_reasons.active"
  elif nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader >/dev/null 2>&1; then
    QTHROTTLE="clocks_event_reasons.active"
  fi

  # ---- background group-J sampler: timestamped clocks/temp/power, ~1 Hz, whole job ----
  # We sample the ENTIRE job and later WINDOW to the GPU arm using paired_benchmark.sh's own gpu.wall artifact
  # (its mtime = arm end, its content = arm wall seconds), so no change to paired_benchmark.sh is needed and the
  # CPU arm (GPU idle) does not pollute power_mean / energy.
  # DIRCARD: the CELL NAMESPACE for this card's output dir. For l40s it is "l40sq" (l40s on gpu_quad),
  # NEVER plain "l40s" -- the headline sweep owns the __l40s cells (measured on the gpu partition), and a
  # card-sweep re-run on a different node class must ADD a row, not overwrite the headline (learned the
  # hard way: a gpu_quad l40s re-measure produced ~2x slower DNA gpu_wall on a sibling node and briefly
  # clobbered 4 headline cells; restored from git). Specs join and gres still use CARD=l40s.
  local DIRCARD="$CARD"; [ "$CARD" = l40s ] && DIRCARD=l40sq
  local OUT=$REPO/benchmark/runs/paired/${GENUS}__T${THREADS}__${DIRCARD}
  mkdir -p "$OUT"
  # The sampler file must live OUTSIDE $OUT: paired_benchmark.sh (called below) starts with `rm -rf "$OUT"`, which would
  # delete a samples file inside $OUT mid-run (that is why early card cells had gpuJ_nsamples=0). Write it
  # as a sibling in paired/ (never wiped) and move it into $OUT for provenance after paired_benchmark.sh has recreated it.
  local SAMP="${OUT}.gpuJ.samples"                    # cols: epoch sm mem maxsm temp power [throttle]
  : > "$SAMP"
  local QCOLS="clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw"
  [ -n "$QTHROTTLE" ] && QCOLS="$QCOLS,$QTHROTTLE"
  (
    while true; do
      now=$(date +%s)
      line=$(nvidia-smi --query-gpu="$QCOLS" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' ' ')
      [ -n "$line" ] && echo "$now $line"
      sleep 1
    done
  ) >> "$SAMP" 2>/dev/null &
  local SAMPLER=$!

  # ---- run the benchmark itself: paired_benchmark.sh, inline on THIS node (its #SBATCH lines are inert) ----
  # Same problem/seed/binary/--gpu-strict/engagement gate as the headline; we do not touch that logic.
  echo "   running paired_benchmark.sh inline (same seed, --gpu-strict, engagement-gated) ..."
  # DATA-TYPE RESOLUTION (must match paired_benchmark.sh's contract): protein uses the default FASTA in benchmark/data/protein/ + full
  # ModelFinder; DNA is staged OUTSIDE benchmark/data/protein/ as PHYLIP (benchmark/data/dna/<set>/aln.phy) with a fixed model.
  # paired_benchmark.sh needs ALN_OVERRIDE + MODEL for DNA, so a card sweep on a dna_* set MUST pass them or paired_benchmark.sh falls
  # back to the (nonexistent) protein path and fails. Honor caller overrides, else auto-resolve by the
  # dna_ prefix -- the same alignment + fixed model as the headline DNA benchmark.
  local CS_ALN="${ALN_OVERRIDE:-}" CS_MODEL="${MODEL:-}"
  case "$GENUS" in
    dna_*) : "${CS_ALN:=$REPO/benchmark/data/dna/$GENUS/aln.phy}"; : "${CS_MODEL:=GTR+F+G4}" ;;
  esac
  # NOTE: must go through `env` -- a VAR=val produced by ${...:+...} expansion is NOT recognised as a
  # prefix assignment (bash parses prefixes before expansion) and becomes "command not found" (exit 127).
  env CARD="$DIRCARD" GENUS="$GENUS" THREADS="$THREADS" \
    ${CS_ALN:+ALN_OVERRIDE="$CS_ALN"} ${CS_MODEL:+MODEL="$CS_MODEL"} bash "$S"
  local frc=$?

  kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null

  # ---- window the samples to the GPU arm and reduce (group J runtime fields) ----
  # GPU-arm window = [ mtime(gpu.wall) - <gpu.wall seconds> , mtime(gpu.wall) ].
  local gwall_end gwall_dur w_start
  if [ -r "$OUT/gpu.wall" ]; then
    gwall_end=$(stat -c %Y "$OUT/gpu.wall" 2>/dev/null || echo 0)
    gwall_dur=$(awk 'NR==1{print $1+0}' "$OUT/gpu.wall" 2>/dev/null || echo 0)
    w_start=$(( gwall_end - gwall_dur ))
  else
    gwall_end=0; gwall_dur=0; w_start=0
  fi

  # gpu_wall_s (IQ-TREE's own reported wall) comes from paired_benchmark.sh's RESULT.tsv - the SAME number the
  # headline uses - so perf_per_watt / perf_per_usd are consistent with the published speedup.
  local RES="$OUT/RESULT.tsv" gpu_wall_s="NA"
  if [ -r "$RES" ]; then
    gpu_wall_s=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i} NR==2{c=h["gpu_wall_s"]; if(c)print $c}' "$RES")
    gpu_wall_s=${gpu_wall_s:-NA}
  fi

  # Reduce windowed samples: means of sm/mem clock, max temp, mean/max power, energy integral,
  # and a throttle flag. HASTHR=1 means a real throttle-reason field was sampled (col 7).
  local HASTHR=0; [ -n "$QTHROTTLE" ] && HASTHR=1
  local reduced
  reduced=$(awk -v s="$w_start" -v e="$gwall_end" -v hasthr="$HASTHR" '
    $1+0>=s && $1+0<=e {
      n++
      sm+=$2; mem+=$3; if($4+0>maxsm) maxsm=$4+0
      if($5+0>tmax) tmax=$5+0
      p=$6+0; psum+=p; if(p>pmax)pmax=p
      # energy: rectangle rule over consecutive in-window samples (power_w * dt_s)
      if(have){ dt=$1-plast; if(dt>0 && dt<30) energy+=plast_p*dt }
      plast=$1; plast_p=p; have=1
      # throttle: a nonzero hex reason bitmask on any windowed sample
      # throttle = any REAL slowdown reason. Bit 0x1 (GpuIdle: clocks drop because there is no work)
      # is masked out -- a launch-bound workload idles the GPU between kernels by nature, and flagging
      # that as "throttled" would falsely invalidate the measurement (seen on all rtx8000 DNA cells:
      # reason 0x1 only, 40C, 58W of a 260W cap -- idle, not slowed).
      if(hasthr){ v=strtonum($7); if(and(v, compl(1)) != 0) thr=1 }
    }
    END{
      if(n==0){ print "NA NA NA NA NA NA NA NA NA"; exit }
      printf "%.0f %.0f %.0f %.0f %.1f %.1f %.1f %d %d\n",
        sm/n, mem/n, maxsm, tmax, psum/n, pmax, energy, (hasthr?thr:-1), n
    }' "$SAMP")
  local sm_mean mem_mean maxsm temp_max pow_mean pow_max energy thr_raw nsamp
  read -r sm_mean mem_mean maxsm temp_max pow_mean pow_max energy thr_raw nsamp <<<"$reduced"
  # move the raw samples into the (now recreated) cell dir for provenance
  [ -r "$SAMP" ] && mv -f "$SAMP" "$OUT/cardsweep_gpuJ.samples" 2>/dev/null || true

  # Throttle flag (group J): prefer the sampled reason bitmask; else fall back to sm-vs-max-sm.
  local throttle="NA"
  if [ "${thr_raw:-NA}" = "-1" ]; then
    # fallback: sustained sm clock < 95% of clocks.max.sm => throttled
    if [ "${sm_mean:-NA}" != NA ] && [ "${maxsm:-NA}" != NA ] && [ "${maxsm:-0}" != "0" ]; then
      throttle=$(awk -v a="$sm_mean" -v b="$maxsm" 'BEGIN{print (a < 0.95*b)?1:0}')
    fi
  elif [ "${thr_raw:-NA}" != NA ]; then
    throttle="$thr_raw"
  fi

  # ---- derived: perf_per_watt = 1/(gpu_wall_s * power_mean); perf_per_usd = 1/(gpu_wall_s * msrp) ----
  local perf_watt="NA" perf_usd="NA"
  if [ "$gpu_wall_s" != NA ] && [ "${pow_mean:-NA}" != NA ] && [ "${pow_mean:-0}" != "0" ]; then
    perf_watt=$(awk -v w="$gpu_wall_s" -v p="$pow_mean" 'BEGIN{d=w*p; if(d>0)printf "%.6e",1/d; else print "NA"}')
  fi
  if [ "$gpu_wall_s" != NA ] && [ "${sp_msrp:-NA}" != NA ] && [ "${sp_msrp:-0}" != "0" ]; then
    perf_usd=$(awk -v w="$gpu_wall_s" -v m="$sp_msrp" 'BEGIN{d=w*m; if(d>0)printf "%.6e",1/d; else print "NA"}')
  fi

  # gpu_kernel_time_s: GPU-KERNEL-ONLY time, isolated from host time (schema J). This needs Nsight
  # Systems/Compute (nsys/ncu) to attribute device-kernel duration; the paired harness records only
  # wall time. Left as a TODO stub until an nsys profiling pass is wired in (see the retired launch-count
  # work). Emitting the column now keeps the card-sweep schema stable for later backfill.
  local gpu_kernel_time_s="TODO_nsight"

  # ---- emit: paired_benchmark.sh core columns (so rows aggregate into RUNS.tsv) + group-J columns ----
  local JHDR JROW OUTJ="$OUT/RESULT_cardsweep_J.tsv"
  JHDR=$(printf '%s\t' \
    gpu_fp64_tflops gpu_fp32_tflops gpu_mem_bandwidth_gbs gpu_mem_type bus_width vram_gb_spec \
    gpu_sm_count gpu_arch fp64_tensor tdp_w pcie_gen msrp_usd \
    gpu_vram_total_mb compute_capability cuda_driver mig_mode ecc_on gpu_power_cap_w host_cpu_model \
    gpu_sm_clock_mean gpu_mem_clock_mean gpu_temp_max_c gpu_throttle \
    gpu_power_mean_w gpu_power_max_w gpu_energy_j gpu_kernel_time_s \
    perf_per_watt perf_per_usd gpuJ_nsamples | sed 's/\t$//')
  JROW=$(printf '%s\t' \
    "$sp_fp64" "$sp_fp32" "$sp_bw" "$sp_memtype" "$sp_bus" "$sp_vram" \
    "$sp_sm" "$sp_arch" "$sp_fp64t" "$sp_tdp" "$sp_pcie" "$sp_msrp" \
    "${vramtot:-NA}" "$cc" "$drv" "$migmode" "$eccmode" "${powcap:-NA}" "$hostcpu" \
    "${sm_mean:-NA}" "${mem_mean:-NA}" "${temp_max:-NA}" "$throttle" \
    "${pow_mean:-NA}" "${pow_max:-NA}" "${energy:-NA}" "$gpu_kernel_time_s" \
    "$perf_watt" "$perf_usd" "${nsamp:-0}" | sed 's/\t$//')

  if [ -r "$RES" ]; then
    # join paired_benchmark.sh's core row (header line 1, data line 2) with the group-J columns, same row order
    {
      paste <(sed -n 1p "$RES") <(printf '%s\n' "$JHDR")
      paste <(sed -n 2p "$RES") <(printf '%s\n' "$JROW")
    } > "$OUTJ"
  else
    # paired_benchmark.sh produced no RESULT.tsv (benchmark failed) - still record the group-J block for audit
    { printf '%s\n' "$JHDR"; printf '%s\n' "$JROW"; } > "$OUTJ"
    echo "   *** WARNING: $RES absent (paired_benchmark.sh rc=$frc); wrote group-J block only ***"
  fi
  echo; echo "== group-J card-sweep row -> $OUTJ =="
  column -t -s $'\t' "$OUTJ" 2>/dev/null | cut -c1-200 || cat "$OUTJ"
  return "$frc"
}

# ============================================================================================
#  SUBMIT MODE (default) - sbatch THIS script once per card. NOTE: this file only DEFINES the
#  submissions; nothing is launched unless the user runs it. Each job re-enters in RUN mode.
# ============================================================================================
submit_mode () {
  # CS_TIME override: DNA ladder cells finish in <15 min, so a short limit (e.g. CS_TIME=01:00:00)
  # makes them backfill-eligible and they land days sooner on a congested partition. Protein keeps 24h.
  local TLIMIT=${CS_TIME:-1-00:00:00}
  # CS_CARDS / CS_PART overrides: teslaV100 (non-S) lives only in the `gpu` partition (also
  # PreemptMode=OFF, so timing-valid) -- submit it with CS_CARDS=teslaV100 CS_PART=gpu.
  for CARD in ${CS_CARDS:-a100 teslaV100s a40 rtx8000 l40s}; do
    local PART=${CS_PART:-${SLURM_PARTITION:-gpu_quad}}   # non-preemptible; see header
    sbatch --job-name="cs_${CARD}" --chdir="$REPO" -o "$LOGDIR/cs_${CARD}_%j.out" -e "$LOGDIR/cs_${CARD}_%j.err" --partition="$PART" --gres=gpu:${CARD}:1 --time="$TLIMIT" \
           --export=ALL,REPO_DIR="$REPO",CS_RUN=1,GENUS="$GENUS",THREADS="$THREADS",CARD="$CARD" "$SELF" \
      && echo "  submitted card sweep $CARD ($GENUS -T $THREADS, t=$TLIMIT) -> $PART"
  done
}

# Entry: RUN mode iff CS_RUN=1 (set by submit_mode's --export) or first arg is 'run'; else submit.
if [ "${CS_RUN:-0}" = 1 ] || [ "${1:-}" = run ]; then
  run_mode
else
  submit_mode
fi
