#!/bin/bash
# =============================================================================
#  IQ-TREE3-GPU - PAIRED benchmark: CPU and GPU arms on the SAME NODE, same job.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY PAIRED. The previous matrix ran every CPU cell on compute-a/b-16-* (Xeon
#  Platinum 8468, one on a 2016 E5-2683 v4) and every GPU cell on compute-gc-17-*
#  (mostly Xeon Gold 6542Y). Zero node overlap, so the reported ~1.46x "GPU speedup"
#  was consistent with the CPU hardware difference alone. Running both arms inside
#  ONE job makes the CPU, memory, and node identical by construction.
#
#  WHY A100 IS THE DEFAULT CARD. The kernel is double precision and, now that it is
#  coalesced, compute-bound in fp64. The L40S runs fp64 at 1/64 of its fp32 rate
#  (~1.43 TFLOPS) whereas the A100 does ~9.7 TFLOPS. Benchmarking the tool on an L40S
#  measures a crippled fp64 path, not the method. Card variability is a SEPARATE
#  experiment (card_sweep.sh), not the headline.
#
#  ENGAGEMENT IS ENFORCED. A GPU cell is only valid if the run printed
#  "GPU acceleration ACTIVE" AND nvidia-smi showed non-zero utilisation. Without both,
#  the cell is written as INVALID. This exists because for four days every "GPU" run
#  silently executed the CPU kernel behind a green self-test.
#
#  WHAT "-T N" MEANS ON EACH ARM -- these are NOT symmetric, and the labelling must not imply they
#  are. On the CPU arm, -T 20 genuinely runs 20 worker threads over disjoint pattern ranges. On the
#  GPU arm, packet 0 computes ALL patterns in one launch and every other packet returns immediately,
#  so -T 20 changes only HOST-side parallelism (model setup, traversal bookkeeping, CPU-fallback
#  models) -- it does not mean "20 threads feeding the GPU".
#
#  The comparison is still sound and is the one users care about: BEST CPU CONFIGURATION versus BEST
#  GPU CONFIGURATION on identical hardware. But it must be reported as
#      "CPU, 20 threads"   vs   "GPU + 20 host threads"
#  and never as "20-thread CPU vs 20-thread GPU", which would imply a symmetry that does not exist.
#  Reported per-cell as cpu_config / gpu_config so the distinction survives into the results table.
#
#  Usage (via submit.sh):  GENUS=g__CAG-269 THREADS=20 CARD=a100 sbatch this
# =============================================================================
#SBATCH --job-name=pair
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
#SBATCH -o logs/pair_%j.out
#SBATCH -e logs/pair_%j.err
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
# GPUBIN defaults to the PRODUCTION binary -- the headline benchmark must always use it. GPUBIN_OVERRIDE
# exists solely for a clearly-labelled PRELIMINARY directional sweep on an isolated build (e.g. the
# on-device-Newton binary) before it is merged into production; such a run is not a headline number.
GPUBIN=${GPUBIN_OVERRIDE:-$IQTREE_BUILD_ROOT/iqtree3_gpu_build/iqtree3}
STOCK=$IQTREE_BUILD_ROOT/iqtree3_build/iqtree3
CHECK=$REPO/scripts/rf_logl_check.py
GENUS=${GENUS:?set GENUS}; THREADS=${THREADS:-20}; CARD=${CARD:-a100}
MODEL=${MODEL:-TESTNEW}
# ALN defaults to the staged amino-acid alignment, but an explicit ALN_OVERRIDE wins -- this is how
# the DNA sweep points at benchmark/data/dna/<set>/aln.phy, which lives outside benchmark/data/protein/ and is PHYLIP not
# FASTA. The LFS-pointer and record-count guards below handle both formats.
ALN=${ALN_OVERRIDE:-$REPO/benchmark/data/protein/$GENUS/concat_alignment.faa}
OUT=$REPO/benchmark/runs/paired/${GENUS}__T${THREADS}__${CARD}

# --- PRE-FLIGHT: never destroy a SUCCESSFUL measurement -------------------------------------------
# The `rm -rf "$OUT"` below gives a re-run a clean slate, which is right for a FAILED cell. But the
# cell directory is addressed ONLY by ${GENUS}__T${THREADS}__${CARD}, so a submitter invoked with a
# triple that already names a successful run would wipe irreplaceable raw output. The aggregator's
# guard (aggregate_runs.py: "ABORT: N SUCCESSFUL cell(s) would vanish") protects ROWS in
# RUNS.tsv -- it never sees the directory, and by the time it runs the trees are already gone.
# Policy: overwriting a failed run is fine; overwriting a successful one is corruption.
if [ -s "$OUT/RESULT.tsv" ] && [ "${FORCE:-0}" != "1" ]; then
  _gp=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="gate_pass") c=i; next}
                     NR==2{print (c ? $c : "")}' "$OUT/RESULT.tsv" 2>/dev/null)
  if [ "$_gp" = "TRUE" ]; then
    echo "REFUSING to overwrite a SUCCESSFUL measurement." >&2
    echo "  cell : ${GENUS}__T${THREADS}__${CARD}" >&2
    echo "  path : $OUT  (RESULT.tsv has gate_pass=TRUE)" >&2
    echo "  Its raw output cannot be reproduced. Use a different CARD namespace for a new" >&2
    echo "  measurement, or set FORCE=1 to overwrite deliberately." >&2
    exit 3
  fi
fi
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT" || exit 1

# Fail fast and legibly on an un-smudged Git LFS pointer. benchmark/data/protein/**/*.faa is LFS-tracked, so a
# clone made without git-lfs leaves a ~130-byte text stub here instead of the alignment. It looks
# like an ordinary file: `grep -c '^>'` returns 0 and IQ-TREE then dies with an opaque parse error
# many minutes into a job. Diagnose it up front instead.
if [ ! -f "$ALN" ]; then
  echo "ERROR: alignment not found: $ALN"; exit 1
fi
if head -1 "$ALN" | grep -q "^version https://git-lfs"; then
  echo "ERROR: $ALN is an unresolved Git LFS pointer, not sequence data."
  echo "       This clone was made without git-lfs. Fix with:"
  echo "           git lfs install && git lfs pull"
  exit 1
fi
case "$ALN" in
  *.phy|*.phylip) NREC=$(head -1 "$ALN" | awk '{print $1+0}') ;;   # PHYLIP: first field of header
  *)              NREC=$(grep -c '^>' "$ALN") ;;                     # FASTA
esac
if [ "${NREC:-0}" -eq 0 ]; then
  echo "ERROR: $ALN contains no sequence records."; exit 1
fi

CARDNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
CPUMODEL=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//;s/ /_/g')
NODE=$(hostname -s); NTAX=${NREC}

# --- Group A: identity/provenance (queried once; both arms share this node/job) ---
RUN_ID=${SLURM_JOB_ID:-NA}                 # A: SLURM job id, the primary key
DATETIME_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # A: run start, ISO-8601 UTC
# data_source: default real MAG; a staged alisim alignment overrides via path/name or DATA_SOURCE env.
DATA_SOURCE=${DATA_SOURCE:-}
if [ -z "$DATA_SOURCE" ]; then
  case "${GENUS}__${ALN}" in
    *alisim*|*simulated*|*alisimulated*) DATA_SOURCE=alisim_simulated ;;
    *)                                   DATA_SOURCE=real_MAG ;;
  esac
fi

# --- Group C: hardware identity from nvidia-smi (total VRAM, driver, compute capability) ---
GPU_SPECS=$(nvidia-smi --query-gpu=memory.total,driver_version,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$GPU_SPECS" ]; then
  GPU_VRAM_TOTAL_GB=$(echo "$GPU_SPECS" | awk -F',' '{printf "%.1f", $1/1024}')
  CUDA_DRIVER_VERSION=$(echo "$GPU_SPECS" | awk -F',' '{gsub(/ /,"",$2);print $2}')
  COMPUTE_CAPABILITY=$(echo "$GPU_SPECS" | awk -F',' '{gsub(/ /,"",$3);print $3}')
else
  GPU_VRAM_TOTAL_GB=NA; CUDA_DRIVER_VERSION=NA; COMPUTE_CAPABILITY=NA
fi
CPU_CORES_ALLOC=${SLURM_CPUS_PER_TASK:-NA}   # C: cores allocated (-c); context for -T
PARTITION=${SLURM_JOB_PARTITION:-NA}         # C: partition (timing rows must be non-preemptible)

# --- Group B: software/build identity ---
TOOL_GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo NA)
[ -z "$TOOL_GIT_COMMIT" ] && TOOL_GIT_COMMIT=NA
BINARY_SHA256=$(sha256sum "$GPUBIN" 2>/dev/null | awk '{print $1}')  # hash of the binary actually run
[ -z "$BINARY_SHA256" ] && BINARY_SHA256=NA
# Sites are NOT the unit of work: IQ-TREE evaluates the likelihood per distinct site PATTERN. For
# amino acid data the two are close, but for DNA patterns are bounded by 4^taxa and collapse hard at
# small n (one of our DNA sets has 195456 sites but only 183 patterns). Recording sites alone would
# make a trivial workload look large. Filled in after the run from IQ-TREE's own report.
NPAT=NA
echo "== PAIRED $GENUS  -T $THREADS  node=$NODE  card=$CARDNAME  cpu=$CPUMODEL  taxa=$NTAX =="
echo "   both arms run HERE, so the CPU is identical between them by construction."
echo "   configs: CPU = ${THREADS} worker threads | GPU = 1 device + ${THREADS} host threads"
echo "            (-T on the GPU arm drives host-side work only; packet 0 does all patterns)"
CPU_CONFIG="CPU_${THREADS}thr"
GPU_CONFIG="GPU+${THREADS}hostthr"

# --- Group A/E: reproducibility + support knobs surfaced as first-class fields ---
SEED=${SEED:-1}                       # A: RNG seed (-seed), so the search is reproducible
UFBOOT=${UFBOOT:-1000}; ALRT=${ALRT:-1000}
# Our benchmark uses FULL ModelFinder for protein (-m TESTNEW) and a fixed model for DNA (-m GTR+F+G4),
# so MF_FLAGS stays EMPTY. It is a generic passthrough only; if ever set, its flags apply identically to
# both arms so the CPU-vs-GPU speedup remains a fair same-model comparison.
IQARGS="-m $MODEL ${MF_FLAGS:+$MF_FLAGS }--ufboot $UFBOOT --alrt $ALRT -seed $SEED"
SUPPORT_SETTINGS="ufboot=${UFBOOT},alrt=${ALRT}"   # E: support_settings
GPU_FLAGS="-gpu --gpu-strict"                       # E: gpu_flags (applied to the GPU arm)

run_arm () {  # arm(cpu|gpu)
  # NOTE: these MUST be separate statements. In `local a=$1 b="$OUT/$a"` bash expands every word
  # BEFORE performing any of the assignments, so $a is still unset when b is built -- which under
  # `set -u` aborts the job with "arm: unbound variable". That is exactly how this script died on
  # its first ever real execution.
  local arm=$1
  # BOTH ARMS USE THE SAME BINARY, differing ONLY by the -gpu flag.
  #
  # Previously the CPU arm ran $STOCK (a separate binary built 11 days earlier) while the GPU arm ran
  # $GPUBIN. Any difference between those two binaries -- inlining, build date, the guarded
  # #ifdef IQTREE_GPU hooks in the CPU path -- would have been silently attributed to "the GPU".
  # Without -gpu the swap block never runs, so $GPUBIN executes the stock CPU kernel: identical code
  # path, one variable. This is the same control used in the earlier phase-decomposition experiment; it should
  # have been here from the start.
  local bin=$GPUBIN
  local gflag=""
  local pfx="$OUT/$arm"
  # --gpu-strict is COMPUTATIONALLY IDENTICAL to -gpu; it only turns any CPU fallback or device
  # error into a hard abort. For a benchmark that is essential: a silent fallback would mean
  # unknowingly timing CPU against CPU, which is exactly the failure this project already shipped
  # once. If the GPU arm aborts, we get no number rather than a wrong one.
  [ "$arm" = gpu ] && { bin=$GPUBIN; gflag="-gpu --gpu-strict"; }
  echo; echo "--- $arm arm (-T $THREADS $gflag) ---"
  local sampler=""
  if [ "$arm" = gpu ]; then
    # Group F/L: sample SM util (utilization.gpu), memory-controller util (utilization.memory) AND
    # VRAM in use (memory.used) together, so the DNA/AA compute-vs-memory regime and peak VRAM are
    # recoverable. CSV columns: sm_util, mem_util, mem_used_MiB.
    ( while true; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$pfx.utilfull" 2>/dev/null &
    sampler=$!
  fi
  local t0=$SECONDS
  /usr/bin/time -f "%e %U %S %M" -o "$pfx.time" \
      "$bin" -s "$ALN" -T "$THREADS" $gflag $IQARGS --prefix "$pfx" -redo > "$pfx.stdout" 2>&1
  local rc=$? wall=$((SECONDS-t0))
  [ -n "$sampler" ] && { kill $sampler 2>/dev/null; wait $sampler 2>/dev/null; }
  # Backward compatibility: keep the single-column SM-util file the engagement gate already reads.
  if [ "$arm" = gpu ] && [ -f "$pfx.utilfull" ]; then
    awk -F',' 'NF{gsub(/ /,"",$1); print $1}' "$pfx.utilfull" > "$pfx.util"
  fi
  echo "$wall" > "$pfx.wall"; echo "$rc" > "$pfx.rc"
  echo "   rc=$rc wall=${wall}s"
  [ "$rc" -eq 0 ] || echo "   *** FAILED: $(grep -m2 -iE 'error|abort' "$pfx.stdout" | head -2)"
}

# Warm the page cache BEFORE either arm. The arms run sequentially, so without this the second arm
# (always the GPU one) reads the alignment from cache while the first paid the disk I/O -- a bias in
# the GPU's favour, in the one direction we most need to avoid overstating.
cat "$ALN" > /dev/null 2>&1

run_arm cpu
run_arm gpu

# ---- engagement gate: did the GPU actually do the work? ----
# NOTE: do NOT write `grep -c ... || echo 0`. `grep -c` PRINTS "0" and EXITS 1 when there is no
# match, so the `||` fires and the substitution captures "0\n0", which makes the integer test below
# die with "integer expression expected". Use `grep -c || true` and let grep's own "0" stand.
BANNER=$(grep -c "GPU acceleration ACTIVE" "$OUT/gpu.stdout" 2>/dev/null || true)
BANNER=${BANNER:-0}
UTILNZ=$(awk 'NF&&$1+0==$1&&$1>0{c++}END{print c+0}' "$OUT/gpu.util" 2>/dev/null || echo 0)
UTILMEAN=$(awk 'NF&&$1+0==$1{n++;s+=$1}END{if(n)printf "%.1f",s/n; else print "NA"}' "$OUT/gpu.util" 2>/dev/null)
ENGAGED=INVALID
[ "$BANNER" -ge 1 ] && [ "$UTILNZ" -ge 1 ] && ENGAGED=OK

# NUMERIC PROOF OF GPU USE. The ACTIVE banner prints once at kernel selection; it does NOT prove every
# likelihood call ran on the device, because each entry point re-tests eligibility per call and can
# delegate silently. GPUSTATS reports the actual split, so we assert on a number instead of a banner.
# NOTE: these lines go to stdout, not the .log -- hence gpu.stdout.
GPUSTATS=$(grep -oP "GPUSTATS \K.*" "$OUT/gpu.stdout" 2>/dev/null | tail -1)
PCT_GPU=$(echo "$GPUSTATS" | grep -oP "pct_gpu=\K[0-9.]+")
CPU_CALLS=$(echo "$GPUSTATS" | grep -oP "cpu_calls=\K[0-9]+")
REBUILDS=$(echo "$GPUSTATS" | grep -oP "rebuilds=\K[0-9]+")
REBINDS=$(echo "$GPUSTATS"  | grep -oP "rebinds=\K[0-9]+")
# Group F: additional engagement proof from the same GPUSTATS line.
GPU_CALLS=$(echo "$GPUSTATS"       | grep -oP "gpu_calls=\K[0-9]+")
DEVICE_ERRORS=$(echo "$GPUSTATS"   | grep -oP "device_errors=\K[0-9]+")
# kernel_launches / newton_ondevice are emitted by the on-device-Newton engine; absent on the current
# production binary (leave NA rather than fabricate a zero).
KERNEL_LAUNCHES=$(echo "$GPUSTATS" | grep -oP "kernel_launches=\K[0-9]+")
NEWTON_ONDEVICE=$(echo "$GPUSTATS" | grep -oP "newton_ondevice=\K[0-9]+")
# Group B: replay_cache engine state (documents the config); default off when not reported.
REPLAY_CACHE=$(echo "$GPUSTATS"    | grep -oP "replay_cache=\K\w+")
[ -z "$REPLAY_CACHE" ] && REPLAY_CACHE=off
echo "   GPUSTATS: ${GPUSTATS:-<absent>}"
# NUMERIC-PROOF engagement upgrade. The nvidia-smi util sampler can miss a trivially small/fast tree
# (e.g. n=4: 183 patterns finish inside one sample window), giving util=0 and a false INVALID. The
# GPUSTATS counts are the stronger, direct proof: banner present, the device handled ~100% of eligible
# calls (pct_gpu>=99), real device calls occurred (gpu_calls>0), no CPU fallback (cpu_calls=0), no
# device errors. When all hold, the run IS a genuine GPU measurement regardless of the coarse sampler.
if [ "$ENGAGED" != OK ] && [ "${BANNER:-0}" -ge 1 ] \
   && [ -n "$PCT_GPU" ] && [ "${PCT_GPU%%.*}" -ge 99 ] \
   && [ "${CPU_CALLS:-1}" -eq 0 ] && [ "${GPU_CALLS:-0}" -gt 0 ] \
   && [ "${DEVICE_ERRORS:-1}" -eq 0 ]; then
  ENGAGED=OK
  echo "   (engaged by numeric GPUSTATS proof: pct_gpu=$PCT_GPU cpu_calls=$CPU_CALLS gpu_calls=$GPU_CALLS device_errors=$DEVICE_ERRORS; util sampler too coarse for this tiny run)"
fi
if [ -n "$CPU_CALLS" ] && [ "$CPU_CALLS" -ne 0 ]; then
  ENGAGED=PARTIAL_CPU
  echo "   *** $CPU_CALLS likelihood calls ran on the CPU - this cell is NOT a pure GPU measurement ***"
  grep -A3 "delegated by reason" "$OUT/gpu.stdout" 2>/dev/null | sed 's/^/     /'
fi
echo; echo "== ENGAGEMENT: $ENGAGED  (banner=$BANNER nonzero_util_samples=$UTILNZ mean_util=${UTILMEAN}%) =="
[ "$ENGAGED" = OK ] || echo "   *** GPU DID NOT RUN THE LIKELIHOOD - this cell is not a GPU measurement ***"

# ---- equivalence gate ----
echo; echo "== EQUIVALENCE (GPU vs CPU) =="
python3 "$CHECK" "$OUT/cpu" "$OUT/gpu" "${GENUS}_T${THREADS}" 2>&1 | sed 's/^/   /'

getwall () { grep -oP "Total wall-clock time used: \K[0-9.]+" "$1.log" 2>/dev/null | tail -1; }
# The final log-likelihood lives in the .iqtree report as "Log-likelihood of the tree:".
# Do NOT rely on "BEST SCORE FOUND :" -- that is emitted to the .log DURING the search and is
# absent from the .iqtree of a -m TESTNEW run, so it silently yields an empty field.
getll   () {
  local v
  v=$(grep -oP "Log-likelihood of the tree: \K-?[0-9.]+" "$1.iqtree" 2>/dev/null | tail -1)
  [ -z "$v" ] && v=$(grep -oP "BEST SCORE FOUND : \K-?[0-9.]+" "$1.log" 2>/dev/null | tail -1)
  echo "${v:-NA}"
}
# Per-phase wall times, so we can see WHERE the GPU helps (ModelFinder is 79-97% of a real run).
getphase () { grep -oP "$2\K[0-9.]+" "$1.log" 2>/dev/null | tail -1; }
NPAT=$(grep -oP "Number of distinct site patterns: \K[0-9]+" "$OUT/cpu.iqtree" 2>/dev/null | tail -1)
NSITE=$(grep -oP "Input data: [0-9]+ sequences with \K[0-9]+" "$OUT/cpu.iqtree" 2>/dev/null | tail -1)
CW=$(getwall "$OUT/cpu"); GW=$(getwall "$OUT/gpu")
SP=$(python3 -c "print(f'{$CW/$GW:.2f}')" 2>/dev/null || echo NA)
CS=$(getphase "$OUT/cpu" "Wall-clock time used for tree search: "); GS=$(getphase "$OUT/gpu" "Wall-clock time used for tree search: ")
CRC=$(cat "$OUT/cpu.rc" 2>/dev/null); GRC=$(cat "$OUT/gpu.rc" 2>/dev/null)

# =============================================================================
#  EXTENDED PER-RUN RECORD (benchmark/BENCHMARK_SCHEMA.md groups A-M). Every field below is
#  parsed defensively and falls back to NA so the row is always emitted (a cell
#  missing engagement/equivalence proof is still recorded, just flagged).
# =============================================================================

# --- Group B: software ---
IQTREE_VERSION=$(grep -oP "IQ-TREE version \K[0-9.]+" "$OUT/cpu.log" 2>/dev/null | head -1)
[ -z "$IQTREE_VERSION" ] && IQTREE_VERSION=NA

# --- Group A: data_type (drives num_states) + exact command line per arm ---
DATA_TYPE=$(grep -m1 -oP "sequences with [0-9]+ \K(amino-acid|nucleotide)" "$OUT/cpu.iqtree" 2>/dev/null)
case "$DATA_TYPE" in
  amino-acid) DATA_TYPE=protein; NUM_STATES=20 ;;
  nucleotide) DATA_TYPE=dna;     NUM_STATES=4  ;;
  *) case "$ALN" in
       *.faa|*.fasta|*.faa.gz) DATA_TYPE=protein; NUM_STATES=20 ;;
       *)                      DATA_TYPE=dna;     NUM_STATES=4  ;;
     esac ;;
esac
CMD_CPU=$(grep -oP "^Command: \K.*" "$OUT/cpu.log" 2>/dev/null | head -1); [ -z "$CMD_CPU" ] && CMD_CPU=NA
CMD_GPU=$(grep -oP "^Command: \K.*" "$OUT/gpu.log" 2>/dev/null | head -1); [ -z "$CMD_GPU" ] && CMD_GPU=NA

# --- Group D: alignment characterisation (from IQ-TREE's own report) ---
INFORMATIVE_SITES=$(grep -oP "Number of parsimony informative sites: \K[0-9]+" "$OUT/cpu.iqtree" 2>/dev/null | tail -1)
CONSTANT_SITES=$(grep -oP "Number of constant sites: \K[0-9]+" "$OUT/cpu.iqtree" 2>/dev/null | tail -1)
# gap_ambiguity_pct is the "****  TOTAL  NN.NN%" line of the composition table in the .log.
GAP_AMBIGUITY_PCT=$(grep -oP "TOTAL\s+\K[0-9.]+" "$OUT/cpu.log" 2>/dev/null | tail -1)
MODEL_USED=$(grep -oP "Model of substitution: \K.*" "$OUT/cpu.iqtree" 2>/dev/null | tail -1)
RATE_CATEGORIES=$(grep -oP "with \K[0-9]+(?= categories)" "$OUT/cpu.iqtree" 2>/dev/null | tail -1)
[ -z "$INFORMATIVE_SITES" ] && INFORMATIVE_SITES=NA
[ -z "$CONSTANT_SITES" ] && CONSTANT_SITES=NA
[ -z "$GAP_AMBIGUITY_PCT" ] && GAP_AMBIGUITY_PCT=NA
[ -z "$MODEL_USED" ] && MODEL_USED=NA
[ -z "$RATE_CATEGORIES" ] && RATE_CATEGORIES=NA

# --- Group F: GPU-engagement extras ---
GPU_BANNER=FALSE; [ "${BANNER:-0}" -ge 1 ] && GPU_BANNER=TRUE
# Utilisation distribution (SM + memory-controller) and peak VRAM, sampled during the GPU arm.
UTIL_SM_MEAN=NA; UTIL_SM_SD=NA; UTIL_SM_MAX=NA
UTIL_MEM_MEAN=NA; UTIL_MEM_SD=NA; UTIL_MEM_MAX=NA
UTIL_NSAMPLES=0; PEAK_VRAM_MB=NA
if [ -f "$OUT/gpu.utilfull" ]; then
  read -r UTIL_SM_MEAN UTIL_SM_SD UTIL_SM_MAX UTIL_MEM_MEAN UTIL_MEM_SD UTIL_MEM_MAX UTIL_NSAMPLES PEAK_VRAM_MB < <(
    awk -F',' '
      { s=$1; m=$2; v=$3; gsub(/ /,"",s); gsub(/ /,"",m); gsub(/ /,"",v)
        # numeric max comparisons (s>smx as strings is lexical: "1639">"3" is false -> corrupts maxima)
        if (NF>=3 && s+0==s) { n++; ss+=s; ss2+=s*s; if(s+0>smx)smx=s+0;
          mm+=m; mm2+=m*m; if(m+0>mmx)mmx=m+0;
          if (v+0==v && v+0>pv) pv=v+0 } }
      END{ if(n){ smean=ss/n; ssd=(n>1)?sqrt((ss2-ss*ss/n)/(n-1)):0;
                  mmean=mm/n; msd=(n>1)?sqrt((mm2-mm*mm/n)/(n-1)):0;
                  printf "%.1f %.1f %.0f %.1f %.1f %.0f %d %.0f", smean,ssd,smx,mmean,msd,mmx,n,pv }
           else printf "NA NA NA NA NA NA 0 NA" }' "$OUT/gpu.utilfull")
fi
[ -z "$UTIL_SM_MEAN" ] && { UTIL_SM_MEAN=NA; UTIL_SM_SD=NA; UTIL_SM_MAX=NA; UTIL_MEM_MEAN=NA; UTIL_MEM_SD=NA; UTIL_MEM_MAX=NA; UTIL_NSAMPLES=0; PEAK_VRAM_MB=NA; }

# --- Group G: per-phase + resource timing ---
getmf () { grep -oP "Wall-clock time for ModelFinder: \K[0-9.]+" "$1.log" 2>/dev/null | tail -1; }
# IQ-TREE has no dedicated wall line for the UFBoot/aLRT phase (it is folded into tree search); parse
# one if a build ever prints it, else NA.
getbs () { grep -oP "Wall-clock time for .*[Bb]ootstrap[^:]*: \K[0-9.]+" "$1.log" 2>/dev/null | tail -1; }
CPU_MODELFINDER_S=$(getmf "$OUT/cpu"); GPU_MODELFINDER_S=$(getmf "$OUT/gpu")
CPU_BOOTSTRAP_S=$(getbs "$OUT/cpu");   GPU_BOOTSTRAP_S=$(getbs "$OUT/gpu")
CPU_TOTALCPU_S=$(grep -oP "Total CPU time used: \K[0-9.]+" "$OUT/cpu.log" 2>/dev/null | tail -1)
[ -z "$CPU_MODELFINDER_S" ] && CPU_MODELFINDER_S=NA
[ -z "$GPU_MODELFINDER_S" ] && GPU_MODELFINDER_S=NA
[ -z "$CPU_BOOTSTRAP_S" ] && CPU_BOOTSTRAP_S=NA
[ -z "$GPU_BOOTSTRAP_S" ] && GPU_BOOTSTRAP_S=NA
[ -z "$CPU_TOTALCPU_S" ] && CPU_TOTALCPU_S=NA
# peak_ram_mb: prefer sacct MaxRSS; fall back to /usr/bin/time %M (KB) captured per arm.
PEAK_RAM_MB=NA; RESTARTS=NA
if [ "${SLURM_JOB_ID:-}" != "" ] && command -v sacct >/dev/null 2>&1; then
  # MaxRSS comes back like "931340K"/"1.24G"; normalise to MB via numfmt-free awk.
  _rss=$(sacct -j "$SLURM_JOB_ID" -n -o MaxRSS 2>/dev/null | awk 'NF{print}' | sort -h | tail -1)
  PEAK_RAM_MB=$(awk -v r="$_rss" 'BEGIN{ u=substr(r,length(r),1); n=r+0;
      if(u=="K")n/=1024; else if(u=="G")n*=1024; else if(u=="M"){} else n/=1048576;
      if(n>0) printf "%.0f", n; else print "NA" }')
  # restarts: use the Restarts accounting field when present, else assume 0 within a live job.
  RESTARTS=$(sacct -j "$SLURM_JOB_ID" -X -n -o Restarts 2>/dev/null | awk 'NF{gsub(/ /,"");print;exit}')
  [ -z "$RESTARTS" ] && RESTARTS=0
fi
if [ "$PEAK_RAM_MB" = NA ]; then
  _cm=$(awk '{print $4}' "$OUT/cpu.time" 2>/dev/null); _gm=$(awk '{print $4}' "$OUT/gpu.time" 2>/dev/null)
  PEAK_RAM_MB=$(awk -v a="${_cm:-0}" -v b="${_gm:-0}" 'BEGIN{m=(a>b)?a:b; if(m>0)printf "%.0f", m/1024; else print "NA"}')
fi

# --- Group H: equivalence (logL diffs, RF, support, single gate bit) ---
SUPPORT_GATE_TOL=${SUPPORT_GATE_TOL:-5}   # support "agrees" if max abs diff over shared splits <= this
EQ_KV="$OUT/equivalence.kv"
python3 - "$OUT/cpu" "$OUT/gpu" "$NTAX" "$SUPPORT_GATE_TOL" > "$EQ_KV" 2>/dev/null <<'PYEQ'
import sys, os, re
pa, pb, ntax, stol = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
try: ntax = int(ntax)
except Exception: ntax = 0

def parse_logl(f):
    if not os.path.exists(f): return None
    t = open(f, errors="ignore").read()
    m = re.search(r"Log-likelihood of the tree:\s*(-?[0-9.]+)", t) or \
        re.search(r"BEST SCORE FOUND :\s*(-?[0-9.]+)", t)
    return float(m.group(1)) if m else None

def parse_tree(f):
    """(taxa set, {canonical_split: support_or_None}); never raises."""
    if not os.path.exists(f): return None, None
    s = open(f, errors="ignore").read().strip()
    if not s: return None, None
    if s.endswith(";"): s = s[:-1]
    i = 0; leaves = []; splits = []
    def parse():
        nonlocal i
        clade = set(); sup = None
        if s[i] == "(":
            i += 1
            while True:
                child, _ = parse(); clade |= child
                if s[i] == ",": i += 1; continue
                if s[i] == ")": i += 1; break
            j = i
            while i < len(s) and s[i] not in ",():": i += 1
            lab = s[j:i]
            if lab:
                try: sup = float(lab)
                except Exception: sup = None
            if i < len(s) and s[i] == ":":
                i += 1
                while i < len(s) and s[i] not in ",()": i += 1
            splits.append((frozenset(clade), sup)); return clade, sup
        j = i
        while i < len(s) and s[i] not in ",()": i += 1
        leaves.append(s[j:i].split(":")[0]); return {leaves[-1]}, None
    try:
        parse()
    except (IndexError, RecursionError):
        return None, None
    taxa = set(leaves); n = len(taxa)
    if not taxa: return None, None
    first = sorted(taxa)[0]; d = {}
    for cl, sup in splits:
        if len(cl) <= 1 or len(cl) >= n - 1: continue
        side = cl if first not in cl else frozenset(taxa - cl); d[side] = sup
    return taxa, d

la, lb = parse_logl(pa + ".iqtree"), parse_logl(pb + ".iqtree")
logl_absdiff = logl_reldiff = None
if la is not None and lb is not None:
    logl_absdiff = abs(la - lb)
    logl_reldiff = abs(la - lb) / abs(la) if la != 0 else None

ta, da = parse_tree(pa + ".treefile"); tb, db = parse_tree(pb + ".treefile")
rf = rf_norm = None
if ta and tb and ta == tb:
    rf = len(set(da) ^ set(db))
    denom = 2 * (len(ta) - 3)
    rf_norm = rf / denom if denom > 0 else None

_, sa = parse_tree(pa + ".contree"); _, sb = parse_tree(pb + ".contree")
support_max = None
if sa and sb:
    shared = set(sa) & set(sb)
    diffs = [abs(sa[k] - sb[k]) for k in shared if sa[k] is not None and sb[k] is not None]
    support_max = max(diffs) if diffs else None

gate = "NA"
if logl_reldiff is not None and rf is not None:
    ok = (logl_reldiff <= 1e-4 and rf == 0)
    if support_max is not None:
        ok = ok and (support_max <= stol)
    gate = "TRUE" if ok else "FALSE"

def fmt(x):  # clean, consistent scientific notation for the equivalence columns (no 16-digit repr)
    if x is None: return "NA"
    return "0.00e+00" if x == 0 else "%.3e" % x
print("logl_absdiff=" + fmt(logl_absdiff))
print("logl_reldiff=" + fmt(logl_reldiff))
print("rf_dist=" + ("NA" if rf is None else str(rf)))
print("rf_norm=" + fmt(rf_norm))
print("support_max_absdiff=" + fmt(support_max))
print("gate_pass=" + gate)
PYEQ
LOGL_ABSDIFF=$(grep -oP "^logl_absdiff=\K.*" "$EQ_KV" 2>/dev/null); [ -z "$LOGL_ABSDIFF" ] && LOGL_ABSDIFF=NA
LOGL_RELDIFF=$(grep -oP "^logl_reldiff=\K.*" "$EQ_KV" 2>/dev/null); [ -z "$LOGL_RELDIFF" ] && LOGL_RELDIFF=NA
RF_DIST=$(grep -oP "^rf_dist=\K.*" "$EQ_KV" 2>/dev/null); [ -z "$RF_DIST" ] && RF_DIST=NA
RF_NORM=$(grep -oP "^rf_norm=\K.*" "$EQ_KV" 2>/dev/null); [ -z "$RF_NORM" ] && RF_NORM=NA
SUPPORT_MAX_ABSDIFF=$(grep -oP "^support_max_absdiff=\K.*" "$EQ_KV" 2>/dev/null); [ -z "$SUPPORT_MAX_ABSDIFF" ] && SUPPORT_MAX_ABSDIFF=NA
GATE_PASS=$(grep -oP "^gate_pass=\K.*" "$EQ_KV" 2>/dev/null); [ -z "$GATE_PASS" ] && GATE_PASS=NA
# A cell without valid engagement can never be a trusted GPU measurement, whatever the trees say.
[ "$ENGAGED" = OK ] || GATE_PASS=NA

# --- Groups L + M: derived DNA/AA + phase-decomposition scalars (big-int safe in python) ---
SINGLETON_SITES=NA; PER_PATTERN_MACS=NA; PATTERN_SATURATION=NA; DOMINANT_PHASE=NA
read -r SINGLETON_SITES PER_PATTERN_MACS PATTERN_SATURATION DOMINANT_PHASE < <(
  python3 - "${NSITE:-NA}" "$CONSTANT_SITES" "$INFORMATIVE_SITES" "$NUM_STATES" "$NTAX" "$RATE_CATEGORIES" "${NPAT:-NA}" "$CPU_MODELFINDER_S" "${CS:-NA}" "${CW:-NA}" <<'PYD'
import sys
def num(x):
    try: return float(x)
    except Exception: return None
sites,const,info,states,taxa,rcat,pat,mf,search,wall = sys.argv[1:11]
si,ci,ii = num(sites),num(const),num(info)
singleton = str(int(si-ci-ii)) if None not in (si,ci,ii) and si-ci-ii>=0 else "NA"
st,rc = num(states), (num(rcat) or 1.0)
ppm = str(int(st*st*rc)) if st else "NA"
pi,ti = num(pat), num(taxa)
sat = "NA"
if pi and si and st and ti:
    try:
        cap = min(si, st**int(ti))   # states^taxa can be astronomically large; python big-ints are fine
        sat = f"{pi/cap:.4f}" if cap else "NA"
    except Exception:
        sat = "NA"
mfv,sv,wv = (num(mf) or 0.0), (num(search) or 0.0), num(wall)
dom = "modelfinder" if mfv>=sv else "tree_search"
if not wv: dom = "NA"
print(singleton, ppm, sat, dom)
PYD
)
[ -z "$SINGLETON_SITES" ] && SINGLETON_SITES=NA
[ -z "$PER_PATTERN_MACS" ] && PER_PATTERN_MACS=NA
[ -z "$PATTERN_SATURATION" ] && PATTERN_SATURATION=NA
[ -z "$DOMINANT_PHASE" ] && DOMINANT_PHASE=NA

# --- SECOND-WAVE STUBS (deliberately deferred; see benchmark/BENCHMARK_SCHEMA.md groups I/J/K/L-roofline) ---
# TODO(second-wave): RAPL cpu_energy_j, smi gpu_power/energy, Nsight achieved_dram/fp64 + roofline_bound,
# and >=3 timing replicates (wall_sd_s/wall_cv). Emitted now as TODO_2W so the column exists in RUNS.tsv.
TODO_2W=TODO_2W

# --- Group M companion: long-form per-phase table (phase, arm, phase_wall_s, phase_speedup) ---
{
  echo -e "phase\tarm\tphase_wall_s\tphase_speedup"
  _ps() { awk -v a="$1" -v b="$2" 'BEGIN{ if(a+0==a && b+0==b && b>0) printf "%.2f", a/b; else print "NA" }'; }
  echo -e "modelfinder\tcpu\t${CPU_MODELFINDER_S}\t$(_ps "$CPU_MODELFINDER_S" "$GPU_MODELFINDER_S")"
  echo -e "modelfinder\tgpu\t${GPU_MODELFINDER_S}\tNA"
  echo -e "tree_search\tcpu\t${CS:-NA}\t$(_ps "${CS:-NA}" "${GS:-NA}")"
  echo -e "tree_search\tgpu\t${GS:-NA}\tNA"
  echo -e "total\tcpu\t${CW:-NA}\t${SP}"
  echo -e "total\tgpu\t${GW:-NA}\tNA"
} > "$OUT/PHASES.tsv"

# =============================================================================
#  Assemble the per-run RESULT.tsv row. The first 25 columns are UNCHANGED (order
#  and meaning) for backward compatibility; new schema-group columns are APPENDED.
# =============================================================================
HDR="genus\ttaxa\tsites\tpatterns\tcpu_config\tgpu_config\tthreads\tcard\tcpu_model\tnode\tengaged\tcpu_rc\tgpu_rc\tcpu_wall_s\tgpu_wall_s\tspeedup\tcpu_search_s\tgpu_search_s\tcpu_logl\tgpu_logl\tmean_gpu_util\tpct_gpu\tcpu_calls\trebuilds\trebinds"
ROW="$GENUS\t$NTAX\t${NSITE:-NA}\t${NPAT:-NA}\t$CPU_CONFIG\t$GPU_CONFIG\t$THREADS\t$CARDNAME\t$CPUMODEL\t$NODE\t$ENGAGED\t${CRC:-NA}\t${GRC:-NA}\t${CW:-NA}\t${GW:-NA}\t$SP\t${CS:-NA}\t${GS:-NA}\t$(getll "$OUT/cpu")\t$(getll "$OUT/gpu")\t$UTILMEAN\t${PCT_GPU:-NA}\t${CPU_CALLS:-NA}\t${REBUILDS:-NA}\t${REBINDS:-NA}"
# Group A: identity/provenance
HDR="$HDR\trun_id\tdatetime_utc\tdata_type\tdata_source\tseed\tcommand_line_cpu\tcommand_line_gpu"
ROW="$ROW\t$RUN_ID\t$DATETIME_UTC\t$DATA_TYPE\t$DATA_SOURCE\t$SEED\t$CMD_CPU\t$CMD_GPU"
# Group B: software/build
HDR="$HDR\tiqtree_version\ttool_git_commit\tbinary_sha256\treplay_cache"
ROW="$ROW\t$IQTREE_VERSION\t$TOOL_GIT_COMMIT\t$BINARY_SHA256\t$REPLAY_CACHE"
# Group C: hardware
HDR="$HDR\tgpu_vram_total_gb\tcuda_driver_version\tcompute_capability\tcpu_cores_alloc\tpartition\trestarts"
ROW="$ROW\t$GPU_VRAM_TOTAL_GB\t$CUDA_DRIVER_VERSION\t$COMPUTE_CAPABILITY\t$CPU_CORES_ALLOC\t$PARTITION\t$RESTARTS"
# Group D: alignment
HDR="$HDR\tinformative_sites\tsingleton_sites\tconstant_sites\tgap_ambiguity_pct\tmodel\trate_categories"
ROW="$ROW\t$INFORMATIVE_SITES\t$SINGLETON_SITES\t$CONSTANT_SITES\t$GAP_AMBIGUITY_PCT\t$MODEL_USED\t$RATE_CATEGORIES"
# Group E: config
HDR="$HDR\tgpu_flags\tsupport_settings"
ROW="$ROW\t$GPU_FLAGS\t$SUPPORT_SETTINGS"
# Group F: GPU engagement
HDR="$HDR\tgpu_banner\tgpu_calls\tkernel_launches\tnewton_ondevice\tdevice_errors\tutil_sm_mean\tutil_sm_sd\tutil_sm_max\tutil_mem_mean\tutil_mem_sd\tutil_mem_max\tutil_nsamples\tpeak_vram_mb"
ROW="$ROW\t$GPU_BANNER\t${GPU_CALLS:-NA}\t${KERNEL_LAUNCHES:-NA}\t${NEWTON_ONDEVICE:-NA}\t${DEVICE_ERRORS:-NA}\t$UTIL_SM_MEAN\t$UTIL_SM_SD\t$UTIL_SM_MAX\t$UTIL_MEM_MEAN\t$UTIL_MEM_SD\t$UTIL_MEM_MAX\t$UTIL_NSAMPLES\t$PEAK_VRAM_MB"
# Group G: timing
HDR="$HDR\tcpu_modelfinder_s\tgpu_modelfinder_s\tcpu_bootstrap_s\tgpu_bootstrap_s\tcpu_totalcpu_s\tpeak_ram_mb"
ROW="$ROW\t$CPU_MODELFINDER_S\t$GPU_MODELFINDER_S\t$CPU_BOOTSTRAP_S\t$GPU_BOOTSTRAP_S\t$CPU_TOTALCPU_S\t$PEAK_RAM_MB"
# Group H: equivalence
HDR="$HDR\tlogl_absdiff\tlogl_reldiff\trf_dist\trf_norm\tsupport_max_absdiff\tgate_pass"
ROW="$ROW\t$LOGL_ABSDIFF\t$LOGL_RELDIFF\t$RF_DIST\t$RF_NORM\t$SUPPORT_MAX_ABSDIFF\t$GATE_PASS"
# Group L: DNA/AA characterisation
HDR="$HDR\tnum_states\tper_pattern_macs\tpattern_saturation"
ROW="$ROW\t$NUM_STATES\t$PER_PATTERN_MACS\t$PATTERN_SATURATION"
# Group M: run-level phase decomposition (long form in PHASES.tsv)
HDR="$HDR\tdominant_phase"
ROW="$ROW\t$DOMINANT_PHASE"
# Second-wave stubs (clearly marked TODO_2W; see note above)
HDR="$HDR\tcpu_energy_j\tgpu_power_mean_w\tgpu_energy_j\tachieved_dram_gbs\tachieved_fp64_gflops\troofline_bound\tn_replicates\twall_sd_s\twall_cv"
ROW="$ROW\t$TODO_2W\t$TODO_2W\t$TODO_2W\t$TODO_2W\t$TODO_2W\t$TODO_2W\t1\t$TODO_2W\t$TODO_2W"
{ echo -e "$HDR"; echo -e "$ROW"; } > "$OUT/RESULT.tsv"
echo; echo "== RESULT.tsv ($(head -1 "$OUT/RESULT.tsv" | tr '\t' '\n' | wc -l) columns) =="; column -t -s$'\t' "$OUT/RESULT.tsv" 2>/dev/null | cut -c1-200

# Keep the master current per job (benchmark/BENCHMARK_SCHEMA.md data-update rule): refresh RUNS.tsv the moment
# this cell finishes. Best-effort and non-fatal; a single-writer flock (non-blocking) means concurrent
# cells never collide -- if another job holds the lock it is already aggregating this cell's dir too.
AGG="$REPO/benchmark/scripts/aggregate_runs.py"
if [ -r "$AGG" ]; then
  flock -n "$REPO/benchmark/runs/.runs_agg.lock" python3 "$AGG" >/dev/null 2>&1 \
    && echo "== RUNS.tsv refreshed ==" || echo "== RUNS.tsv refresh skipped (locked/again later) =="
fi
