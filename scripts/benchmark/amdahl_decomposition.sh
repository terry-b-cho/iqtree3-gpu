#!/bin/bash
#SBATCH --job-name=amdahl
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
#SBATCH -t 8:00:00
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/amdahl_%j.out
#SBATCH -e logs/amdahl_%j.err
# =============================================================================
#  IQ-TREE3-GPU - Amdahl decomposition: WHERE does the GPU speedup go?
# =============================================================================
#  An end-to-end production measurement (-m TESTNEW --ufboot --alrt) mixes GPU-accelerated
#  likelihood work with host-side work the GPU never touches
#  (model eigendecompositions, traversal bookkeeping, bootstrap resampling). This
#  experiment strips the host-side work away in tiers:
#
#    T1  fixed topology + fixed model  (-te tree -m LG+G4)  -> almost pure likelihood
#    T2  tree search, fixed model      (-m LG+G4)           -> + topology search
#    T3  full production               (already measured)   -> + ModelFinder + UFBoot
#
#  If speedup(T1) >> speedup(T3), the end-to-end number is Amdahl-bounded by host-side
#  work and the likelihood engine itself IS substantially faster.
#
#  CONTROL: both arms use the SAME binary (the GPU build), differing only by the -gpu
#  flag, so the only variable is the likelihood kernel swap. The stock binary is run
#  once on T1 to confirm the two CPU arms agree.
# =============================================================================
set -uo pipefail
# toolchain: gcc FIRST (libstdc++ for the C++ binary), then cuda (libcudart.so.12)
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
GPUBIN=$IQTREE_BUILD_ROOT/iqtree3_gpu_build/iqtree3
STOCK=$IQTREE_BUILD_ROOT/iqtree3_build/iqtree3
OUT=$REPO/results/benchmarks/amdahl
TSV=$OUT/amdahl_raw.tsv
T=20
mkdir -p "$OUT"
[ -s "$TSV" ] || echo -e "genus\ttaxa\ttier\tarm\tthreads\twall_s\tcpu_s\tlogl\tcard" > "$TSV"

CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
CPUMODEL=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//;s/ /_/g')
echo "== CPU model: $CPUMODEL  (both arms run on THIS node -> hardware controlled) =="
echo "== node $(hostname) card=$CARD threads=$T =="

run () {  # genus tier arm extra_flags...
  local g=$1 tier=$2 arm=$3; shift 3
  local aln=$REPO/data/protein/g__$g/concat_alignment.faa
  local pfx=$OUT/${g}__${tier}__${arm}
  local bin=$GPUBIN; local gflag="-gpu"
  [ "$arm" = cpu ]    && gflag=""
  [ "$arm" = stock ]  && { bin=$STOCK; gflag=""; }
  local ntax; ntax=$(grep -c '^>' "$aln")
  echo "-- $g $tier $arm"
  local t0=$SECONDS
  /usr/bin/time -f "%e %U %S" -o "$pfx.time" \
     "$bin" -s "$aln" -T $T $gflag -seed 1 --prefix "$pfx" -redo "$@" > "$pfx.stdout" 2>&1
  local rc=$? wall=$((SECONDS-t0))
  local cpu; cpu=$(tail -1 "$pfx.time" 2>/dev/null | awk 'NF==3{printf "%.1f", $2+$3}')
  local ll;  ll=$(grep -oP 'BEST SCORE FOUND : \K[-0-9.]+' "$pfx.iqtree" 2>/dev/null | tail -1)
  [ -z "$ll" ] && ll=$(grep -oP 'Log-likelihood of the tree: \K[-0-9.]+' "$pfx.iqtree" 2>/dev/null | tail -1)
  echo -e "$g\t$ntax\t$tier\t$arm\t$T\t$wall\t${cpu:-NA}\t${ll:-NA}\t$CARD" >> "$TSV"
  echo "   rc=$rc wall=${wall}s cpu=${cpu:-NA}s logL=${ll:-NA}"
  [ "$rc" -ne 0 ] && { echo "   *** FAILED (rc=$rc): $(head -2 "$pfx.stdout")"; }
}

# ---- Tier 1: fixed topology, fixed model -> near-pure likelihood evaluation ----
for g in CAG-269 Lachnospira Roseburia; do
  # Fixed topology from the PAIRED benchmark. The old results/benchmarks/matrix was archived on
  # 2026-07-21 (produced by a binary whose GPU kernel was never installed), so its trees are gone.
  TREE=$(ls $REPO/results/benchmarks/paired/g__${g}__T20__*/gpu.treefile 2>/dev/null | head -1)
  [ -f "$TREE" ] || { echo "SKIP $g T1: no fixed tree"; continue; }
  cp "$TREE" "$OUT/${g}.fixed.tre"
  for arm in gpu cpu; do run $g T1 $arm -te "$OUT/${g}.fixed.tre" -m LG+G4; done
done
run CAG-269 T1 stock -te "$OUT/CAG-269.fixed.tre" -m LG+G4    # CPU-arm cross-check

# ---- Tier 2: tree search with a FIXED model (no ModelFinder, no bootstrap) ----
for g in CAG-269 Lachnospira; do
  for arm in gpu cpu; do run $g T2 $arm -m LG+G4; done
done

echo "== done =="; column -t "$TSV"
