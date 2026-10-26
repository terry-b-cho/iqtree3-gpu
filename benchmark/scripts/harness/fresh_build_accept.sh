#!/bin/bash
#SBATCH --job-name=freshbuild
#SBATCH -p gpu_requeue
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: a LITERAL value on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$SLURM_GRES" here is passed to the scheduler verbatim and the allocation fails).
# This is the default for a bare `sbatch scripts/...`; the submitters pass --gres="$SLURM_GRES" on
# the sbatch COMMAND LINE, where the shell does expand it, and a command-line flag overrides the
# directive. Set SLURM_GRES in config/site.env to change it everywhere the submitters are used.
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --exclude=compute-gc-17-242
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/freshbuild_%j.out
#SBATCH -e logs/freshbuild_%j.err
# Release acceptance test (#18 final check): build the shippable src/ from scratch (fresh clone + patch
# + new files), then confirm the resulting binary produces correct, GPU-engaged trees for BOTH protein
# and DNA. This is what a public user gets. PASS = engagement OK + RF=0 on protein and DNA n6/n30.
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
RF=$REPO/scripts/rf_logl_check.py
W=/tmp/freshbuild_$SLURM_JOB_ID; rm -rf "$W"; mkdir -p "$W"
echo "== $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) =="

echo; echo "=== [build] build_iqtree_gpu.sh from shippable src/ into a fresh dir (arch 89=L40S) ==="
# A fresh clone needs Eigen 3.4 + Boost headers; build_iqtree_gpu.sh reads these env vars for them.
# (Public-repo TODO: document this dependency in docs/DEPENDENCIES.md / vendor the headers.)
export EIGEN3_INCLUDE_DIR=$IQTREE_BUILD_ROOT/deps/eigen-3.4.0
export Boost_INCLUDE_DIR=$IQTREE_BUILD_ROOT/deps/boostenv/include
bash $REPO/build_iqtree_gpu.sh "$W/build" 89 2>&1 | tail -8
BIN="$W/build/iqtree3"
[ -x "$BIN" ] || { echo "FRESH BUILD FAILED: no binary"; exit 1; }
echo "  binary: $BIN  sha256=$(sha256sum "$BIN" | cut -c1-16)"

chk () { # tag aln model
  local tag=$1 aln=$2 model=$3
  "$BIN" -s "$aln" -m "$model" -T 1 -seed 1 --prefix "$W/${tag}_c" -redo > "$W/${tag}_c.log" 2>&1
  "$BIN" -s "$aln" -m "$model" -T 1 -gpu --gpu-strict -seed 1 --prefix "$W/${tag}_g" -redo > "$W/${tag}_g.log" 2>&1
  local grc=$? crash rf pg cc nod
  crash=$(grep -c "Using one NNI reduces LogL\|SIGNAL ABORTED" "$W/${tag}_g.log" 2>/dev/null)
  rf=$(python3 "$RF" "$W/${tag}_c" "$W/${tag}_g" $tag 2>&1|grep -oP 'rf_dist=\K\S+')
  pg=$(grep -oP "pct_gpu=\K[0-9.]+" "$W/${tag}_g.log"|tail -1)
  cc=$(grep -oP "cpu_calls=\K[0-9]+" "$W/${tag}_g.log"|tail -1)
  nod=$(grep -oP "newton_ondevice=\K[0-9]+" "$W/${tag}_g.log"|tail -1)
  printf "  %-16s grc=%s crash=%s RF=%-4s pct_gpu=%-6s cpu_calls=%-3s newton=%s\n" "$tag" "$grc" "$crash" "${rf:-NA}" "${pg:-NA}" "${cc:-NA}" "${nod:-NA}"
}
echo; echo "=== [verify] the freshly-built shippable binary, protein + DNA ==="
chk protein_n8 $REPO/benchmark/data/protein/n08_g__Hominicoprocola/concat_alignment.faa LG+F+G4
chk dna_n6     $REPO/benchmark/data/dna/dna_n06/aln.phy GTR+F+G4
chk dna_n30    $REPO/benchmark/data/dna/dna_n30/aln.phy GTR+F+G4
echo; echo "== PASS = fresh clone builds + every row crash=0, RF=0, pct_gpu=100, cpu_calls=0 =="
