#!/bin/bash
#SBATCH --job-name=dnaref
#SBATCH -p short
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH -t 6:00:00
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# NOTE: repo-RELATIVE on purpose. SLURM does not expand shell variables in #SBATCH directives
# (a literal "$REPO" here would create a directory called '$REPO'), and an absolute path would pin
# the repository to one filesystem. SLURM resolves a relative -o/-e against the job's working
# directory, which defaults to the SUBMIT directory: run sbatch FROM THE REPOSITORY ROOT, or let a
# submitter drive it (submitters pass an absolute -o/-e on the command line, which overrides
# these). logs/.gitkeep is tracked because slurmd opens this file BEFORE the script body runs,
# so an in-script mkdir cannot save a fresh clone.
#SBATCH -o logs/dnaref_%j.out
#SBATCH -e logs/dnaref_%j.err
# =============================================================================
#  CPU reference trees + timings for the simulated DNA sweep.
# =============================================================================
#  Runs entirely on the STOCK CPU binary, so it needs no GPU and cannot be disturbed by (or disturb)
#  the GPU build. Two purposes:
#    1. Reference trees and log-likelihoods for every DNA dataset, so when the GPU DNA kernel exists
#       its equivalence can be checked immediately rather than after another CPU run.
#    2. The CPU side of the eventual DNA speedup, and an early read on how DNA cost compares to
#       amino acid at matched tree sizes -- DNA does 4x4=16 multiply-adds per site/category against
#       20x20=400 for protein, so per-site work is ~25x lower and the kernel is expected to be far
#       more memory-bound. That shapes what the DNA kernel should optimise for.
#  Also records whether ModelFinder's chosen DNA model recovers the TRUE simulated model.
# =============================================================================
set -uo pipefail
[ -r /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh 2>/dev/null || true
[ "${NO_LMOD:-0}" = "1" ] || { command -v module >/dev/null 2>&1 && \
  module load $SLURM_MODULES >/dev/null 2>&1; } || true
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
STOCK=$IQTREE_BUILD_ROOT/iqtree3_build/iqtree3
OUT=$REPO/results/benchmarks/dna_cpu_reference
TSV=$OUT/dna_cpu_reference.tsv
mkdir -p "$OUT"
TRUE=$(cat $REPO/data/dna/TRUE_MODEL.txt 2>/dev/null)
echo -e "dataset\ttaxa\tsites\tthreads\twall_s\tlogl\tbest_model\ttrue_model" > "$TSV"
echo "== DNA CPU reference on $(hostname); true model: $TRUE =="
for d in $REPO/data/dna/dna_n*/; do
  tag=$(basename "$d"); aln="$d/aln.phy"
  [ -f "$aln" ] || { echo "  skip $tag (no alignment)"; continue; }
  n=$(head -1 "$aln" | awk '{print $1}'); L=$(head -1 "$aln" | awk '{print $2}')
  t0=$SECONDS
  "$STOCK" -s "$aln" -m TESTNEW -T 20 -seed 1 --prefix "$OUT/$tag" -redo > "$OUT/$tag.stdout" 2>&1
  rc=$?; w=$((SECONDS-t0))
  ll=$(grep -oP "Log-likelihood of the tree: \K-?[0-9.]+" "$OUT/$tag.iqtree" 2>/dev/null | tail -1)
  bm=$(grep -oP "Best-fit model according to BIC: \K\S+" "$OUT/$tag.iqtree" 2>/dev/null | tail -1)
  printf "  %-10s n=%-3s sites=%-7s rc=%s wall=%-6ss model=%-22s logL=%s\n" "$tag" "$n" "$L" "$rc" "$w" "${bm:-NA}" "${ll:-NA}"
  echo -e "$tag\t$n\t$L\t20\t$w\t${ll:-NA}\t${bm:-NA}\t$TRUE" >> "$TSV"
done
echo; column -t "$TSV"
