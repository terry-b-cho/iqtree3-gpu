#!/usr/bin/env bash
# =============================================================================
#  Build the FULL Zenodo deposit for the IQ-TREE3-GPU record (author action R27, round 8).
#
#  build_zenodo_bundle.sh stages only the 14 alignments too large for GitHub. This script
#  assembles everything the manuscript's Data Availability statement promises, in one folder:
#    data/protein/            35 amino-acid alignments (+ PROVENANCE.md)
#    data/dna/        all 26 simulated nucleotide sets (aln.phy, PROVENANCE.md, alisim.log, guide tree)
#    results/benchmarks/ RUNS.tsv/.csv, PHASE_STEPS, REPLICATES_*, KEY_NUMBERS, cpu_production_scaling,
#                        and every paired cell directory (both arms' logs, trees, support, RESULT.tsv)
#    results/profiling/  nsys/nsight summaries and the launch-latency measurement
#    (not staged: results/figures/ and the scoping-review packet. The figures and Source Data travel
#     with the submission and the packet with the code repository; both were withdrawn from the
#     record on 2026-10-07 so that it holds the raw inputs, the measured record and the code.)
#    scripts/ src/ config/ containers/ examples/ build_iqtree_gpu.sh and the top-level documents
#    MANIFEST.sha256     one line per file, verified with `sha256sum -c`
#
#  Hard links where the filesystem allows, so the staging costs no extra disk. gitignored.
#  The DOI and the release version are minted on deposit: this script writes neither, and the
#  manuscript keeps the literal "[to be inserted on deposit]" until then.
#
#  Usage: bash scripts/dev/build_zenodo_record.sh [version]
# =============================================================================
set -uo pipefail
_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "$PWD") || { echo "FATAL: cannot locate repository root" >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"
cd "$REPO" || exit 1

VER="${1:-$(cat VERSION 2>/dev/null || echo 0.1.0-dev)}"
B="zenodo_upload_bundle/iqtree3-gpu-record-v${VER}"
rm -rf "$B"; mkdir -p "$B"

stage () {  # stage SRC_PATH  (file or directory, relative to REPO); hard-link, fall back to copy
  local src="$1"; [ -e "$src" ] || { echo "  (absent) $src"; return; }
  mkdir -p "$B/$(dirname "$src")"
  if [ -d "$src" ]; then
    cp -rl "$src" "$B/$(dirname "$src")/" 2>/dev/null || cp -r "$src" "$B/$(dirname "$src")/"
  else
    ln "$src" "$B/$src" 2>/dev/null || cp "$src" "$B/$src"
  fi
}

echo "== staging the record into $B =="
for d in data results/profiling scripts src config containers examples; do stage "$d"; done
for f in results/benchmarks/RUNS.tsv results/benchmarks/RUNS.csv results/benchmarks/RUNS_summary.tsv \
         results/benchmarks/RUNS_summary.csv results/benchmarks/PHASE_STEPS.tsv \
         results/benchmarks/REPLICATES_protein20T.tsv results/benchmarks/REPLICATES_dna20T.tsv \
         results/benchmarks/KEY_NUMBERS.tsv results/benchmarks/KEY_NUMBERS.md results/benchmarks/Supplementary_Data_benchmark.xlsx \
         results/benchmarks/cpu_production_scaling.tsv \
         README.md USER_GUIDE.md REPRODUCING.md BENCHMARK_SCHEMA.md DEPENDENCIES.md CHANGELOG.md \
         CITATION.cff LICENSE VERSION build_iqtree_gpu.sh CONTRIBUTING.md CODE_OF_CONDUCT.md; do stage "$f"; done
stage results/benchmarks/paired
stage results/benchmarks/dna_cpu_reference
find "$B" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null

cat > "$B/README.md" <<'MD'
# IQ-TREE3-GPU: the archived record (inputs, measurements, code)

This deposit holds the raw inputs and the measured record behind the manuscript, with the code
that produced them: every benchmark alignment (`data/protein/`, `data/dna/`) and the source genomes
they were built from, the measured record (`results/benchmarks/`, `results/profiling/`), the CUDA
backend source (`src/`), the build script and the scripts that produced every table and every
number quoted in the text (`scripts/`). `MANIFEST.sha256` lists every file; verify with

    sha256sum -c MANIFEST.sha256

The figure files and the Source Data are supplied with the submission, and the scoping-review
packet (Supplementary Note 1) is distributed with the code repository; neither is part of this
record. Reproduction starts at `REPRODUCING.md`. The release version and the DOI of this deposit
are cited from `CITATION.cff` and the manuscript's Data Availability statement.
MD

echo "== hashing =="
( cd "$B" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
n=$(wc -l < "$B/MANIFEST.sha256")
echo "== $n files staged into $B =="
echo "   apparent size: $(du -sh --apparent-size "$B" | cut -f1)"
echo "   AA alignments: $(ls -d "$B"/data/protein/*/ | wc -l)   DNA sets: $(ls -d "$B"/data/dna/*/ | wc -l)   paired cells: $(ls -d "$B"/results/benchmarks/paired/*/ | wc -l)"
echo "   upload '$(basename "$B")' to Zenodo as one deposit; then write the DOI into the manuscript,"
echo "   CITATION.cff, REPRODUCING.md and USER_GUIDE.md where '[to be inserted on deposit]' stands."
