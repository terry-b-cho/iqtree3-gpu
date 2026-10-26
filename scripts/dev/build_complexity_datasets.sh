#!/usr/bin/env bash
# =============================================================================
#  build_complexity_datasets.sh - datasets that CLOSE THE DNA/PROTEIN COMPLEXITY GAP.
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
#  WHY. The original ladder (build_dna_datasets.sh + data/protein/n??_*) varied only TAXA and applied the same n=4..30
#  ladder to both data types. But complexity = branches x patterns x rate_cats x states^2, and
#  states^2 is 16 for DNA vs 400 for AA -- a fixed 25x gap at identical taxa and sites, plus ~1.5x
#  from rate categories (DNA fixed +G4 = 4; ModelFinder picks +R3..+R8 for protein). The measured
#  ranges therefore barely overlap:
#      DNA     5.9e4 .. 4.8e8        protein  2.2e8 .. 2.6e10
#  so the GPU benefit for DNA at HIGH complexity, and for protein at LOW complexity, was never
#  measured. This script builds the missing inputs.
#
#  HARD LIMIT (a real result, not a gap in the design): DNA patterns are capped by 4^n, so maximum
#  DNA complexity is (2n-3) x min(L, 4^n) x 64 = 8.2e4 at n=4, 2.4e6 at n=6, 5.5e7 at n=8. DNA
#  CANNOT reach protein complexity below n ~= 10-14 at any alignment length. The paired sets below
#  therefore start at n=14.
#
#  WHAT IT BUILDS
#   A. dna_pt_n<NN>      - DNA simulated along the PROTEIN run's own inferred topology (same taxa,
#                          same tree, same branch lengths), length chosen so DNA complexity matches
#                          that protein cell's complexity. This is the faithful answer to "run DNA on
#                          the same trees as protein": the upstream pipeline stages only amino-acid alignments,
#                          and back-translating needs CDS that were never staged, so the topology --
#                          not the alphabet -- is what can be shared.
#   B. dna_hc_*          - high-complexity DNA on random Yule-Harding trees, extending the two axes
#                          never explored: taxa > 30 and sites > 233k.
#   C. p_lc_*            - low-complexity PROTEIN by seeded column subsampling of a staged real
#                          alignment (same precedent as the taxon subsampling in
#                          build_1thread_genomes.py: real data, fewer columns, both benchmark
#                          arms byte-identical).
#
#  Usage:  build_complexity_datasets.sh [pilot|rest|all]     (default: all)
#          RATIO=0.55 to override the assumed DNA pattern/site ratio (the pilot measures the truth).
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
BIN=${IQTREE_BIN:-$IQTREE_BUILD_ROOT/iqtree3_gpu_build/iqtree3}
DNAOUT=$REPO/data/dna
AAOUT=$REPO/data/protein
PAIRED=$REPO/results/benchmarks/paired
MODE=${1:-all}
RATIO=${RATIO:-0.5164}        # MEASURED on the dna_pt_n30 pilot (2,051,474 patterns / 3,972,289 sites)

# Same model string, seeding convention and rationale as build_dna_datasets.sh (reproducible, true model known).
MODEL='GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}'

# ---- A. paired DNA on the protein topologies: n -> "<protein cell dir>:<protein complexity>" ----
declare -A PT=(
  [14]="n14_g__Nanosyncoccus__T20__l40s:1.48e9"
  [17]="n17_g__Lachnospira__T20__l40s:1.11e10"
  [20]="n20_g__Coprococcus_C__T20__l40s:1.16e10"
  [23]="n23_g__Onthovivens__T20__l40s:7.20e9"
  [26]="n26_g__Ruminococcus__T20__l40s:1.72e10"
  [28]="n28_g__Roseburia__T20__l40s:2.59e10"
  [30]="n30_g__Enterousia__T20__l40s:7.97e9"
)
# VRAM guard: DNA partial = patterns x 4 states x 4 cats x 8 B = 128 B/pattern, held for ~n slots.
# Keep patterns x n <= PAT_SLOT_CAP so the arena stays ~<=20 GB on a 48 GB L40S; a cell that does not
# fit would silently fall back to the CPU and FAIL the engagement gate, so we cap length instead.
PAT_SLOT_CAP=${PAT_SLOT_CAP:-120000000}   # patterns x taxa; keeps the arena <= ~20 GB
# (calibrated: arena_bytes ~= patterns x 128 B x 1.3 x taxa, from the measured n28 protein cell)

build_dna () {   # tag  taxa  length  tree_or_RANDOM  seed
  local tag=$1 n=$2 L=$3 tree=$4 seed=$5
  local d="$DNAOUT/$tag"; mkdir -p "$d"
  if [ -s "$d/aln.phy" ]; then echo "   skip $tag (exists)"; return; fi
  local targ
  if [ "$tree" = "RANDOM" ]; then targ="RANDOM{yh/$n}"; else targ="$tree"; fi
  "$BIN" --alisim "$d/aln" -t "$targ" -m "$MODEL" --length "$L" -seed "$seed" -redo \
      > "$d/alisim.log" 2>&1
  if [ -s "$d/aln.phy" ]; then
    local gn gl; gn=$(head -1 "$d/aln.phy" | awk '{print $1}'); gl=$(head -1 "$d/aln.phy" | awk '{print $2}')
    printf "   %-18s n=%-4s len=%-9s (%s)\n" "$tag" "$gn" "$gl" "$(du -h "$d/aln.phy" | cut -f1)"
    cat > "$d/PROVENANCE.md" <<EOF
# $tag - simulated DNA benchmark input

Built by \`scripts/dev/build_complexity_datasets.sh\` to close the DNA/protein complexity gap.

- taxa: $gn, sites: $gl
- guide tree: \`$targ\`
- model: \`$MODEL\` (true model known by construction)
- seed: $seed
- command: \`iqtree3 --alisim aln -t "$targ" -m "$MODEL" --length $L -seed $seed\`

Complexity = (2n-3) x patterns x 4 rate categories x 16 states^2. DNA patterns are capped by 4^n,
which is why paired DNA/protein comparisons are only possible for n >= 14.
EOF
  else
    echo "   FAILED $tag"; tail -3 "$d/alisim.log" | sed 's/^/      /'
  fi
}

do_pt () {   # build one paired dataset for taxa n
  local n=$1 spec="${PT[$1]}"
  local cell="${spec%%:*}" comp="${spec##*:}"
  local tree="$PAIRED/$cell/gpu.treefile"
  [ -s "$tree" ] || { echo "   MISSING protein tree for n=$n ($tree)"; return; }
  # L to match that protein cell's complexity, then apply the VRAM cap.
  local L
  L=$(python3 -c "
n=$n; comp=float('$comp'); r=float('$RATIO'); cap=float('$PAT_SLOT_CAP')
L = comp / ((2*n-3) * r * 64)
pat = r*L
if pat*n > cap: L = cap/(n*r)          # VRAM-capped
print(int(round(L)))")
  build_dna "dna_pt_n$(printf %02d "$n")" "$n" "$L" "$tree" $((2000 + n))
}

# ---- B2. ULTRA-high-complexity DNA: beyond the protein maximum (2.59e10) ----
#  Reviewer request: data points past 1e10 and past our previous ceiling, for BOTH alphabets.
#  Host RAM is the binding constraint, not VRAM -- the engine adapts its device arena, so the OOM
#  cells still completed their GPU arm. From 18 measured DNA cells,
#      host_RAM ~= patterns x taxa x 600 B,  and  complexity ~= 128 x taxa x patterns
#  hence host_RAM_GB ~= complexity x 4.7e-9: ~160 GB at 3.4e10 and ~200 GB at 4.3e10. The l40s node
#  (compute-g-17-169) has 500 GB, so these fit and will run at ~85% utilisation - a large request
#  that is actually used, unlike a flat reservation.
do_ultra () {
  build_dna dna_ux_n60_L6M   60 6000000 RANDOM 3062
  build_dna dna_ux_n100_L4M 100 4000000 RANDOM 3101
}

# ---- B. high-complexity DNA ladder (taxa > 30 and sites > 233k were never explored) ----
do_hc () {
  build_dna dna_hc_n30_L1M   30 1000000 RANDOM 3030
  build_dna dna_hc_n30_L3M   30 3000000 RANDOM 3031
  build_dna dna_hc_n60_L2M   60 2000000 RANDOM 3060
  build_dna dna_hc_n60_L4M   60 4000000 RANDOM 3061
  build_dna dna_hc_n100_L2M 100 2000000 RANDOM 3100
}

# ---- C. low-complexity PROTEIN by seeded column subsampling of a staged real alignment ----
do_lc () {
  python3 - "$AAOUT" <<'PY'
import os, random, sys, textwrap
AA = sys.argv[1]
# (new dataset, source dataset, columns kept)  -- source alignments are real staged MAG data
JOBS = [("p_lc_n04_L200",   "n04_g__Brotaphodocola",    200),
        ("p_lc_n04_L2k",    "n04_g__Brotaphodocola",   2000),
        ("p_lc_n04_L20k",   "n04_g__Brotaphodocola",  20000),
        ("p_lc_n06_L20k",   "n06_g__CAG-269",         20000),
        ("p_lc_n08_L50k",   "n08_g__Hominicoprocola", 50000)]

def read_fasta(p):
    names, seqs, cur = [], [], []
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if cur: seqs.append("".join(cur)); cur = []
            names.append(line[1:])
        elif line:
            cur.append(line)
    if cur: seqs.append("".join(cur))
    return names, seqs

for tag, src, keep in JOBS:
    sp = f"{AA}/{src}/concat_alignment.faa"
    dd = f"{AA}/{tag}"; op = f"{dd}/concat_alignment.faa"
    if not os.path.exists(sp):
        print(f"   MISSING source {sp}"); continue
    if os.path.exists(op) and os.path.getsize(op) > 0:
        print(f"   skip {tag} (exists)"); continue
    names, seqs = read_fasta(sp)
    L = len(seqs[0])
    if keep >= L:
        print(f"   skip {tag}: requested {keep} >= source length {L}"); continue
    # deterministic, seeded column subsample; columns kept in original order
    cols = sorted(random.Random(hash(tag) & 0xffff).sample(range(L), keep))
    os.makedirs(dd, exist_ok=True)
    with open(op, "w") as fh:
        for nm, s in zip(names, seqs):
            fh.write(f">{nm}\n")
            sub = "".join(s[c] for c in cols)
            for i in range(0, len(sub), 60):
                fh.write(sub[i:i+60] + "\n")
    open(f"{dd}/PROVENANCE.md", "w").write(textwrap.dedent(f"""\
        # {tag} - low-complexity protein benchmark input

        Built by `scripts/dev/build_complexity_datasets.sh`. Real staged MAG data with
        alignment COLUMNS subsampled so the protein arm reaches the low-complexity regime that the
        taxa-only ladder never covered (every previous protein dataset used a full-length 99k-233k
        site alignment, so even n=4 sat at 2.2e8 complexity).

        - source: `data/protein/{src}/concat_alignment.faa` ({len(names)} taxa, {L} sites)
        - columns kept: {keep} (seeded `random.Random` sample, original order preserved)
        - sequences: unchanged real sequence data; only the number of columns is reduced
        - both benchmark arms read this identical file, which is the only requirement for a
          performance comparison; the RF=0 equivalence gate applies exactly as elsewhere
        """))
    print(f"   {tag:18s} n={len(names)} sites={keep}")
PY
}

echo "== building complexity-gap datasets (mode=$MODE, ratio=$RATIO) =="
case "$MODE" in
  pilot) do_pt 30 ;;                                   # calibration dataset only
  ultra) do_ultra ;;                                   # beyond the protein maximum
  rest)  for n in 14 17 20 23 26 28; do do_pt $n; done; do_hc; do_lc ;;
  all)   for n in 14 17 20 23 26 28 30; do do_pt $n; done; do_hc; do_ultra; do_lc ;;
  *)     echo "usage: $0 [pilot|rest|ultra|all]"; exit 1 ;;
esac
echo "== done =="
