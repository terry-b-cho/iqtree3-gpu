# dna_hc_n60_L2M - simulated DNA benchmark input

Built by `benchmark/scripts/dev/build_complexity_datasets.sh` to close the DNA/protein complexity gap.

- taxa: 60, sites: 2000000
- guide tree: `RANDOM{yh/60}`
- model: `GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}` (true model known by construction)
- seed: 3060
- command: `iqtree3 --alisim aln -t "RANDOM{yh/60}" -m "GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}" --length 2000000 -seed 3060`

Complexity = (2n-3) x patterns x 4 rate categories x 16 states^2. DNA patterns are capped by 4^n,
which is why paired DNA/protein comparisons are only possible for n >= 14.
