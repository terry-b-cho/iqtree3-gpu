# dna_hc_n30_L3M - simulated DNA benchmark input

Built by `scripts/dev/build_complexity_datasets.sh` to close the DNA/protein complexity gap.

- taxa: 30, sites: 3000000
- guide tree: `RANDOM{yh/30}`
- model: `GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}` (true model known by construction)
- seed: 3031
- command: `iqtree3 --alisim aln -t "RANDOM{yh/30}" -m "GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}" --length 3000000 -seed 3031`

Complexity = (2n-3) x patterns x 4 rate categories x 16 states^2. DNA patterns are capped by 4^n,
which is why paired DNA/protein comparisons are only possible for n >= 14.
