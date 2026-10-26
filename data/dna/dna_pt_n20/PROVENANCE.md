# dna_pt_n20 - simulated DNA benchmark input

Built by `scripts/dev/build_complexity_datasets.sh` to close the DNA/protein complexity gap.

- taxa: 20, sites: 9486152
- guide tree: `../../results/benchmarks/paired/n20_g__Coprococcus_C__T20__l40s/gpu.treefile`
- model: `GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}` (true model known by construction)
- seed: 2020
- command: `iqtree3 --alisim aln -t "../../results/benchmarks/paired/n20_g__Coprococcus_C__T20__l40s/gpu.treefile" -m "GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}" --length 9486152 -seed 2020`

Complexity = (2n-3) x patterns x 4 rate categories x 16 states^2. DNA patterns are capped by 4^n,
which is why paired DNA/protein comparisons are only possible for n >= 14.
