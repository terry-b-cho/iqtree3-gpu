# dna_pt_n30 - simulated DNA benchmark input

Built by `scripts/dev/build_complexity_datasets.sh` to close the DNA/protein complexity gap.

- taxa: 30, sites: 3972289
- guide tree: `../../results/benchmarks/paired/n30_g__Enterousia__T20__l40s/gpu.treefile`
- model: `GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}` (true model known by construction)
- seed: 2030
- command: `iqtree3 --alisim aln -t "../../results/benchmarks/paired/n30_g__Enterousia__T20__l40s/gpu.treefile" -m "GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}" --length 3972289 -seed 2030`

Complexity = (2n-3) x patterns x 4 rate categories x 16 states^2. DNA patterns are capped by 4^n,
which is why paired DNA/protein comparisons are only possible for n >= 14.
