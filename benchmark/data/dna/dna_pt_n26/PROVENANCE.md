# dna_pt_n26 - simulated DNA benchmark input

Built by `benchmark/scripts/dev/build_complexity_datasets.sh` to close the DNA/protein complexity gap.

- taxa: 26, sites: 8937615
- guide tree: `../../benchmark/runs/paired/n26_g__Ruminococcus__T20__l40s/gpu.treefile`
- model: `GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}` (true model known by construction)
- seed: 2026
- command: `iqtree3 --alisim aln -t "../../benchmark/runs/paired/n26_g__Ruminococcus__T20__l40s/gpu.treefile" -m "GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}" --length 8937615 -seed 2026`

Complexity = (2n-3) x patterns x 4 rate categories x 16 states^2. DNA patterns are capped by 4^n,
which is why paired DNA/protein comparisons are only possible for n >= 14.
