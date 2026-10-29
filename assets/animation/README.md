# The README animation

`iqtree3-gpu.svg` (and `iqtree3-gpu.png`, a still) is the animated overview at the top of the repository README.
It shows eight bacterial genomes from the worked example being aligned and becoming a candidate tree; Felsenstein
pruning computing conditional likelihoods from the tips to the root; the same calculation running on CPU IQ-TREE 3
and on IQ-TREE3-GPU for the largest benchmark case; and one nearest-neighbour interchange reaching the
maximum-likelihood tree, shown with its bootstrap support.

**What each stage uses.** Stages 1, 2 and 4 use the example genome set shipped in `examples/` (8 CAG-269 genomes, LG+G4), with IQ-TREE's own conditional likelihoods and log-likelihoods. Stage 3 is the largest analysis in the benchmark (65 *Colicola* genomes, 236,088 sites, ModelFinder, 20 threads, L40S): 17.6 h on the CPU, 3.4 h on the GPU, 5.12×, same tree. Benchmark medians: 3.73× for amino acid + ModelFinder at 20 threads (n = 67); over all datasets 1.33× at 20 threads (n = 145) and 6.97× at 1 thread (n = 30).

Every number is from `benchmark/runs/KEY_NUMBERS.md`, the generated file that is authoritative over anything typed here.
