# examples/data — provenance and citation

**What these are:** deliberately tiny amino-acid alignments used by `examples/run_example.sh` so a new
user can run IQ-TREE3-GPU end to end in ~1-2 minutes, watch real output, and confirm the installation
works. **Several genera are provided** so you can try more than one tree.

| file | source genus | taxa | columns | reduced from | min occupancy |
|---|---|---|---|---|---|
| `example_CAG-269.faa` | CAG-269 | 8 | 3,997 | 10 taxa x 150,679 cols | 60% |
| `example_Nanosyncoccus.faa` | Nanosyncoccus | 8 | 3,374 | 20 taxa x 104,808 cols | 60% |
| `example_Lachnospira.faa` | Lachnospira | 8 | 3,620 | 25 taxa x 213,975 cols | 51% |

**Source (real, public data):** concatenated marker-gene alignments from this repository's benchmark
set (`benchmark/data/protein/<genus>/concat_alignment.faa`), produced by the upstream marker-gene pipeline (Prodigal gene
calling -> DIAMOND marker search -> MAFFT alignment -> concatenation) on metagenome-assembled genomes
(MAGs) classified with **GTDB-Tk** against the **GTDB** taxonomy and quality-controlled with
**CheckM2**. Tip labels record the host species and sample the genome came from, e.g.
`Alces_alces_W_Sv17_PPQH`.

**How they were reduced** (`scripts/dev/build_example_dataset.py`, fully deterministic): seeded taxon
subset (`random.Random(1).sample`) plus a contiguous column slice taken from the alignment midpoint,
after which all-gap columns are dropped.

**Validity:** the residues are unmodified real sequence data; only the number of taxa and columns is
reduced. These are *demonstration* datasets for learning the interface and verifying an installation.
The paper's benchmarks use the full-size alignments in `benchmark/data/protein/`.

**A note on the trees:** these are **bacterial** phylogenies whose tips happen to be labelled by host.
Do not expect tips to group by host species: gut symbiont clades may or may not track host relatedness,
and recent host switching or genuinely non-codiversifying lineages are biologically real. Whether
codiversification occurs is a *research question*, not an assumption — and 8 genomes could not settle
it either way. Use these runs to confirm the software works, not to draw biological conclusions.

**Citation:** please cite GTDB (Parks et al.), CheckM2 (Chklovski et al.), IQ-TREE 3
(Wong et al. 2026, https://doi.org/10.1093/molbev/msag117), and this work (Cho and Kostic 2026,
IQ-TREE3-GPU) — see the repository `README.md` Citation section and `assets/CITATION.cff`.
