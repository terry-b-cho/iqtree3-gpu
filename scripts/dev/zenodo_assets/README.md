# IQ-TREE3-GPU — large benchmark inputs

Companion data for the IQ-TREE3-GPU repository. **Download this folder whole, run the installer, and
every benchmark in the paper becomes reproducible on your own machine.**

These 14 simulated DNA alignments total ~2.4 GB and 11 of them exceed 100 MB, which is GitHub's hard
per-file limit — they cannot live in the repository. Everything else you need (the tool, all scripts,
the real amino-acid alignments, the full measured record, and 12 smaller DNA alignments) ships in the
repository itself.

## Install

```bash
# from inside this folder, pointing at your clone
bash install_into_repo.sh /path/to/iqtree-gpu
```

The installer copies each `aln.phy` into the matching `data/dna/<set>/` directory of your clone
and then verifies every file against `CHECKSUMS.sha256`. It refuses to overwrite a file that is
already present and correct.

Or do it by hand:

```bash
rsync -av data/dna/ /path/to/iqtree-gpu/data/dna/
cd /path/to/iqtree-gpu && sha256sum -c data/dna/CHECKSUMS.sha256
```

## What is here

| | |
|---|---|
| `data/dna/<set>/aln.phy` | the alignment (the part too large for GitHub) |
| `data/dna/<set>/PROVENANCE.md` | taxa, sites, guide tree, exact model string, seed, full AliSim command |
| `data/dna/<set>/aln.treefile` | the guide tree, where the set used one |
| `CHECKSUMS.sha256` | SHA256 for every alignment, including the 12 that ship in the repo |

Three families: `dna_hc_*` (high-complexity ladder), `dna_pt_*` (DNA simulated on our own inferred
protein topologies, for the paired-alphabet test) and `dna_ux_*` (beyond the taxa-only ceiling).

## You may not need this at all

Every alignment here is **simulated and regenerable**. Each `PROVENANCE.md` carries the literal
command, and `scripts/dev/build_complexity_datasets.sh` in the repository rebuilds all of them.
Simulated with **IQ-TREE 3.1.3**; AliSim is deterministic given a seed, but only within a version.
This archive exists so you do not have to spend the CPU time, and so that a checksum mismatch after
regeneration can be diagnosed against a known-good copy.

## Citation

Cite the IQ-TREE3-GPU paper and the Zenodo DOI for this deposit. IQ-TREE itself must also be cited —
see `CITATION.cff` in the repository.
