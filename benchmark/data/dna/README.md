# Simulated DNA benchmark inputs

Every DNA benchmark row in `benchmark/runs/RUNS.tsv` was run on an alignment in this directory.
They are **simulated**, not primary data: each was generated with IQ-TREE's AliSim from a recorded
guide tree, substitution model and random seed, so the true model is known by construction.

## Where the alignments are

The 26 `aln.phy` files (2.4 GB; 11 exceed GitHub's 100 MB per-file limit) are on Zenodo,
doi:10.5281/zenodo.22942159, as `benchmark_data_dna.tar.gz`: `bash benchmark/scripts/fetch_benchmark_data.sh alignments`
puts them here. This directory ships the **recipe**, so every set can also be regenerated and verified.

Each set contains:

| File | In git? | Purpose |
|---|---|---|
| `PROVENANCE.md` | yes | taxa, sites, guide tree, exact model string, seed, and the full AliSim command |
| `aln.treefile` | yes | the guide tree itself — so `RANDOM{yh/N}` sets do not depend on reproducing IQ-TREE's RNG |
| `alisim.log` | no | the simulator's own log for the run that produced our alignment (in the Zenodo archive) |
| `aln.phy` | no | from Zenodo, or regenerate it, then verify against `CHECKSUMS.sha256` |

## Regenerating

All sets at once:

```bash
bash benchmark/scripts/dev/build_complexity_datasets.sh
```

Or one set, straight from its recipe — `PROVENANCE.md` carries the literal command, for example:

```bash
iqtree3 --alisim aln -t "RANDOM{yh/60}" \
        -m "GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}" \
        --length 6000000 -seed 3062
```

Then confirm you reproduced our exact input:

```bash
sha256sum -c benchmark/data/dna/CHECKSUMS.sha256
```

**Simulated with IQ-TREE version 3.1.3.** AliSim is deterministic given a seed, but only within a
version; a different IQ-TREE release may emit a different alignment from the same command. If a
checksum does not match, check your version first. Where a set's guide tree is one of our own
inferred trees (the `dna_pt_*` paired-alphabet sets), `PROVENANCE.md` points at the tracked
`benchmark/runs/paired/<cell>/gpu.treefile` by relative path, so nothing external is needed.

## The amino-acid alignments

The **real** amino-acid alignments under `../protein/` are primary data and are **not** regenerable;
the same Zenodo record holds them (`benchmark_data_protein.tar.gz`).
