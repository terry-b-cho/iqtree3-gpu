# Simulated DNA benchmark inputs

Every DNA benchmark row in `results/benchmarks/RUNS.tsv` was run on an alignment in this directory.
They are **simulated**, not primary data: each was generated with IQ-TREE's AliSim from a recorded
guide tree, substitution model and random seed, so the true model is known by construction.

## Why the large alignments are not in the repository

Twelve smaller sets ship in full. The remaining **14 alignments total 2.4 GB, and 11 of them exceed
100 MB — GitHub's hard per-file limit.** They cannot be pushed to GitHub at all, with or without
Git LFS. Rather than hide that behind an external download, this directory ships the **recipe**,
which is strictly better than shipping the artifact: you can regenerate every byte and verify it.

Each set therefore contains:

| File | Ships? | Purpose |
|---|---|---|
| `PROVENANCE.md` | yes | taxa, sites, guide tree, exact model string, seed, and the full AliSim command |
| `aln.treefile` | yes | the guide tree itself — so `RANDOM{yh/N}` sets do not depend on reproducing IQ-TREE's RNG |
| `alisim.log` | yes | the simulator's own log for the run that produced our alignment |
| `aln.phy` | **no** (>100 MB) | regenerate it, then verify against `CHECKSUMS.sha256` |

## Regenerating

All sets at once:

```bash
bash scripts/dev/build_complexity_datasets.sh
```

Or one set, straight from its recipe — `PROVENANCE.md` carries the literal command, for example:

```bash
iqtree3 --alisim aln -t "RANDOM{yh/60}" \
        -m "GTR{1.0,2.5,1.2,0.9,2.8}+F{0.28,0.22,0.22,0.28}+G4{0.7}" \
        --length 6000000 -seed 3062
```

Then confirm you reproduced our exact input:

```bash
sha256sum -c data/dna/CHECKSUMS.sha256
```

**Simulated with IQ-TREE version 3.1.3.** AliSim is deterministic given a seed, but only within a
version; a different IQ-TREE release may emit a different alignment from the same command. If a
checksum does not match, check your version first. Where a set's guide tree is one of our own
inferred trees (the `dna_pt_*` paired-alphabet sets), `PROVENANCE.md` points at the tracked
`results/benchmarks/paired/<cell>/gpu.treefile` by relative path, so nothing external is needed.

## What ships in full

The **real** amino-acid alignments under `../protein/` are primary data and are **not** regenerable,
so all 35 used by the benchmark are tracked in full (~102 MB). Those are the inputs behind every
protein row, and behind the headline speedup results.
