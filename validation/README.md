# Validation output

**This directory is where the validation harnesses write YOUR results.** It ships essentially empty
on purpose.

| Harness | Writes here | What it proves |
|---|---|---|
| `scripts/equivalence_matrix.sh` | `equivalence/<genus>/`, `equivalence_matrix.tsv` | GPU and CPU agree (RF = 0, logL within 1e-4) across models x thread counts |
| `scripts/differential.sh` | `differential/<genus>/` | Per-site and per-rate-category log-likelihoods agree on a fixed topology |

Run one with, for example:

```bash
GENUS=n14_g__Nanosyncoccus sbatch scripts/equivalence_matrix.sh
```

`GENUS` names a directory under `benchmark/data/protein/`; the harnesses read the benchmark
alignments because they are the inputs every recorded run used, so a verdict here is comparable
with the published record. Any other alignment works by setting `GENUS` to a directory that holds a
`concat_alignment.faa`.

## Where our own records are

The equivalence-matrix evidence, arm-symmetry check and on-device Newton validation records produced
during development are team-internal and are not distributed. They are historical records of our
runs; this directory is for yours. Nothing in `benchmark/figures/` or
`benchmark/runs/` depends on either — those regenerate from `benchmark/runs/RUNS.tsv` alone.
