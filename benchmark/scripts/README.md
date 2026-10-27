# benchmark/scripts/ - the benchmark tier

The code that PRODUCED the record in `benchmark/runs/`. Running it needs a SLURM cluster with Lmod
and NVIDIA GPUs (the site settings are in `config/site.env`); reading it needs nothing. Presenting
the record is the job of `benchmark/figures_tables/`.

| path | role |
|---|---|
| `harness/` | the paired CPU-vs-GPU benchmark: one cell = both arms on one node from one binary, engagement-gated, one row of `RESULT.tsv` per cell; the submitters, the card sweep and the profiling runs ([`harness/README.md`](harness/README.md)) |
| `aggregate_runs.py` | publishes `runs/RUNS.tsv`, `RUNS.csv` and `RUNS_summary.tsv` from every `runs/paired/<cell>/RESULT.tsv`; never overwrites a successful measurement |
| `phase_extract.py` | parses both arms' IQ-TREE logs of every cell into `runs/PHASE_STEPS.tsv` (per-step wall and CPU time) |
| `verify_data.py` | proves `data/` is complete and intact against its checksums |
| `dev/` | the dataset builders that made `data/`, and the Zenodo packaging ([`dev/README.md`](dev/README.md)) |

To re-run the measurements: [`../REPRODUCING.md`](../REPRODUCING.md), tier 3.
