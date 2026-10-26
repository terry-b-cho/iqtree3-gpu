# scripts/dev/ - development history (one-offs, dataset builders, superseded tools)

Kept for transparency: these scripts built the datasets, diagnosed and fixed engine bugs, or were
superseded. None are needed to build, verify, or use the tool, and none feed the published numbers.

- Dataset builders (provenance of `data/protein/` and `data/dna/`): `build_capped_genomes.py`,
  `build_dna_datasets.sh`, `build_example_dataset.py`, `build_1thread_genomes.py`,
  `dna_cpu_reference.sh`
- One-off diagnostics from the DNA NNI-crash investigation (resolved; replay cache off by default):
  `diagnose_nni_crash.sh`, `rebuild_verify_sigfix.sh`
- On-device-Newton bring-up validators (the path is now in the production binary):
  `build_graphs.sh`, `measure_launches.sh`, `validate_graphs.sh`,
  `validate_newton_replay.sh`, `validate_ondevice_newton.sh`, `submit_dna_ondevice.sh`
- Superseded / maintainer utilities: `rebuild_gpu.sh`, `plot_trees.R` (replaced by
  `../trees.py`), `sync_src.sh` (syncs the working source tree into shippable `src/`)
- Release packaging and presentation: `build_zenodo_bundle.sh`, `build_zenodo_record.sh`,
  `zenodo_upload.py` (the Zenodo deposit), `make_author_cards.py` (regenerates the author cards in
  `assets/authors/` used by the top-level README; needs network access to fetch GitHub avatars)
