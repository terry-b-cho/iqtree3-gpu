# scripts/dev/ - maintainer utilities for the tool

None of these are needed to build, verify or use the tool, and none feed the published numbers.
The builders of the benchmark datasets and the Zenodo packaging live with the benchmark, in
[`benchmark/scripts/dev/`](../../benchmark/scripts/dev/README.md).

| script | role |
|---|---|
| `build_example_dataset.py` | builds the small alignments in `examples/data/` from the benchmark alignments (provenance in `examples/data/PROVENANCE.md`) |
| `regen_expected_output.sh` | regenerates `examples/expected_output/` with the current production binary |
| `plot_trees.R` | draws the tree plot that `examples/run_example.sh` shows (optional; needs R with ape) |
| `rebuild_gpu.sh` | incremental rebuild of the GPU binary for development |
| `sync_src.sh` | syncs the working source tree into the shippable `src/` (patch + new files) |
| `make_author_cards.py` | regenerates the author cards in `assets/authors/` used by the top-level README (needs network access to fetch GitHub avatars) |
