# p_lc_n04_L200 - low-complexity protein benchmark input

Built by `scripts/dev/build_complexity_datasets.sh`. Real staged MAG data with
alignment COLUMNS subsampled so the protein arm reaches the low-complexity regime that the
taxa-only ladder never covered (every previous protein dataset used a full-length 99k-233k
site alignment, so even n=4 sat at 2.2e8 complexity).

- source: `data/protein/n04_g__Brotaphodocola/concat_alignment.faa` (4 taxa, 195456 sites)
- columns kept: 200 (seeded `random.Random` sample, original order preserved)
- sequences: unchanged real sequence data; only the number of columns is reduced
- both benchmark arms read this identical file, which is the only requirement for a
  performance comparison; the RF=0 equivalence gate applies exactly as elsewhere
