# p_lc_n06_L20k - low-complexity protein benchmark input

Built by `scripts/dev/build_complexity_datasets.sh`. Real staged MAG data with
alignment COLUMNS subsampled so the protein arm reaches the low-complexity regime that the
taxa-only ladder never covered (every previous protein dataset used a full-length 99k-233k
site alignment, so even n=4 sat at 2.2e8 complexity).

- source: `data/protein/n06_g__CAG-269/concat_alignment.faa` (6 taxa, 143055 sites)
- columns kept: 20000 (seeded `random.Random` sample, original order preserved)
- sequences: unchanged real sequence data; only the number of columns is reduced
- both benchmark arms read this identical file, which is the only requirement for a
  performance comparison; the RF=0 equivalence gate applies exactly as elsewhere
