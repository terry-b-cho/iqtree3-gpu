# 1Thread_g__CAG-269_n6

Member of the **small, matched input set for the 1-thread comparison** (CPU-1T vs GPU-1T).
The 20-thread comparison (CPU-20T vs GPU-20T) uses the full-size counterpart
`data/protein/g__CAG-269/` (n = 10). Both sets contain **one dataset per genus for the same
14 genera**, so the two comparisons use the same genera and the same number of trees;
the only difference is input size.

**Why smaller for 1 thread:** a 1-thread run is ~19x slower in wall time (one core does
all the site-parallel work). On the full-size genera CPU-1T projects to 0.8-9+ days each,
and GPU-1T would exceed the 5-day wall cap of every available GPU partition - those cells
could never complete. Sizing this set to n = 6 keeps even CPU-1T to ~6-14 h, which
makes the 1T-vs-1T comparison obtainable while leaving the 20-thread comparison free to
explore the large genomes.

**Source (copied, never moved):** `data/protein/g__CAG-269/concat_alignment.faa`
(this repository's own staged copy of the upstream alignment:
Prodigal -> DIAMOND -> MAFFT -> concatenate).
**Construction:** seeded taxon subsample 10 -> 6 (random.Random(1).sample)
**Result:** n = 6 taxa, 150679 alignment columns.
**Validity:** real sequence data; only the number of tips is reduced. CPU and GPU receive
byte-identical input, which is what the performance comparison requires.
