# Source genomes: provenance and the metadata table

Every amino-acid alignment in this repository is built from the 435 metagenome-assembled genomes
(MAGs) in this directory, staged as `g__<genus>/<stable_genome_id>.fna` under the GTDB genus they
were classified to. The tip label of every sequence in every alignment (`data/protein/*/`,
`examples/data/`) is the `stable_genome_id` of one of these files, so `genome_metadata.tsv`
resolves any tip to its study, host, public accessions and quality statistics.

## `genome_metadata.tsv`

One row per genome, 42 columns, tab-separated, exported from the curated metadata of the
source-genome collection the alignments were built from. `python3 scripts/verify_data.py` checks that the
table lists exactly the files on disk, that every file's md5 matches its row, that every alignment
tip is a listed genome, and that each full-genus alignment carries exactly the genomes staged for
that genus.

| column | meaning |
|---|---|
| `stable_genome_id` | the file name stem and alignment tip label: `<Host_species>_<lifestyle>_<paper>_<4-char id>` (see below) |
| `repo_path` | path of the `.fna` inside this repository |
| `n_alignments`, `alignments` | which alignment directories (and `examples/<file>`) carry this genome as a tip |
| `md5_fna` | md5 of the file |
| `dataset` | source study (`Carter_et_al_2023`, `Li_X_al_2025`, ...) |
| `genome_id`, `source_filename` | the genome's name and file as published by the study |
| `source_accession` | the deposit the file was taken from: a GenBank/ENA assembly (GCA) or analysis (ERZ), an NGDC Genome Warehouse accession (GWH), an ENA/SRA run (ERR/SRR), or an archive DOI (Zenodo, Dryad); per-study meaning in the table below |
| `source_sample` | the metagenome sample the MAG was binned from, as the study names it (ERS, SRR, or a study-internal sample id) |
| `source_bioproject` | NCBI/ENA/CNGB BioProject(s) of the underlying reads or assemblies |
| `source_biosample` | per-MAG or per-sample BioSample (SAMN/SAMEA/ERS/CNS) when the deposit has one |
| `genome_assembly_accession` | per-MAG assembly accession (GCA / ERZ / ENA WGS set / GWH) when one exists |
| `sequencing_platform`, `sequencing_instrument_model`, `assembly_software` | as reported by the study, when reported |
| `host_species`, `host_common_name_group`, `host_order`, `host_family`, `host_wild_status`, `anatomical_site`, `sample_material`, `sampling_context`, `environment_type`, `sample_collection_YYYYMMDD`, `collection_region`, `collection_latitude`, `collection_longitude` | host and sample metadata harmonized from the study's own tables; collection date `00100101` means wholly unknown, `20170101` and `20210101` are publication-year placeholders (Svartström, Glendinning) |
| `GTDB_tk_genus`, `GTDB_tk_species`, `GTDB_tk_taxonomy` | GTDB-Tk classification; `source_taxonomy` and `source_taxonomy_type` are the authors' own taxonomy verbatim and its origin |
| `checkm2_completeness`, `checkm2_contamination`, `checkm2_genome_size_bp`, `checkm2_N50_contig` | CheckM2 quality |
| `genome_type`, `lifestyle_code`, `paper_id`, `crockford_id` | MAG; the three components of the identifier |

### The identifier

`Homo_sapiens_H_Ca23_0AVS` = host species `Homo sapiens`, lifestyle code `H`, paper id `Ca23`
(Carter 2023), and a 4-character Crockford-base32 id derived from the md5 of the source file.
The lifestyle code is derived from `host_wild_status`; the codes present here are `W` = wild (356),
`H` = wild_hunter_gatherer (78), `A` = ancient_preindustrial_human (1). Paper ids: `Ca23`, `Li25`,
`Sa23`, `Bo21`, `Gl21`, `Sv17`, `Pu23`, `Wi21`.

## The eight source studies

Counts are genomes in this repository.

| study (`dataset`) | n | hosts here | paper | where the genome files come from | what the accession columns hold |
|---|---|---|---|---|---|
| Li_X_al_2025 | 160 | Pantholops hodgsonii (Tibetan antelope), Equus kiang (Tibetan wild ass) | Li X et al. 2025, Microbiome 13:236, doi:10.1186/s40168-025-02232-5 | NGDC Genome Warehouse, one GWH accession per SGB (`https://download.cncb.ac.cn/gwh/Metagenomes/`); reads and contigs under CNGB project CNP0001390 | `source_accession` = `genome_assembly_accession` = GWH accession; `source_sample` = SGB representative sample from Table S4; `source_bioproject` CNP0001390 and `source_biosample` CNS... for 32 genomes, blank for the other 128 |
| Sanders_et_al_2023 | 151 | Pan troglodytes, Pan paniscus (new Pan MAGs, 96); Papio cynocephalus, Papio anubis, Papio hamadryas, Alouatta pigra, Cercopithecus ascanius, Piliocolobus badius, Gorilla gorilla (non-Pan NHP MAGs, 55) | Sanders JG et al. 2023, Nature Microbiology 8:1039, doi:10.1038/s41564-023-01388-w | Pan MAGs: Dryad archive doi:10.5061/dryad.00000006x (`ChimpMAGs.tar`); non-Pan NHP MAGs: Sonnenburg-lab cohort assemblies (TungJ_2015, AmatoKR_2018, HicksAL_2018; Qiita study 11212) whose reads are ENA PRJEB35610 | Pan (96): `source_accession` = the Dryad DOI; `source_bioproject` = reads BioProject PRJNA842587 (plus PRJNA842693 for the 6 hybrid Nanopore assemblies); `source_sample` = WC sample id. Non-Pan (55): `source_accession` = `source_bioproject` = PRJEB35610 only, `source_sample` blank. No per-MAG assembly accession or BioSample exists for any Sanders MAG |
| Carter_et_al_2023 | 78 | Homo sapiens (69 Hadza hunter-gatherers, Tanzania; 9 Chepang hunter-gatherers, Nepal) | Carter MM et al. 2023, Nature 621:122, doi:10.1038/s41586-023-06370-2 | ENA, per-MAG assemblies (59 GCA accessions, 19 ERZ analysis accessions for the REFINED-named MAGs) under BioProjects PRJEB27517 (69, Hadza cohort) and PRJEB29137 (9, Nepal Chepang cohort) | `source_accession` = `genome_assembly_accession` = versioned GCA or ERZ; `source_sample` = ERS of the metagenome sample the MAG was binned from; `source_biosample` = per-MAG ERS assembly sample |
| Bowerman_et_al_2021 | 21 | Apodemus agrarius, Apodemus flavicollis, Microtus arvalis, Microtus agrestis, Microtus oeconomus, Myodes glareolus, Mus musculus | Bowerman KL et al. 2021, ISME Communications 1:49, doi:10.1038/s43705-021-00053-9 | Zenodo archive doi:10.5281/zenodo.5039598 (`rodent_microbiome_MAGs.tar.gz`) | `source_accession` = the Zenodo DOI (no per-MAG assembly accession exists); `source_sample` = SRR/ERR run of the sample for individually assembled MAGs, blank for the 3 co-assembled `SC*_comb_*` MAGs; `source_bioproject` = PRJNA725899 (16, Lithuanian wild rodents newly sequenced by the study), PRJNA540893 (4) or PRJEB32890 (1) for MAGs re-assembled from public metagenomes |
| Glendinning_et_al_2021 | 14 | Cervus elaphus (red deer), Rangifer tarandus (reindeer) | Glendinning L et al. 2021, Scientific Reports 11:1990, doi:10.1038/s41598-021-81668-9 | Edinburgh DataShare archive doi:10.7488/ds/2640 (`Rumen_RUGs.zip`); per-RUG ENA WGS sets under BioProject PRJEB34458 | `source_accession` = ENA run (ERR) of the per-animal assembly (13) or PRJEB34458 for the 1 RUG from the 12-sample co-assembly; `source_sample` = Red_deer_1..4 / Reindeer_1..2 / co_assembly_all_12; `genome_assembly_accession` = per-RUG ENA WGS-set accession (CAJO...); `source_biosample` = SAMEA of the animal (blank for the co-assembly RUG) |
| Svartstrom_et_al_2017 | 8 | Alces alces (moose) | Svartström O et al. 2017, ISME Journal 11:2538, doi:10.1038/ismej.2017.108 | ENA BioProject PRJEB12797: six per-sample GCA assemblies, each split back into its MAGs by contig name | `source_accession` = `genome_assembly_accession` = per-sample GCA; `source_sample` = sample 101B..106; `source_biosample` = SAMEA |
| Pu_et_al_2023 | 2 | Ochotona curzoniae (plateau pika) | Pu J et al. 2023, Microbiology Spectrum, doi:10.1128/spectrum.03495-22 | NCBI GenBank, one GCA assembly per SGB under BioProject PRJNA692773 | `source_accession` = `genome_assembly_accession` = GCA; `source_sample` = `source_biosample` = per-genome SAMN (the deposit gives no bin-to-metagenome-sample mapping) |
| Wibowo_et_al_2021 | 1 | Homo sapiens (ancient palaeofeces, Zape, Mexico, ~850 CE) | Wibowo MC et al. 2021, Nature 594:234, doi:10.1038/s41586-021-03532-0 | MetaGEM output archive on Zenodo doi:10.5281/zenodo.7414438 (`dna_bins`); reads only at BioProject PRJNA561510 | `source_accession` = SRR run of the sample; `source_sample` = archaeological sample name; `source_biosample` = per-sample SAMN; no per-MAG assembly accession exists |

Where a column is blank, the study's public deposit does not supply that identifier; nothing was
inferred to fill it.

## From genome to alignment

Each `data/protein/g__<genus>/concat_alignment.faa` was produced on exactly the genomes in
`g__<genus>/` here by a marker-gene pipeline: Prodigal gene calling, DIAMOND marker search, MAFFT
alignment, concatenation; the genomes were classified with GTDB-Tk and quality-controlled with
CheckM2 (`tree_qc.tsv` beside each alignment records the tree-building QC). The subsets
(`n??_g__*`, `1Thread_*`, `p_lc_*`, `examples/data/`) are seeded, deterministic taxon or column
subsamples of those alignments; each carries its own `PROVENANCE.md` naming its parent.

## Citing the data

Please cite the eight studies above for the genomes, GTDB (Parks et al.) and GTDB-Tk for the
taxonomy, and CheckM2 (Chklovski et al.) for the quality statistics.
