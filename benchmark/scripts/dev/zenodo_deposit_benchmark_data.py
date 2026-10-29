#!/usr/bin/env python3
"""Stage the IQ-TREE3-GPU benchmark data, package it, and upload it to the existing Zenodo draft.

    python3 benchmark/scripts/dev/zenodo_deposit_benchmark_data.py [--stage-only] [--scan PATTERN,...]

Staging goes to the gitignored zenodo_upload_bundle/ at the repository root. --scan lists strings
whose presence in staged files is counted and reported (nothing is altered); default: none.

Deposition 22942159 already exists as an unsubmitted draft with the reserved DOI 10.5281/zenodo.22942159,
which the manuscript cites. This script REPLACES its metadata and uploads the archives. It never publishes.
Token: $ZENODO_TOKEN or ~/.zenodo_token.
"""
import hashlib, json, os, re, shutil, subprocess, sys, datetime, urllib.request, urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VER = "0.1.0"
DEP_ID = 22942159
API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN") or open(os.path.expanduser("~/.zenodo_token")).read().strip()
BUNDLE = os.path.join(REPO, "zenodo_upload_bundle")
B = os.path.join(BUNDLE, f"iqtree3-gpu-benchmark-data-v{VER}")      # unpacked staging tree
A = os.path.join(BUNDLE, f"archives_data_v{VER}")                     # what is uploaded
STATE = os.path.join(BUNDLE, "DEPOSIT_benchmark_data.json")
os.chdir(REPO)

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, **kw).stdout

def ls_files(path):
    return [p for p in sh(f"git ls-files -z -- '{path}'").split("\0") if p]

def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst): os.remove(dst)
    try: os.link(src, dst)
    except OSError: shutil.copy2(src, dst)

# ------------------------------------------------------------------ 1. stage
print("== staging ==", flush=True)
shutil.rmtree(B, ignore_errors=True); os.makedirs(B)
tracked = []
for d in ("benchmark/data/genomes", "benchmark/data/protein", "benchmark/runs/paired"):
    tracked += ls_files(d)
# the data payloads are deliberately untracked (gitignored; they live on Zenodo), so take them from disk
for pat in ("benchmark/data/genomes/g__*/*.fna", "benchmark/data/protein/*/concat_alignment.faa"):
    import glob as _glob
    tracked += sorted(os.path.relpath(p, REPO) for p in _glob.glob(os.path.join(REPO, pat)))
tracked += ["benchmark/runs/RUNS.tsv", "benchmark/runs/RUNS.csv", "benchmark/runs/RUNS_summary.tsv",
            "benchmark/runs/RUNS_summary.csv", "benchmark/runs/PHASE_STEPS.tsv", "benchmark/runs/REPLICATES_20T.tsv",
            "benchmark/runs/KEY_NUMBERS.tsv", "benchmark/runs/KEY_NUMBERS.md", "benchmark/BENCHMARK_SCHEMA.md"]
tracked += ls_files("benchmark/runs/dna_cpu_reference")
for p in tracked:
    link(os.path.join(REPO, p), os.path.join(B, p))
# nucleotide sets: from disk, so the 14 large aln.phy files (not in git) are included
n_dna = 0
for root, dirs, files in os.walk(os.path.join(REPO, "benchmark/data/dna")):
    for f in files:
        src = os.path.join(root, f); rel = os.path.relpath(src, REPO)
        link(src, os.path.join(B, rel)); n_dna += 1
print(f"   tracked files staged: {len(tracked)}; nucleotide files from disk: {n_dna}", flush=True)


def deposit_tree():
    """Text tree of every archive's contents, with counts where directories repeat a pattern."""
    L = []
    def size(name): 
        p = os.path.join(A, name); return f"{os.path.getsize(p)/1e6:.0f} MB" if os.path.exists(p) else ""
    def files_in(d): return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    def dirs_in(d): return sorted(f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f)))
    L.append(f"doi:10.5281/zenodo.{DEP_ID}/")
    # genomes
    g = os.path.join(B, "benchmark/data/genomes")
    L.append(f"|-- benchmark_data_genomes.tar.gz   ({size('benchmark_data_genomes.tar.gz')})  unpacks to benchmark/data/genomes/")
    for f in files_in(g): L.append(f"|   |-- {f}")
    gd = dirs_in(g)
    for i, d in enumerate(gd):
        n = len(files_in(os.path.join(g, d)))
        L.append(f"|   {'`--' if i == len(gd)-1 else '|--'} {d}/   ({n} genomes, <Host_species>_<lifestyle>_<study>_<id>.fna)")
    # protein
    pdir = os.path.join(B, "benchmark/data/protein")
    L.append(f"|-- benchmark_data_protein.tar.gz   ({size('benchmark_data_protein.tar.gz')})  unpacks to benchmark/data/protein/")
    for f in files_in(pdir): L.append(f"|   |-- {f}")
    pd = dirs_in(pdir)
    for i, d in enumerate(pd):
        fl = ", ".join(files_in(os.path.join(pdir, d)))
        L.append(f"|   {'`--' if i == len(pd)-1 else '|--'} {d}/   ({fl})")
    # dna
    ddir = os.path.join(B, "benchmark/data/dna")
    L.append(f"|-- benchmark_data_dna.tar.gz   ({size('benchmark_data_dna.tar.gz')})  unpacks to benchmark/data/dna/")
    for f in files_in(ddir): L.append(f"|   |-- {f}")
    dd = dirs_in(ddir)
    for i, d in enumerate(dd):
        fl = ", ".join(files_in(os.path.join(ddir, d)))
        L.append(f"|   {'`--' if i == len(dd)-1 else '|--'} {d}/   ({fl})")
    # runs
    r = os.path.join(B, "benchmark/runs")
    L.append(f"|-- benchmark_runs.tar.gz   ({size('benchmark_runs.tar.gz')})  unpacks to benchmark/runs/")
    for f in files_in(r): L.append(f"|   |-- {f}")
    for f in files_in(os.path.join(r, "dna_cpu_reference")): L.append(f"|   |-- dna_cpu_reference/{f}")
    cells = dirs_in(os.path.join(r, "paired"))
    rep = files_in(os.path.join(r, "paired", cells[0]))
    L.append(f"|   `-- paired/   ({len(cells)} run directories named <dataset>__T<threads>__<card namespace>/, each holding:)")
    for j, f in enumerate(rep):
        L.append(f"|       {'`--' if j == len(rep)-1 else '|--'} {f}")
    # supporting
    L.append(f"`-- supporting_files.tar.gz   unpacks to supporting_files/")
    sup = ["README.md  (this description, with installation steps and full citations)",
           "install_into_repo.sh  (unpacks the four archives into a clone of the repository)",
           "MANIFEST.sha256  (SHA-256 of every file in the unpacked tree)",
           "ARCHIVES.sha256  (SHA-256 of the four archives)",
           "BENCHMARK_SCHEMA.md  (column dictionary of RUNS.tsv)",
           "zenodo_metadata.json  (the metadata of this record)",
           "zenodo_deposit_benchmark_data.py  (the script that staged and uploaded this record)"]
    for j, f in enumerate(sup): L.append(f"    {'`--' if j == len(sup)-1 else '|--'} {f}")
    return "\n".join(L)

# ------------------------------------------------------------------ 2. README and installer
import csv as _csv
def _studies():
    """(label, n genomes, citation, paper DOI, deposit text, deposit DOI) from the repository tables."""
    cites = {r["dataset"]: r for r in _csv.DictReader(open(os.path.join(REPO, "benchmark/data/genomes/study_citations.tsv")), delimiter="\t")}
    n = {}
    for r in _csv.DictReader(open(os.path.join(REPO, "benchmark/data/genomes/genome_metadata.tsv")), delimiter="\t"):
        n[r["dataset"]] = n.get(r["dataset"], 0) + 1
    out = []
    for ds, r in sorted(cites.items(), key=lambda kv: -n.get(kv[0], 0)):
        m = re.search(r"doi:(10\.\d{4,9}/\S+)", r["genome_deposit"])
        label = re.sub(r"_et_al_|_X_al_", " et al. ", ds).replace("_", " ")
        out.append((label, n.get(ds, 0), r["citation"], r["doi"], r["genome_deposit"], m.group(1).rstrip(")") if m else None))
    return out
STUDIES = _studies()
TOOLS = [
 ("IQ-TREE 3", "Wong TKF et al. 2026. IQ-TREE 3: phylogenomic inference software using complex evolutionary models. Molecular Biology and Evolution 43(5):msag117.", "10.1093/molbev/msag117"),
 ("ModelFinder", "Kalyaanamoorthy S et al. 2017. ModelFinder: fast model selection for accurate phylogenetic estimates. Nature Methods 14:587-589.", "10.1038/nmeth.4285"),
 ("UFBoot2", "Hoang DT et al. 2018. UFBoot2: improving the ultrafast bootstrap approximation. Molecular Biology and Evolution 35:518-522.", "10.1093/molbev/msx281"),
 ("AliSim", "Ly-Trong N et al. 2022. AliSim: a fast and versatile phylogenetic sequence simulator for the genomic era. Molecular Biology and Evolution 39:msac092.", "10.1093/molbev/msac092"),
 ("AliSim-HPC", "Ly-Trong N et al. 2023. AliSim-HPC: parallel sequence simulator for phylogenetics. Bioinformatics 39:btad540.", "10.1093/bioinformatics/btad540"),
 ("GTDB-Tk", "Chaumeil P-A et al. 2022. GTDB-Tk v2: memory friendly classification with the genome taxonomy database. Bioinformatics 38:5315-5316.", "10.1093/bioinformatics/btac672"),
 ("GTDB", "Parks DH et al. 2022. GTDB: an ongoing census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy. Nucleic Acids Research 50:D785-D794.", "10.1093/nar/gkab776"),
 ("CheckM2", "Chklovski A et al. 2023. CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning. Nature Methods 20:1203-1212.", "10.1038/s41592-023-01940-w"),
 ("Prodigal", "Hyatt D et al. 2010. Prodigal: prokaryotic gene recognition and translation initiation site identification. BMC Bioinformatics 11:119.", "10.1186/1471-2105-11-119"),
 ("PhyloPhlAn 3", "Asnicar F et al. 2020. Precise phylogenetic analysis of microbial isolates and genomes from metagenomes using PhyloPhlAn 3.0. Nature Communications 11:2500.", "10.1038/s41467-020-16366-7"),
 ("DIAMOND", "Buchfink B et al. 2021. Sensitive protein alignments at tree-of-life scale using DIAMOND. Nature Methods 18:366-368.", "10.1038/s41592-021-01101-x"),
 ("MAFFT", "Katoh K, Standley DM. 2013. MAFFT multiple sequence alignment software version 7: improvements in performance and usability. Molecular Biology and Evolution 30:772-780.", "10.1093/molbev/mst010"),
]
DOI = f"10.5281/zenodo.{DEP_ID}"
study_rows = "\n".join(f"| {s[0]} | {s[1]} | {s[2]}, doi:{s[3]} | {s[4]} |" for s in STUDIES)
tool_rows = "\n".join(f"- {t[1]} doi:{t[2]}" for t in TOOLS)

README = f"""# IQ-TREE3-GPU benchmark data: source genomes, alignments and the paired CPU-GPU run record

Companion data deposit for the manuscript

> Cho B, Kostic AD. **IQ-TREE3-GPU: GPU-accelerated maximum-likelihood phylogenetic tree inference for
> complex amino acid substitution models.**
> Byeongyeon Cho (1, 2, 3), Aleksandar D. Kostic (2, 3, corresponding author, Aleksandar.Kostic@joslin.harvard.edu).
> 1 Department of Biomedical Informatics, Harvard Medical School, Boston, MA, USA. 2 Department of Microbiology
> and Immunobiology, Harvard Medical School, Boston, MA, USA. 3 Section on Pathophysiology and Molecular
> Pharmacology, Joslin Diabetes Center, Boston, MA, USA.

and for the IQ-TREE3-GPU software repository: https://github.com/kosticlab/iqtree3-gpu
(mirror: https://github.com/terry-b-cho/iqtree3-gpu). The repository holds the tool, the benchmark harness,
the analysis and figure code and the documentation; this deposit holds the **input data and the measured
record** of the benchmark, so that the published numbers can be reproduced without Git LFS and so that the
processed datasets (source genomes, marker-gene alignments, simulated alignments) can be reused directly.

DOI of this deposit: https://doi.org/{DOI}

## Contents

| Archive | Unpacks to | What it holds |
|---|---|---|
| `benchmark_data_genomes.tar.gz` | `benchmark/data/genomes/` | 435 bacterial metagenome-assembled genomes (MAGs) as FASTA, one directory per GTDB genus (14 genera), with `genome_metadata.tsv` (one row per genome: study, public accessions, host, GTDB-Tk taxonomy, CheckM2 quality, md5) and `study_citations.tsv` |
| `benchmark_data_protein.tar.gz` | `benchmark/data/protein/` | 45 concatenated marker-gene amino-acid alignments (FASTA) with a `PROVENANCE.md` beside each: the 14 full-genus alignments, the 12-dataset taxon series (4-30 genomes), 14 six-genome subsets and 5 column-subsampled sets; 35 of them are the benchmark datasets of the manuscript |
| `benchmark_data_dna.tar.gz` | `benchmark/data/dna/` | 26 nucleotide alignments simulated with AliSim (PHYLIP), each with its guide tree, simulation log and, where present, `PROVENANCE.md` with the exact command and seed; `CHECKSUMS.sha256`; `TRUE_MODEL.txt`. Fourteen of these exceed GitHub's per-file limit and are available only here or by regeneration |
| `benchmark_runs.tar.gz` | `benchmark/runs/` | the measured record: `RUNS.tsv` (one row per paired CPU-versus-GPU run), its summary, `PHASE_STEPS.tsv`, `REPLICATES_20T.tsv`, `KEY_NUMBERS.tsv/.md`, the CPU reference runs for the nucleotide sets, and every paired-run directory under `runs/paired/` with both arms' IQ-TREE log, standard output, maximum-likelihood tree, consensus tree with branch supports, `.iqtree` report, exit code and timing files, `RESULT.tsv`, `PHASES.tsv` and `equivalence.kv` |
| `supporting_files.tar.gz` | `supporting_files/` | this README, `install_into_repo.sh` (unpacks the four archives into a clone at the paths above), `MANIFEST.sha256` (every file of the unpacked tree), `ARCHIVES.sha256`, `BENCHMARK_SCHEMA.md` (column dictionary of `RUNS.tsv`), the record metadata and the script that produced this deposit |

Archive names are the repository paths with `/` replaced by `_`, so `benchmark_data_genomes.tar.gz` is the folder `benchmark/data/genomes/` of https://github.com/kosticlab/iqtree3-gpu.

## Tree of this deposit

```
__TREE__
```

## Installing into a clone of the repository

```bash
git clone https://github.com/kosticlab/iqtree3-gpu.git          # or the mirror
tar -xzf supporting_files.tar.gz && bash supporting_files/install_into_repo.sh /path/to/iqtree3-gpu   # run beside the archives
cd /path/to/iqtree3-gpu && python3 benchmark/scripts/verify_data.py   # checks every genome md5, alignment and nucleotide checksum
```

Or unpack by hand: every archive extracts relative to the repository root. With the data in place,
`benchmark/REPRODUCING.md` in the repository describes the three tiers of reproduction, from regenerating every
figure and table from `RUNS.tsv` (no GPU needed) to re-running the paired benchmark.

## The source genomes and who to credit for them

The 435 genomes are copies of MAG assemblies deposited publicly by eight studies, renamed to a stable identifier
of the form `<Host_species>_<lifestyle>_<study>_<id>` (for example `Homo_sapiens_H_Ca23_60AF`: a bacterial genome
recovered from a Homo sapiens sample in Carter et al. 2023, with the four-character id derived from the file's
md5). `genome_metadata.tsv` maps every file back to its study, its public accessions (BioProject, BioSample,
assembly accession, run or archive DOI, as each deposit provides them) and its host; `study_citations.tsv` lists
the eight papers. **Anyone using these genomes should cite the study that produced them**, not only this deposit.
The genomes remain subject to the terms of the archives they were obtained from.

| Study | Genomes here | Citation | Where the genomes were obtained |
|---|---|---|---|
{study_rows}

The genomes were classified with GTDB-Tk against GTDB release 232 and quality-assessed with CheckM2 (completeness
90.05-100%, contamination 0-4.98%); the taxonomy and quality values are columns of `genome_metadata.tsv`.
The amino-acid alignments were built from these genomes with Prodigal gene prediction, DIAMOND assignment of
predicted proteins to the PhyloPhlAn 3 marker database, per-marker alignment with MAFFT and concatenation within
each genus; the tip label of every sequence is the genome identifier above. The nucleotide alignments are
simulations (AliSim in IQ-TREE 3) and contain no biological sequence; seven of them were simulated along trees
inferred from the amino-acid alignments and carry those genome labels as taxon names.

## The run record

`RUNS.tsv` has one row per paired run: both arms were run on the same compute node from the same executable with
the same alignment, model, random seed and thread count, differing only in the `-gpu --gpu-strict` flags of the
GPU arm. `runs/paired/` holds 226 run directories; the 182 runs analysed in the manuscript are the rows whose
`binary_sha256` begins with `edf783c3`, which is how every figure and table in the repository selects them.
`BENCHMARK_SCHEMA.md` defines every column, including the equivalence criteria (`rf_dist`, `logl_reldiff`,
`support_max_absdiff`, `gate_pass`) and the GPU engagement counters.

## How to cite

This deposit: Cho B, Kostic AD. IQ-TREE3-GPU benchmark data: source genomes, alignments and the paired CPU-GPU
run record. Zenodo. https://doi.org/{DOI}

The software and manuscript: Cho B, Kostic AD. IQ-TREE3-GPU: GPU-accelerated maximum-likelihood phylogenetic
tree inference for complex amino acid substitution models. https://github.com/kosticlab/iqtree3-gpu

The genomes: the study that produced each genome (table above).

The software the data were produced with:
{tool_rows}

## Licence

The compilation (this deposit: the metadata tables, alignments, simulated data, run record and documentation) is
released under CC BY 4.0. The genome sequences in `01_source_genomes` were produced by the studies cited above and
remain subject to the terms under which those studies deposited them; cite them when you use the genomes.
"""
INSTALLER = f"""#!/usr/bin/env bash
# Unpack the IQ-TREE3-GPU benchmark data archives into a clone of https://github.com/kosticlab/iqtree3-gpu.
#   bash install_into_repo.sh /path/to/iqtree3-gpu        (run from the folder that holds the archives)
set -euo pipefail
CLONE="${{1:?usage: install_into_repo.sh /path/to/iqtree3-gpu}}"
[ -f "$CLONE/benchmark/BENCHMARK_SCHEMA.md" ] || {{ echo "not an iqtree3-gpu clone: $CLONE" >&2; exit 1; }}
SUP="$(cd "$(dirname "$0")" && pwd)"; HERE="$(dirname "$SUP")"
[ -f "$SUP/ARCHIVES.sha256" ] && (cd "$HERE" && sha256sum -c "$SUP/ARCHIVES.sha256")
for a in "$HERE"/benchmark_data_genomes.tar.gz "$HERE"/benchmark_data_protein.tar.gz "$HERE"/benchmark_data_dna.tar.gz "$HERE"/benchmark_runs.tar.gz; do
  echo "unpacking $(basename "$a")"
  tar -xzf "$a" -C "$CLONE"
done
echo "done; verify with: cd $CLONE && python3 benchmark/scripts/verify_data.py"
"""
open(os.path.join(B, "README.md"), "w").write(README.replace("__TREE__", "(tree inserted after packaging)"))
open(os.path.join(B, "install_into_repo.sh"), "w").write(INSTALLER); os.chmod(os.path.join(B, "install_into_repo.sh"), 0o755)

# ------------------------------------------------------------------ 3. manifest + archives
print("== manifest ==", flush=True)
sh(f"cd '{B}' && find . -type f ! -name MANIFEST.sha256 ! -name README.md ! -name install_into_repo.sh -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256")
nfiles = int(sh(f"wc -l < '{B}/MANIFEST.sha256'"))
print(f"   {nfiles} files in the unpacked tree", flush=True)
print("== scan (files containing each --scan pattern; reported, not altered) ==", flush=True)
_scan = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--scan=")]
for pat in (_scan[0].split(",") if _scan else []):
    n = sh(f"cd '{B}' && grep -rlI --exclude=MANIFEST.sha256 -- '{pat}' . | wc -l").strip()
    print(f"   {pat}: {n} files", flush=True)

shutil.rmtree(A, ignore_errors=True); os.makedirs(A)
ARCHIVES = [("benchmark_data_genomes", ["benchmark/data/genomes"]),
            ("benchmark_data_protein", ["benchmark/data/protein"]),
            ("benchmark_data_dna", ["benchmark/data/dna"]),
            ("benchmark_runs", ["benchmark/runs"])]
print("== archives ==", flush=True)
for name, paths in ARCHIVES:
    out = os.path.join(A, f"{name}.tar.gz")
    sh(f"cd '{B}' && tar -cf - {' '.join(paths)} | pigz -p 8 > '{out}'")
    print(f"   {os.path.basename(out)}: {os.path.getsize(out)/1e6:.0f} MB", flush=True)
TREE = deposit_tree()
open(os.path.join(B, "README.md"), "w").write(README.replace("__TREE__", TREE))
SUP = os.path.join(A, "supporting_files"); os.makedirs(SUP)
sh(f"cd '{A}' && sha256sum *.tar.gz > '{SUP}/ARCHIVES.sha256'")
for f in ("README.md", "install_into_repo.sh", "MANIFEST.sha256"):
    shutil.copy2(os.path.join(B, f), os.path.join(SUP, f))
shutil.copy2(os.path.join(REPO, "benchmark/BENCHMARK_SCHEMA.md"), os.path.join(SUP, "BENCHMARK_SCHEMA.md"))
shutil.copy2(__file__, os.path.join(SUP, "zenodo_deposit_benchmark_data.py"))

# ------------------------------------------------------------------ 4. metadata
def api(method, path, data=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo API {method} {path} -> HTTP {e.code}: {e.read().decode()[:800]}")

AFF_CHO = ("Department of Biomedical Informatics, Harvard Medical School, Boston, MA, USA; Department of Microbiology and "
           "Immunobiology, Harvard Medical School, Boston, MA, USA; Section on Pathophysiology and Molecular Pharmacology, "
           "Joslin Diabetes Center, Boston, MA, USA")
AFF_KOS = ("Department of Microbiology and Immunobiology, Harvard Medical School, Boston, MA, USA; Section on "
           "Pathophysiology and Molecular Pharmacology, Joslin Diabetes Center, Boston, MA, USA")
related = [{"identifier": "https://github.com/kosticlab/iqtree3-gpu", "relation": "isSupplementTo", "resource_type": "software"},
           {"identifier": "https://github.com/terry-b-cho/iqtree3-gpu", "relation": "isSupplementTo", "resource_type": "software"}]
for s in STUDIES:
    related.append({"identifier": s[3], "relation": "isDerivedFrom", "resource_type": "publication-article"})
    if s[5]:
        related.append({"identifier": s[5], "relation": "isDerivedFrom", "resource_type": "dataset"})
for t in TOOLS:
    related.append({"identifier": t[2], "relation": "cites", "resource_type": "publication-article"})

desc = ("<p>Input data and measured record of the benchmark in the manuscript <i>IQ-TREE3-GPU: GPU-accelerated "
        "maximum-likelihood phylogenetic tree inference for complex amino acid substitution models</i> (Byeongyeon Cho and "
        "Aleksandar D. Kostic; corresponding author Aleksandar D. Kostic, Aleksandar.Kostic@joslin.harvard.edu). Companion to "
        "the IQ-TREE3-GPU software repository, https://github.com/kosticlab/iqtree3-gpu (mirror "
        "https://github.com/terry-b-cho/iqtree3-gpu), which holds the tool, the benchmark harness, the analysis and figure "
        "code and the documentation. This deposit lets the published results be reproduced without Git LFS and makes the "
        "processed datasets available for reuse.</p>"
        "<p><b>Contents.</b> benchmark_data_genomes: 435 publicly deposited bacterial metagenome-assembled genomes (14 GTDB genera, eight studies) "
        "with a per-genome metadata table (study, public accessions, host, GTDB-Tk taxonomy, CheckM2 quality). "
        "benchmark_data_protein: 45 concatenated marker-gene amino-acid alignments built from those genomes (Prodigal, DIAMOND against the "
        "PhyloPhlAn 3 marker database, MAFFT), 35 of which are the benchmark datasets of the manuscript. "
        "benchmark_data_dna: 26 nucleotide alignments simulated with AliSim in IQ-TREE 3, with guide trees, seeds and logs. "
        "benchmark_runs: the paired CPU-versus-GPU run record (RUNS.tsv with one row per paired run, phase timings, replicate table, "
        "key numbers, every run directory with both arms' IQ-TREE logs, trees, support files and equivalence results, and "
        "the column dictionary). supporting_files: README.md explaining every folder with full citations, MANIFEST.sha256 and "
        "ARCHIVES.sha256 checksums, BENCHMARK_SCHEMA.md and install_into_repo.sh, which unpacks the archives into a clone of "
        "the repository.</p>"
        "<p><b>Credit for the genomes.</b> The genomes are copies of assemblies deposited by Li et al. 2025 (160 genomes), "
        "Sanders et al. 2023 (151), Carter et al. 2023 (78), Bowerman et al. 2021 (21), Glendinning et al. 2021 (14), "
        "Svartström et al. 2017 (8), Pu et al. 2023 (2) and Wibowo et al. 2021 (1), renamed to stable identifiers; the "
        "metadata table maps each file to its study and public accessions. Users of the genomes should cite the study that "
        "produced them (listed under related identifiers and in README.md); the genomes remain subject to the terms of the "
        "archives they were obtained from.</p>"
        "<p><b>Licence.</b> The compilation is released under CC BY 4.0. IQ-TREE 3 (Wong et al. 2026, "
        "doi:10.1093/molbev/msag117) must be cited alongside IQ-TREE3-GPU.</p>"
        "<p><b>Tree of this deposit.</b> Archive names are the repository paths with / replaced by _; each unpacks to that "
        "path inside a clone of the repository.</p>"
        "<pre>" + TREE.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>")
METADATA = {
    "upload_type": "dataset",
    "publication_date": datetime.date.today().isoformat(),
    "title": "IQ-TREE3-GPU benchmark data: source genomes, alignments and the paired CPU-GPU run record",
    "creators": [
        {"name": "Cho, Byeongyeon", "affiliation": AFF_CHO, "orcid": "0000-0003-1370-2629"},
        {"name": "Kostic, Aleksandar D.", "affiliation": AFF_KOS, "orcid": "0000-0002-0837-4360"},
    ],
    "contributors": [{"name": "Kostic, Aleksandar D.", "type": "ContactPerson", "affiliation": AFF_KOS, "orcid": "0000-0002-0837-4360"}],
    "description": desc,
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": VER,
    "language": "eng",
    "keywords": ["phylogenetics", "maximum likelihood", "GPU", "CUDA", "IQ-TREE", "benchmark", "reproducibility",
                 "metagenome-assembled genomes", "amino acid substitution models", "ModelFinder"],
    "related_identifiers": related,
    "prereserve_doi": True,
    "notes": "Deposited to accompany the manuscript; the DOI is cited in its Methods and Data availability statement. "
             "The software and code are in the GitHub repository, not in this deposit.",
}
json.dump(METADATA, open(os.path.join(SUP, "zenodo_metadata.json"), "w"), indent=1)
sh(f"cd '{A}' && tar -cf - supporting_files | pigz > supporting_files.tar.gz && rm -rf supporting_files")
print(f"   supporting_files.tar.gz: {os.path.getsize(os.path.join(A,'supporting_files.tar.gz'))/1e6:.1f} MB", flush=True)
if "--stage-only" in sys.argv:
    print("staged only; no upload"); sys.exit(0)
print("== metadata ==", flush=True)
dep = api("GET", f"/deposit/depositions/{DEP_ID}")
assert not dep.get("submitted"), "deposition is already published; refusing to overwrite"
dep = api("PUT", f"/deposit/depositions/{DEP_ID}", {"metadata": METADATA})
doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi") or dep.get("doi")
bucket = dep["links"]["bucket"]
print(f"   title set; reserved DOI {doi}; draft {dep['links'].get('html')}", flush=True)

# ------------------------------------------------------------------ 5. upload
print("== upload ==", flush=True)
old = api("GET", f"/deposit/depositions/{DEP_ID}/files")
for f in old:
    api("DELETE", f"/deposit/depositions/{DEP_ID}/files/{f['id']}")
    print(f"   removed old file {f['filename']} ({float(f.get('filesize',0))/1e6:.0f} MB)", flush=True)
existing = {}
uploaded = []
for name in sorted(os.listdir(A)):
    path = os.path.join(A, name)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.getsize(path) < 2e8 else sh(f"md5sum '{path}'").split()[0]
    if name in existing and existing[name].get("checksum") == md5:
        print(f"   {name}: already present, checksum matches", flush=True); uploaded.append((name, md5, "present")); continue
    r = subprocess.run(["curl", "-sS", "--retry", "5", "--retry-delay", "20", "-H", f"Authorization: Bearer {TOKEN}",
                        "--upload-file", path, f"{bucket}/{name}"], capture_output=True, text=True)
    try: j = json.loads(r.stdout)
    except Exception: sys.exit(f"upload of {name} failed: {r.stdout[:300]} {r.stderr[:300]}")
    remote = (j.get("checksum") or "").replace("md5:", "")
    ok = remote == md5
    print(f"   {name}: {os.path.getsize(path)/1e6:.0f} MB, remote md5 {'matches' if ok else 'MISMATCH'}", flush=True)
    if not ok: sys.exit(f"checksum mismatch for {name}: local {md5} remote {remote}")
    uploaded.append((name, md5, "uploaded"))
dep = api("GET", f"/deposit/depositions/{DEP_ID}")
json.dump({"deposition_id": DEP_ID, "doi": doi, "html": dep["links"].get("html"), "state": dep.get("state"),
           "submitted": dep.get("submitted"), "files": uploaded, "unpacked_files": nfiles}, open(STATE, "w"), indent=1)
print(f"== done: draft {dep['links'].get('html')} (unpublished), DOI {doi}, {len(dep.get('files', []))} files on Zenodo ==", flush=True)
