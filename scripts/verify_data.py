#!/usr/bin/env python3
# Environment: python3 (standard library only)
"""verify_data.py - prove that the input data in data/ is complete and is what the record describes.

Self-contained: needs nothing outside this repository. Checks, in order,

  1. data/genomes/genome_metadata.tsv lists every .fna under data/genomes/ and nothing else, and
     every file's md5 equals the md5 recorded for it;
  2. every tip label of every amino-acid alignment (data/protein/*/concat_alignment.faa and
     examples/data/*.faa) is a genome in that table, and each full-genus alignment
     (data/protein/g__*/) carries exactly the genomes staged under data/genomes/g__*/;
  3. every data/dna/<set>/aln.phy that is present matches data/dna/CHECKSUMS.sha256
     (absent sets are reported, not failed: the large ones are regenerated from their PROVENANCE.md).

Exit 0 only if every check passes. Usage, from anywhere:

    python3 scripts/verify_data.py            # full check (~1 min: reads 830 MB of genomes)
    python3 scripts/verify_data.py --quick    # skip the md5 / sha256 reads
"""
import csv
import glob
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOMES = os.path.join(REPO, "data", "genomes")
PROTEIN = os.path.join(REPO, "data", "protein")
DNA = os.path.join(REPO, "data", "dna")
META = os.path.join(GENOMES, "genome_metadata.tsv")
QUICK = "--quick" in sys.argv[1:]

fails = []
def fail(msg): fails.append(msg); print("FAIL  " + msg)
def ok(msg): print("ok    " + msg)

def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def tips(fasta):
    with open(fasta) as f:
        return [line[1:].split()[0] for line in f if line.startswith(">")]

# 1. genome table vs files on disk
if not os.path.exists(META):
    sys.exit(f"FAIL  missing {META}")
rows = list(csv.DictReader(open(META, newline=""), delimiter="\t"))
listed = {r["stable_genome_id"]: r for r in rows}
on_disk = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(GENOMES, "g__*", "*.fna"))}
if set(listed) != set(on_disk):
    fail(f"genome table and data/genomes/ differ: {len(set(listed) - set(on_disk))} listed but absent, "
         f"{len(set(on_disk) - set(listed))} on disk but unlisted")
else:
    ok(f"{len(on_disk)} genomes on disk, all listed in genome_metadata.tsv")
bad_dir = [g for g, r in listed.items() if g in on_disk and
           os.path.relpath(on_disk[g], REPO) != r["repo_path"]]
if bad_dir:
    fail(f"{len(bad_dir)} genomes are not at the path the table records, e.g. {bad_dir[:3]}")
else:
    ok("every genome sits at the path the table records")
if not QUICK:
    bad = [g for g, r in listed.items() if g in on_disk and digest(on_disk[g], "md5") != r["md5_fna"]]
    if bad: fail(f"{len(bad)} genomes differ from their recorded md5, e.g. {bad[:3]}")
    else: ok(f"md5 of all {len(on_disk)} genome files matches the table")

# 2. alignments
alns = sorted(glob.glob(os.path.join(PROTEIN, "*", "concat_alignment.faa"))) + \
       sorted(glob.glob(os.path.join(REPO, "examples", "data", "*.faa")))
if not alns:
    fail("no amino-acid alignments found")
unknown = {}
for a in alns:
    t = tips(a)
    if len(set(t)) != len(t):
        fail(f"duplicate tip labels in {os.path.relpath(a, REPO)}")
    u = [x for x in t if x not in listed]
    if u: unknown[os.path.relpath(a, REPO)] = u[:3]
if unknown: fail(f"tips that are not listed genomes in {len(unknown)} alignments: {unknown}")
else: ok(f"{len(alns)} alignments, every tip is a listed genome")
mismatch = []
for gdir in sorted(glob.glob(os.path.join(PROTEIN, "g__*"))):
    g = os.path.basename(gdir)
    beside = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(GENOMES, g, "*.fna"))}
    in_aln = set(tips(os.path.join(gdir, "concat_alignment.faa")))
    if beside != in_aln: mismatch.append(g)
if mismatch: fail(f"full-genus alignments whose tips differ from data/genomes/<genus>/: {mismatch}")
else: ok("each full-genus alignment carries exactly the genomes staged for that genus")

# 3. DNA checksums
chk = os.path.join(DNA, "CHECKSUMS.sha256")
if not os.path.exists(chk):
    fail("missing data/dna/CHECKSUMS.sha256")
else:
    present, absent, wrong = 0, [], []
    for line in open(chk):
        line = line.strip()
        if not line or line.startswith("#"): continue
        want, rel = line.split(None, 1)
        rel = rel.lstrip("*")
        # entries are written relative to the repository root (data/dna/<set>/aln.phy)
        p = rel if os.path.isabs(rel) else os.path.join(REPO, rel) if rel.startswith("data/") else os.path.join(DNA, rel)
        if not os.path.exists(p):
            absent.append(rel); continue
        present += 1
        if not QUICK and digest(p, "sha256") != want: wrong.append(rel)
    if wrong: fail(f"DNA alignments with a wrong sha256: {wrong}")
    else: ok(f"DNA alignments present: {present}" + ("" if QUICK else ", all sha256 match"))
    if absent: print(f"note  {len(absent)} DNA alignments not present (regenerable; see their PROVENANCE.md): "
                     + ", ".join(absent[:5]) + (" ..." if len(absent) > 5 else ""))

print(f"\n{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
