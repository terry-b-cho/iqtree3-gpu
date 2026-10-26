#!/usr/bin/env python3
"""zenodo_upload.py - create the Zenodo record as a DRAFT and upload the deposit archives.

    python3 benchmark/scripts/dev/zenodo_upload.py [--sandbox] [--deposition ID] [--publish]

Reads the personal access token from $ZENODO_TOKEN or ~/.zenodo_token (never printed). Creates one
deposition with the metadata below (Zenodo pre-reserves the DOI at creation), uploads every file in
zenodo_upload_bundle/archives/ to the deposition's bucket with curl (streaming, resumable by
re-running: files already present with the same size and checksum are skipped), and writes the
reserved DOI and record URL to zenodo_upload_bundle/DEPOSIT.json. It does NOT publish unless
--publish is given: publishing is irreversible and is left to the author on the Zenodo page.
"""
import hashlib, json, os, subprocess, sys, time, urllib.request, urllib.error

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCH = os.path.join(REPO, "zenodo_upload_bundle", "archives")
SANDBOX = "--sandbox" in sys.argv
STATE = os.path.join(REPO, "zenodo_upload_bundle", "DEPOSIT_sandbox.json" if SANDBOX else "DEPOSIT.json")
API = "https://sandbox.zenodo.org/api" if SANDBOX else "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN") or open(os.path.expanduser("~/.zenodo_sandbox_token" if SANDBOX else "~/.zenodo_token")).read().strip()
VERSION = open(os.path.join(REPO, "VERSION")).read().strip().replace("-dev", "")

METADATA = {
    "upload_type": "dataset",
    "title": "IQ-TREE3-GPU: benchmark inputs, results and code (exact GPU likelihoods for IQ-TREE 3)",
    "creators": [
        {"name": "Cho, Byeongyeon",
         "affiliation": "Department of Biomedical Informatics, Harvard Medical School; Department of Microbiology "
                        "and Immunobiology, Harvard Medical School; Section on Pathophysiology and Molecular "
                        "Pharmacology, Joslin Diabetes Center",
         "orcid": "0000-0003-1370-2629"},
        {"name": "Kostic, Aleksandar D.",
         "affiliation": "Department of Microbiology and Immunobiology, Harvard Medical School; Section on "
                        "Pathophysiology and Molecular Pharmacology, Joslin Diabetes Center",
         "orcid": "0000-0002-0837-4360"},
    ],
    "description": (
        "<p>The archived record behind the manuscript <i>IQ-TREE3-GPU: GPU-accelerated maximum-likelihood "
        "phylogenetic tree inference for complex amino acid substitution models</i> (Cho and Kostic, 2026): "
        "IQ-TREE3-GPU, a CUDA backend for the conditional-likelihood kernel of IQ-TREE 3 (Wong et al. 2026, "
        "doi:10.1093/molbev/msag117), and its paired CPU-versus-GPU benchmark campaign. Code repository: "
        "https://github.com/kosticlab/iqtree3-gpu (mirror https://github.com/terry-b-cho/iqtree3-gpu).</p>"
        "<p><b>Files.</b> 01: the 45 amino-acid alignment files (concatenated marker-gene alignments of "
        "metagenome-assembled genomes, one genus each): the 35 benchmark datasets of the paper (the size ladder, "
        "the single-thread sets, the column-subsampled sets and the full genera run at scale) and the 10 further "
        "full-genus and single-thread alignments they were subsampled from or that were staged but not run. "
        "02: the 26 simulated nucleotide alignments with the guide tree, model string, seed and AliSim log that "
        "regenerate each. 03: the source MAG genome FASTAs of the 14 genera. 04: the benchmark results (the run "
        "table with one row per paired run; key numbers; phase steps; replicate tables; for every cell both arms' "
        "IQ-TREE logs, standard output with the engine's arena report, trees, support files and command lines; "
        "profiling records including the empty-kernel launch measurement). 06: the source code of the backend, the "
        "IQ-TREE 3 patch, the benchmark harness, the analysis and figure pipeline, tests and documentation "
        "(GPL-2.0-only; LICENSE inside). The figure files and the Source Data are supplied with the submission "
        "and are not part of this record. MANIFEST.sha256 lists every file of the unpacked record with its checksum; "
        "ARCHIVES.sha256 covers the archives themselves.</p>"
        "<p><b>Reproducing.</b> Unpack 06, follow benchmark/REPRODUCING.md to build the binary from the IQ-TREE 3 source at "
        "the recorded commit with the patch, then run the paired benchmark on the alignments of 01 and 02; "
        "benchmark/scripts/refresh_all.sh regenerates every table, key number and figure from the run outputs.</p>"
        "<p>The data are released under CC BY 4.0; the code archive (06) is GPL-2.0-only as stated in its LICENSE.</p>"
    ),
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": "1",                      # record version; the code archive carries its own VERSION (0.1.0)
    "keywords": ["phylogenetics", "maximum likelihood", "GPU", "CUDA", "IQ-TREE", "benchmark",
                 "reproducibility", "metagenome-assembled genomes"],
    "prereserve_doi": True,
    "notes": "Deposited to accompany the manuscript submission; the DOI is cited in its Data availability statement.",
}


def api(method, path, data=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo API {method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    dep_id = None
    for i, a in enumerate(sys.argv):
        if a == "--deposition": dep_id = int(sys.argv[i + 1])
    if dep_id is None and state.get("api") == API: dep_id = state.get("id")
    if dep_id is None:
        dep = api("POST", "/deposit/depositions", {"metadata": METADATA})
        dep_id = dep["id"]
        print(f"  created deposition {dep_id} (draft)")
    else:
        dep = api("PUT", f"/deposit/depositions/{dep_id}", {"metadata": METADATA})
        print(f"  updated metadata of deposition {dep_id}")
    doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi") or dep.get("doi")
    bucket = dep["links"]["bucket"]
    state = dict(api=API, id=dep_id, doi=doi, html=dep["links"].get("html"), bucket=bucket, files={})
    json.dump(state, open(STATE, "w"), indent=1)
    print(f"  reserved DOI: {doi}\n  page: {dep['links'].get('html')}")
    # files already in the bucket
    have = {f["filename"]: f for f in api("GET", f"/deposit/depositions/{dep_id}/files")}
    names = sorted(f for f in os.listdir(ARCH) if not f.startswith(".") and not f.endswith(".sh"))
    if "--prune" in sys.argv:                 # files on the record that the deposit folder no longer holds
        for name in sorted(set(have) - set(names)):
            api("DELETE", f"/deposit/depositions/{dep_id}/files/{have[name]['id']}")
            print(f"  {name}: removed from the record (not in {ARCH})")
            del have[name]
    for name in names:
        path = os.path.join(ARCH, name); size = os.path.getsize(path)
        if name in have and int(have[name].get("filesize", -1)) == size and have[name].get("checksum") == md5(path):
            print(f"  {name}: already uploaded ({size:,} B)"); state["files"][name] = have[name]["id"]; continue
        if name in have:
            api("DELETE", f"/deposit/depositions/{dep_id}/files/{have[name]['id']}")
        t0 = time.time()
        r = subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "30", "-X", "PUT",
                            "-H", f"Authorization: Bearer {TOKEN}", "--upload-file", path, f"{bucket}/{name}"],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"  upload of {name} failed: {r.stderr[-400:]}")
        info = json.loads(r.stdout)
        ok = info.get("checksum", "").replace("md5:", "") == md5(path)
        print(f"  {name}: uploaded {size:,} B in {time.time()-t0:.0f} s, checksum {'OK' if ok else 'MISMATCH'}")
        if not ok: sys.exit(1)
        state["files"][name] = info.get("version_id", "")
        json.dump(state, open(STATE, "w"), indent=1)
    final = {f["filename"]: f for f in api("GET", f"/deposit/depositions/{dep_id}/files")}
    bad = [n for n, f in final.items() if n not in names or f.get("checksum") != md5(os.path.join(ARCH, n))]
    print(f"  record holds {len(final)} files; checksums {'all equal the local files' if not bad else 'DIFFER for ' + ', '.join(bad)}")
    if bad: sys.exit(1)
    if "--publish" in sys.argv:
        pub = api("POST", f"/deposit/depositions/{dep_id}/actions/publish")
        print(f"  PUBLISHED: {pub.get('doi_url')}")
    else:
        print("  draft complete; not published (open the page, check, and press Publish)")


if __name__ == "__main__":
    main()
