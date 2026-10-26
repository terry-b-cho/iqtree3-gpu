#!/usr/bin/env python3
"""supplementary_data.py - the Supplementary Data workbook of the tool paper, and its check.

    python3 scripts/supplementary_data.py [<out.xlsx>]

Nature Communications distinguishes Supplementary Tables (in the SI document) from Supplementary
Data (machine-readable files). The scoping review's tables are Supplementary Tables S1-S32 in
Supplementary Note 1 and travel in their own workbook. This workbook holds the tables behind the
main text and Supplementary Note 2, numbered Supplementary Data 1-12, one sheet each, from the
archived record's files without alteration, plus a README sheet and a column dictionary for the run
table taken from docs/BENCHMARK_SCHEMA.md. Data 7 (the engine's arena report per run) is assembled here
from every run's gpu.stdout with the same parser the key numbers use. After writing, the workbook is
read back and every cell compared with its source; any difference fails the build.

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
"""
import csv, os, re, sys, importlib.util as ilu
import pandas as pd

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results", "benchmarks", "Supplementary_Data_benchmark.xlsx")


def _load(n, path=None):
    p = path or os.path.join(REPO, "scripts", "lib", n + ".py")
    s = ilu.spec_from_file_location(n, p); m = ilu.module_from_spec(s); s.loader.exec_module(m)
    return m


def read_tsv(path):
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    rows = list(csv.reader(lines, delimiter="\t"))
    hdr = rows[0]
    body = [r + [""] * (len(hdr) - len(r)) for r in rows[1:] if any(c.strip() for c in r)]
    return pd.DataFrame(body, columns=hdr, dtype=str).fillna("")


def arena_table():
    """Supplementary Data 7: the engine's arena report for every recorded run."""
    K = _load("k", os.path.join(REPO, "scripts", "key_numbers.py"))
    FD = _load("figdata")                       # the single definition of the superseded build
    runs = read_tsv(os.path.join(REPO, "results", "benchmarks", "RUNS.tsv"))
    out = []
    for _, r in runs.iterrows():
        rep = K.arena_report(r)
        build = "superseded" if (r.get("binary_sha256") or "").startswith(FD.SUPERSEDED_SHA_PREFIX) else "reported"
        row = dict(cell=r["cell"], run_id=r["run_id"], build=build, data_type=r["data_type"],
                   genus=r["genus"], taxa=r["taxa"], patterns=r["patterns"], model=r["model"], card=r["card"],
                   threads=r["threads"])
        if rep:
            g = rep.get("guard") or {}
            row.update(ncat_selected=rep["ncat_selected"], ncat_capacity_allocated=rep["ncat_cap"],
                       capacity_reduced="yes" if g else "no", capacity_requested=g.get("frm", rep["ncat_cap"]),
                       slots=rep["slots"], iqtree_lh_slots=rep["iqtree_slots"], slot_partials_mib=rep["slot_mb"],
                       slot_counters_mib=rep["counters_mb"], arena_footprint_gib=rep["footprint_gb"],
                       need_at_requested_gib=g.get("need_gb", ""), device_usable_gib=g.get("have_gb", ""),
                       budget_gib=("%.2f" % g["budget_gb"]) if g else "")
        else:
            row.update(ncat_selected="", ncat_capacity_allocated="", capacity_reduced="", capacity_requested="", slots="",
                       iqtree_lh_slots="", slot_partials_mib="", slot_counters_mib="", arena_footprint_gib="",
                       need_at_requested_gib="", device_usable_gib="", budget_gib="")
        row.update(peak_vram_sampled_mib=r.get("peak_vram_mb", ""), peak_ram_job_mib=r.get("peak_ram_mb", ""))
        out.append(row)
    d = pd.DataFrame(out).astype(str).replace({"nan": "", "None": ""})
    return d


def column_dictionary():
    """The run table's column dictionary, from docs/BENCHMARK_SCHEMA.md, restricted to columns present."""
    runs_cols = list(read_tsv(os.path.join(REPO, "results", "benchmarks", "RUNS.tsv")).columns)
    rows, group = [], ""
    for line in open(os.path.join(REPO, "docs", "docs/BENCHMARK_SCHEMA.md")):
        m = re.match(r"^### ([A-M])\. (.*)$", line.strip())
        if m: group = f"{m.group(1)}. {m.group(2)}"
        m = re.match(r"^\| ([a-z_0-9]+) \| ([^|]*) \| ([^|]*) \| (.*) \|\s*$", line.rstrip("\n"))
        if m and m.group(1) in runs_cols:
            rows.append(dict(column=m.group(1), schema_group=group, source=m.group(2).strip(), status=m.group(3).strip(), definition=m.group(4).strip()))
    seen = {r["column"] for r in rows}
    for c in runs_cols:
        if c not in seen:
            rows.append(dict(column=c, schema_group="", source="", status="", definition="(present in RUNS.tsv; see docs/BENCHMARK_SCHEMA.md)"))
    return pd.DataFrame(rows, dtype=str)


INT_RX = re.compile(r"^-?\d+$")
FLOAT_RX = re.compile(r"^-?\d+\.\d+$|^-?\d+(\.\d+)?[eE][+-]?\d+$")


def _sig_digits(v):
    m = re.sub(r"[-+]", "", v.split("e")[0].split("E")[0]).replace(".", "").lstrip("0")
    return len(m.rstrip("0")) if m else 0

def to_number(col):
    """Numbers only when Excel can hold every value exactly (at most 15 significant digits)."""
    vals = [v for v in col if v != ""]
    if not vals: return None
    if any(_sig_digits(v) > 15 for v in vals): return None
    if all(INT_RX.match(v) for v in vals) and all(len(v) < 16 for v in vals) and not any(v != "0" and v.startswith("0") for v in vals):
        return col.map(lambda v: int(v) if v != "" else None).astype("object")
    if all(INT_RX.match(v) or FLOAT_RX.match(v) for v in vals):
        return col.map(lambda v: float(v) if v != "" else None).astype("object")
    return None


def main():
    B = os.path.join(REPO, "results", "benchmarks"); P = os.path.join(REPO, "results", "profiling")
    items = [
        (1, "RUNS", "Run table", "One row per paired run (226: the 182 runs of the reported build and the 44 runs of the superseded build, identified by binary_sha256, whose checksum begins 07c70cae): dataset, workflow, thread count, card, node, host CPU, driver, binary checksum, selected model, both arms' timings and log-likelihoods, the equivalence outcome, the engine's engagement counters and the sampled device metrics. Column dictionary in the last sheet.", read_tsv(f"{B}/RUNS.tsv"), "results/benchmarks/RUNS.tsv", "Methods; Results; Note 2 §1, §11.1"),
        (2, "RUNS_summary", "Run-table summary", "Mean and s.d. of speedup and device utilisation per (alphabet, model treatment, thread count, card) stratum, with engagement and gate-pass counts.", read_tsv(f"{B}/RUNS_summary.tsv"), "results/benchmarks/RUNS_summary.tsv", "Note 2 §11.1"),
        (3, "KEY_NUMBERS", "Key numbers", "Every number quoted in the Abstract, Results and Discussion as key-value pairs (395 keys), regenerated from the run table by scripts/key_numbers.py; no number in the manuscript is hand-transcribed.", read_tsv(f"{B}/KEY_NUMBERS.tsv"), "results/benchmarks/KEY_NUMBERS.tsv", "Note 2 §10, §11.2"),
        (4, "PHASE_STEPS", "Phase steps", "Per-step wall and CPU time for both arms of all 226 recorded cells, with a level column separating the two top-level phases from the setup steps nested inside them.", read_tsv(f"{B}/PHASE_STEPS.tsv"), "results/benchmarks/PHASE_STEPS.tsv", "Results (Fig. 4); Supplementary Fig. S13; Note 2 §11.1"),
        (5, "REPLICATES_protein20T", "Replicate sets, amino acid", "Replicate sets by size at 20 threads: median, mean, s.d., coefficient of variation and the contributing speedups.", read_tsv(f"{B}/REPLICATES_protein20T.tsv"), "results/benchmarks/REPLICATES_protein20T.tsv", "Results; Supplementary Fig. S6"),
        (6, "REPLICATES_dna20T", "Replicate sets, nucleotide", "As Data 5, nucleotide.", read_tsv(f"{B}/REPLICATES_dna20T.tsv"), "results/benchmarks/REPLICATES_dna20T.tsv", "Results; Supplementary Fig. S7"),
        (7, "ARENA_REPORTS", "Device-arena reports", "The engine's own arena report for every recorded run, parsed from gpu.stdout with the parser the key numbers use: selected and allocated rate-category counts, whether the capacity was reduced (the six guard runs) and from what, slot count, per-slot partial-likelihood and counter sizes, arena footprint, the need at the requested capacity, the usable device memory and the 70% budget, with the sampled device peak and the job's peak host memory from the run table.", arena_table(), "results/benchmarks/paired/<cell>/gpu.stdout; results/benchmarks/RUNS.tsv", "Results (device-memory demand; Fig. 6); Methods (implementation); Note 2 §7"),
        (8, "CARD_SPECS", "Card specifications", "Vendor-nominal specifications per card (fp64, fp32, bandwidth, VRAM, TDP, MSRP) used to order the cards in Fig. 5, with the vendor document named per row and a flag for values derived rather than published.", read_tsv(os.path.join(REPO, "scripts", "benchmark", "card_specs.tsv")), "scripts/benchmark/card_specs.tsv", "Results (Fig. 5); Note 2 §9"),
        (9, "LAUNCH_LATENCY", "Empty-kernel launch latency", "Empty-kernel launch cost on the L40S: sync (launch plus synchronise, timed per launch) and async (back-to-back launches, one synchronisation): n, median, quartiles, mean and minimum in microseconds.", read_tsv(f"{P}/launch_latency_l40s.tsv"), "results/profiling/launch_latency_l40s.tsv", "Results (mechanism); Methods (profiling); Note 2 §8"),
        (10, "NSYS_KERNEL_SUMMARY", "Nsight Systems kernel attribution", "One row per alphabet from the profiled run: wall, kernel and memcpy seconds, launch count, the top kernel and its share.", read_tsv(f"{P}/nsys_kernel_summary.tsv"), "results/profiling/nsys_kernel_summary.tsv", "Results (Fig. 3D); Note 2 §8"),
        (11, "CPU_PRODUCTION_SCALING", "Production-scale CPU runs", "CPU-only production phylogenomic runs used to characterise the workload (genus, genome count, alignment length, wall time, peak host memory); not paired benchmark measurements.", read_tsv(f"{B}/cpu_production_scaling.tsv"), "results/benchmarks/cpu_production_scaling.tsv", "Supplementary Fig. S5"),
        (12, "NSIGHT_SUMMARY", "Nsight Compute summary (header only)", "Header-only by design: the Nsight Compute hardware counters were administratively restricted on the cluster, so no kernel rows were collected; the empty table is distributed so that its emptiness is visible.", read_tsv(f"{P}/nsight_summary.tsv"), "results/profiling/nsight_summary.tsv", "Methods (profiling); Note 2 §11.1"),
    ]
    coldict = column_dictionary()
    readme = pd.DataFrame([dict(supplementary_data=f"Supplementary Data {n}", sheet=f"Data {n}", title=t, contents=c, rows=len(d), columns=len(d.columns), source_file=src, cited_in=cit)
                           for n, s, t, c, d, src, cit in items])
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        head = pd.DataFrame({"Supplementary Data": [
            "Machine-readable tables behind the main text and Supplementary Note 2 of the IQ-TREE3-GPU manuscript, one sheet per item, numbered Supplementary Data 1-12.",
            "The scoping review's tables (Supplementary Tables S1-S32) are in Supplementary Note 1 and its own workbook, not here.",
            "Each sheet is the archived record's file without alteration: a cell holds a number only when every value in its column is a number; identifiers, checksums, model strings and command lines are text. Empty cells are empty in the source as well.",
            "The last sheet, RUNS_columns, defines every column of Data 1 (from docs/BENCHMARK_SCHEMA.md in the record).",
        ]})
        head.to_excel(xw, sheet_name="README", index=False)
        readme.to_excel(xw, sheet_name="README", index=False, startrow=len(head) + 2)
        for n, s, t, c, d, src, cit in items:
            e = d.copy()
            for col in e.columns:
                v = to_number(e[col])
                if v is not None: e[col] = v
            e.to_excel(xw, sheet_name=f"Data {n}", index=False)
            xw.sheets[f"Data {n}"].freeze_panes = "A2"
        coldict.to_excel(xw, sheet_name="RUNS_columns", index=False)
        xw.sheets["README"].column_dimensions["C"].width = 34; xw.sheets["README"].column_dimensions["D"].width = 110
    # verify
    import openpyxl
    wb = openpyxl.load_workbook(OUT, read_only=True)
    expected = ["README"] + [f"Data {n}" for n, *_ in items] + ["RUNS_columns"]
    problems = [] if wb.sheetnames == expected else [f"sheets {wb.sheetnames}"]
    cells = 0
    for n, s, t, c, d, src, cit in items + [(None, None, None, None, coldict, None, None)]:
        name = f"Data {n}" if n else "RUNS_columns"
        rows = list(wb[name].iter_rows(values_only=True))
        hdr = [("" if v is None else str(v)) for v in rows[0]]
        if hdr != list(d.columns): problems.append(f"{name}: header"); continue
        if len(rows) - 1 != len(d): problems.append(f"{name}: {len(rows)-1} rows vs {len(d)}"); continue
        for i, r in enumerate(rows[1:]):
            for j, col in enumerate(d.columns):
                cells += 1
                sv = d.iat[i, j]; x = r[j] if j < len(r) else None
                if x is None:
                    if sv != "": problems.append(f"{name} r{i+2} {col}: empty vs {sv!r}"); break
                elif isinstance(x, (int, float)) and not isinstance(x, bool):
                    if float(sv) != float(x): problems.append(f"{name} r{i+2} {col}: {x} vs {sv}"); break
                elif str(x) != sv: problems.append(f"{name} r{i+2} {col}: {str(x)[:30]!r} vs {sv[:30]!r}"); break
            if len(problems) > 10: break
    if problems:
        print("  VERIFY FAILED: " + "; ".join(problems[:10])); sys.exit(1)
    print(f"  wrote {os.path.relpath(OUT, REPO)}: {len(items)} data sheets + README + RUNS_columns; verified {cells:,} cells against the sources")
    import subprocess
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "style_workbooks.py"), OUT], check=True)


if __name__ == "__main__":
    main()
