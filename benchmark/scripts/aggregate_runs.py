#!/usr/bin/env python3
"""aggregate_runs.py - assemble the master benchmark record from per-run cells.

benchmark/BENCHMARK_SCHEMA.md rule: every benchmark job is one fully-recorded row. This aggregator scans every
benchmark/runs/paired/<cell>/RESULT.tsv (the row emitted by paired_benchmark.sh) plus its
companion PHASES.tsv, then:

  1. UPSERTS each cell into benchmark/runs/RUNS.tsv - the append-only master record, keyed by
     (run_id, cell). Re-running updates a cell's row in place; rows for cells no longer on disk are
     retained (append-only history). New schema columns are unioned in, never dropped.
  2. REGENERATES benchmark/runs/RUNS_summary.tsv - one row per (data_type, threads, card) with
     n, mean+/-sd of speedup, mean pct_gpu, mean SM/mem utilisation, and the gate_pass rate.

A cell missing GPU-engagement proof is recorded with engaged=INVALID; a cell missing an equivalence
result is recorded with gate_pass=NA. Such rows are still written to RUNS.tsv (never silently dropped)
but are excluded from the summary means, exactly as the schema requires.

Stdlib only (csv/statistics). Usage: python3 aggregate_runs.py

Paths: the repository root is resolved at run time from $REPO_DIR or this file's own location.
Nothing is hardcoded, so the repository runs from any directory.
"""
import os, shutil, glob, csv, sys, statistics, re
from datetime import datetime, timezone

REPO = os.environ.get("REPO_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PAIRED = os.path.join(REPO, "benchmark", "runs", "paired")
RUNS = os.path.join(REPO, "benchmark", "runs", "RUNS.tsv")
SUMMARY = os.path.join(REPO, "benchmark", "runs", "RUNS_summary.tsv")

# Values that mean "not a number" for aggregation purposes (schema uses NA; second-wave stubs TODO_2W).
_NULLS = {"", "NA", "N/A", "NONE", "NULL", "TODO_2W", "TODO"}

# Columns that older per-cell RESULT.tsv files carry but the published run table does not. The
# per-cell files are raw evidence and are left as written; the column is dropped here, at ingest,
# so it enters neither RUNS.tsv nor anything derived from it.
DROPPED_COLUMNS = {"accelerable_fraction"}


def to_float(x):
    """Return float(x) or None for any null/placeholder token. Never raises."""
    if x is None:
        return None
    s = str(x).strip()
    if s.upper() in _NULLS:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_result(path):
    """First (only) data row of a RESULT.tsv as a dict, or None if empty/unreadable."""
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
    except OSError:
        return None
    return dict(rows[0]) if rows else None


def collect_cells():
    """Every paired cell's RESULT.tsv, normalised and annotated with provenance/flags."""
    cells = []
    for f in sorted(glob.glob(os.path.join(PAIRED, "*", "RESULT.tsv"))):
        r = load_result(f)
        if r is None:
            continue
        cell = os.path.basename(os.path.dirname(f))
        # Only ingest well-formed cell dirs: <genus>__T<threads>__<card>. This rejects stray or
        # renamed directories (.hidden, .bak, manual copies) that would otherwise be ingested as
        # spurious NEW cells and silently inflate the table.
        if not re.match(r"^[A-Za-z0-9_.\-]+__T\d+__[A-Za-z0-9_]+$", cell):
            print(f"  skipping malformed cell dir: {cell}")
            continue
        # Provenance columns not present inside RESULT.tsv itself.
        #
        # These two are stored REPOSITORY-RELATIVE. They are not frozen provenance like the
        # command_line_* columns: they are regenerated on every aggregation pass, so storing them
        # absolute baked whichever machine last ran the aggregator into the distributed table. A
        # relative pointer resolves in any clone; join it to the repository root to get a usable path.
        r["cell"] = cell
        r["result_path"] = os.path.relpath(f, REPO)
        phases = os.path.join(os.path.dirname(f), "PHASES.tsv")
        r["phases_file"] = os.path.relpath(phases, REPO) if os.path.exists(phases) else "NA"
        # CARD-SWEEP GROUP-J MERGE: card_sweep.sh writes RESULT_cardsweep_J.tsv = the core row
        # joined with the group-J columns (static card specs + sustained clocks/throttle/power). Fold
        # those extra columns into this cell's row so the card-sweep data lives in RUNS.tsv too, fully
        # harmonised. setdefault never overwrites a core value; it only ADDS the group-J columns.
        jf = os.path.join(os.path.dirname(f), "RESULT_cardsweep_J.tsv")
        jr = load_result(jf) if os.path.exists(jf) else None
        if jr:
            for k, v in jr.items():
                # A REAL group-J measurement must beat a placeholder: paired_benchmark.sh stubs second-wave columns
                # (gpu_power_mean_w, gpu_energy_j, ...) as TODO_2W in its own row, and plain setdefault
                # let the stub win over the card sweep's measured value. Overwrite only null tokens --
                # a genuine core value is still never replaced.
                if (r.get(k) is None) or (str(r.get(k)).strip().upper() in _NULLS):
                    r[k] = v
            r["cardsweep_J"] = "yes"
        # Normalise the two trust flags so a blank never masquerades as valid.
        if not r.get("engaged"):
            r["engaged"] = "INVALID"
        if not r.get("gate_pass"):
            r["gate_pass"] = "NA"
        # MODEL-SELECTION MODE -- an explicit, reader-facing column derived from the ACTUAL command
        # (the ground truth in command_line_gpu), so every row states unambiguously whether ModelFinder
        # was bounded to a representative model set, run in full, or skipped for a fixed model. This is
        # required for honest publication: a bounded-ModelFinder speedup must never be read as a full one.
        cmd = r.get("command_line_gpu") or r.get("command_line") or ""
        mset = re.search(r"-mset\s+(\S+)", cmd)
        mrate = re.search(r"-mrate\s+(\S+)", cmd)
        m_mf = re.search(r"-m\s+(TESTNEW|TEST|MFP|MF)\b", cmd)
        if mset:
            r["model_search"] = "bounded_ModelFinder"
            r["mf_flags"] = " ".join(x for x in (
                "-mset " + mset.group(1),
                ("-mrate " + mrate.group(1)) if mrate else "") if x)
        elif m_mf:
            r["model_search"] = "full_ModelFinder_" + m_mf.group(1)
            r["mf_flags"] = "none"
        elif re.search(r"-m\s+\S", cmd):
            r["model_search"] = "fixed_model"
            r["mf_flags"] = "none"
        else:
            r["model_search"] = "NA"
            r["mf_flags"] = "NA"
        # DATA-SOURCE PROVENANCE (round 8, R26). paired_benchmark.sh derives data_source from the alignment path
        # and labels anything outside an "alisim"/"simulated" path as real_MAG; the nucleotide sets
        # live under benchmark/data/dna/dna_* and were all simulated with AliSim (benchmark/data/dna/TRUE_MODEL.txt,
        # PROVENANCE.md per set), so every nucleotide row carried the wrong label. Corrected here at
        # aggregation: this is a provenance column, not a measurement, and the overwrite guard below
        # compares measurements only.
        if (r.get("data_type") or "").lower() == "dna":
            r["data_source"] = "alisim_simulated"
        for k in DROPPED_COLUMNS:
            r.pop(k, None)
        cells.append(r)
    return cells


def read_existing(path):
    """Existing RUNS.tsv as (header_order, {(run_id,cell): row})."""
    if not os.path.exists(path):
        return [], {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        index = {}
        for row in reader:
            key = (row.get("run_id", "NA") or "NA", row.get("cell", "") or "")
            index[key] = dict(row)
    return header, index


def column_order(header, records):
    """Preserve the existing header order, then append any new columns seen, keeping 'cell'/'run_id'
    at the front so the master is easy to eyeball."""
    cols = list(header)
    for rec in records:
        for k in rec:
            if k not in cols:
                cols.append(k)
    # promote key columns to the front
    for front in ("phases_file", "result_path"):
        if front in cols:
            cols.remove(front); cols.append(front)  # keep bulky path columns at the end
    # Reader-facing columns first (reverse order -- each insert(0) prepends), so opening RUNS.tsv
    # immediately shows: which cell, what data/model-search mode, and the headline result + trust bit.
    for front in ("speedup", "gate_pass", "engaged", "mf_flags", "model_search",
                  "threads", "data_type", "datetime_utc", "run_id", "cell"):
        if front in cols:
            cols.remove(front); cols.insert(0, front)
    return cols


def write_master(cols, index):
    os.makedirs(os.path.dirname(RUNS), exist_ok=True)
    # Deterministic order: datetime then cell (falls back gracefully when datetime is NA).
    rows = sorted(index.values(), key=lambda r: (r.get("datetime_utc", ""), r.get("cell", "")))
    # Write both the TSV master and a CSV sibling (RUNS.csv) with identical content, so the record can
    # be opened directly in spreadsheet tools that assume comma-separated. csv.writer quotes any field
    # containing a comma/quote/newline, so the CSV is lossless even though some columns hold command lines.
    for path, delim in ((RUNS, "\t"), (RUNS[:-4] + ".csv", ",")):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter=delim, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "NA") for c in cols})
    return len(rows)


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "NA", "NA", 0
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4g}", f"{sd:.4g}", len(vals)


def write_summary(index):
    groups = {}
    # Group by model_search too, so a bounded-ModelFinder mean is NEVER pooled with a full-ModelFinder
    # or fixed-model mean -- they are different measurements and must be reported separately.
    # Archived rows (cell dir no longer on disk) are kept in RUNS.tsv for the record but excluded from
    # the summary means -- only current measurements are pooled.
    for r in index.values():
        if r.get("row_status") == "archived":
            continue
        key = (r.get("data_type", "NA") or "NA",
               r.get("model_search", "NA") or "NA",
               r.get("threads", "NA") or "NA",
               r.get("card", "NA") or "NA")
        groups.setdefault(key, []).append(r)

    cols = ["data_type", "model_search", "threads", "card", "n_runs", "n_engaged",
            "speedup_mean", "speedup_sd", "speedup_n",
            "pct_gpu_mean", "util_sm_mean", "util_mem_mean",
            "gate_pass_rate", "n_gate_evaluated", "n_gate_pass"]
    out = []
    for (dt, ms, thr, card), rs in sorted(groups.items()):
        engaged = [r for r in rs if r.get("engaged") == "OK"]
        sp_m, sp_sd, sp_n = mean_sd([to_float(r.get("speedup")) for r in engaged])
        pct_m, _, _ = mean_sd([to_float(r.get("pct_gpu")) for r in engaged])
        usm_m, _, _ = mean_sd([to_float(r.get("util_sm_mean")) for r in engaged])
        umem_m, _, _ = mean_sd([to_float(r.get("util_mem_mean")) for r in engaged])
        gate_vals = [r.get("gate_pass") for r in rs]
        n_eval = sum(1 for g in gate_vals if g in ("TRUE", "FALSE"))
        n_pass = sum(1 for g in gate_vals if g == "TRUE")
        rate = f"{n_pass / n_eval:.3f}" if n_eval else "NA"
        out.append({
            "data_type": dt, "model_search": ms, "threads": thr, "card": card,
            "n_runs": len(rs), "n_engaged": len(engaged),
            "speedup_mean": sp_m, "speedup_sd": sp_sd, "speedup_n": sp_n,
            "pct_gpu_mean": pct_m, "util_sm_mean": usm_m, "util_mem_mean": umem_m,
            "gate_pass_rate": rate, "n_gate_evaluated": n_eval, "n_gate_pass": n_pass,
        })
    for path, delim in ((SUMMARY, "\t"), (SUMMARY[:-4] + ".csv", ",")):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter=delim)
            w.writeheader()
            w.writerows(out)
    return out


def main():
    cells = collect_cells()
    if not cells:
        print("  no RESULT.tsv cells found under benchmark/runs/paired/")
        return
    # ONE ROW PER CELL, APPEND-SAFE. Keyed by the cell directory (genus+threads+card). Current disk
    # state upserts its cell's row; rows whose cell dir is NO LONGER on disk are RETAINED from the
    # existing RUNS.tsv and marked row_status=archived -- data is never silently dropped. A re-run of
    # the SAME cell intentionally supersedes that cell's row (same measurement point, newer
    # measurement); every prior snapshot is additionally preserved in git history, so nothing is ever
    # lost. New cells (new sizes, new cards, replicates) simply ADD rows.
    _, prev = read_existing(RUNS)
    index = {}
    for key, old in prev.items():
        cellname = key[1] if isinstance(key, tuple) else key
        old["row_status"] = "archived"          # provisional; flipped to current if still on disk
        for k in DROPPED_COLUMNS:
            old.pop(k, None)
        index[cellname] = old
    skipped_legacy = 0
    for r in cells:
        # Exclude pre-schema (legacy 25-column) cells: they lack data_type, come from a superseded binary,
        # and are being re-run on the current engine.
        if (r.get("data_type") or "NA").strip() in ("", "NA"):
            skipped_legacy += 1
            continue
        r["row_status"] = "current"
        index[r["cell"]] = r          # one RESULT.tsv per dir -> upsert this cell's row
    if not index:
        print("  no full-schema cells yet (legacy cells excluded); nothing to publish")
        return
    upserts = len(index)

    # ---- NON-DESTRUCTIVE GUARANTEE (hard, not aspirational) -------------------------------------
    # RUNS.tsv/.csv are the primary record; every other benchmark file (RUNS_summary.*,
    # REPLICATES_*.tsv) is DERIVED and regenerated from it, so protecting this table protects all of
    # them. Before publishing we assert the two invariants that "only add rows, never corrupt" means:
    #   (1) no previously recorded cell may DISAPPEAR, and
    #   (2) a previously recorded cell's measured speedup may only change if that cell was actually
    #       re-run (its dir is on disk this pass) - so a re-run of a FAILED cell can be repaired,
    #       while an untouched cell is frozen.
    # A violation aborts before anything is written, and a timestamped copy of the pre-existing table
    # is kept under benchmark/runs/_snapshots/ so any pass is recoverable.
    prior = {}
    if os.path.exists(RUNS):
        with open(RUNS) as fh:
            prior = {r["cell"]: r for r in csv.DictReader(fh, delimiter="\t") if r.get("cell")}
    if prior:
        snapdir = os.path.join(REPO, "benchmark", "runs", "_snapshots")
        os.makedirs(snapdir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(RUNS, os.path.join(snapdir, f"RUNS_{stamp}.tsv"))
        # THE POLICY, stated exactly: losing or altering a SUCCESSFUL measurement is corruption and
        # aborts the write. A FAILED row (gate never evaluated, or the run did not produce a usable
        # speedup) carries no measurement, so replacing or dropping it is housekeeping, not loss.
        def ok(row):
            try:
                float(row.get("speedup"))
            except (TypeError, ValueError):
                return False
            return row.get("engaged") == "OK" and row.get("gate_pass") == "TRUE"

        good_prior = {k: r for k, r in prior.items() if ok(r)}
        lost = sorted(set(good_prior) - set(index))
        if lost:
            raise SystemExit(f"ABORT: {len(lost)} SUCCESSFUL cell(s) would vanish: {lost[:5]} "
                             f"- refusing to write RUNS.tsv")
        clobbered = [f"{k}: {good_prior[k].get('speedup')} -> {index[k].get('speedup')}"
                     for k in good_prior
                     if not ok(index[k]) or index[k].get("speedup") != good_prior[k].get("speedup")]
        if clobbered:
            raise SystemExit("ABORT: a successful measurement would be overwritten by a worse or "
                             f"different value: {clobbered[:5]} - refusing to write RUNS.tsv")
        # --clean-failed: drop rows that carry no measurement. Safe by construction - `ok()` rows are
        # never touched, and the abort checks above run first.
        if "--clean-failed" in sys.argv:
            stale = [k for k, r in index.items() if not ok(r)]
            for k in stale:
                del index[k]
            if stale:
                print(f"  --clean-failed: removed {len(stale)} failed row(s): {stale}")
        repaired = [k for k, r in prior.items() if not ok(r) and k in index and ok(index[k])]
        dropped = sorted(set(prior) - set(index))          # failed-only, by the check above
        print(f"  non-destructive check PASSED: {len(good_prior)} successful cells preserved "
              f"byte-for-byte (snapshot: _snapshots/RUNS_{stamp}.tsv)")
        if repaired:
            print(f"  repaired {len(repaired)} previously-failed cell(s): {repaired[:5]}")
        if dropped:
            print(f"  dropped {len(dropped)} failed row(s) with no cell dir: {dropped[:5]}")

    cols = column_order([], list(index.values()))
    n = write_master(cols, index)
    summary = write_summary(index)
    print(f"  RUNS.tsv: {n} rows ({upserts} cells upserted this pass), {len(cols)} columns")
    print(f"  RUNS_summary.tsv: {len(summary)} (data_type x threads x card) groups")
    flagged = [k for k, r in index.items()
               if r.get("engaged") != "OK" or r.get("gate_pass") not in ("TRUE", "FALSE")]
    if flagged:
        print(f"  flagged (engaged!=OK or gate_pass=NA), recorded but excluded from means: {len(flagged)}")
    for s in summary:
        print(f"    {s['data_type']:>8} {s.get('model_search','NA'):<22} T{s['threads']:<3} {s['card']:<12} "
              f"n={s['n_runs']} engaged={s['n_engaged']} "
              f"speedup={s['speedup_mean']}+/-{s['speedup_sd']} (n={s['speedup_n']}) "
              f"gate_pass_rate={s['gate_pass_rate']}")


if __name__ == "__main__":
    main()
