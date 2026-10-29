#!/usr/bin/env bash
# iqtree3-gpu-verify.sh — user-friendly self-check for GPU-accelerated IQ-TREE 3.
#
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
# Runs a tiny amino-acid alignment through BOTH the CPU and the GPU likelihood, then checks two
# things in order:
#   1. ENGAGEMENT - the GPU kernel was actually installed (not just a device reported)
#   2. EQUIVALENCE - the GPU tree matches the CPU tree: RF = 0 and logL within 1e-4
# Both must hold. Engagement is checked FIRST because a log-likelihood match proves nothing on its
# own: if the kernel is not installed, both runs execute the same CPU code and agree trivially.
# This is a CORRECTNESS check, not a speed test - the demo alignment is far too small to time.
#
# Usage:   iqtree3-gpu-verify.sh [path/to/iqtree3] [alignment.faa]
# Requires: an IQ-TREE 3 built with -DIQTREE_GPU=ON, a CUDA GPU, and python3.
#           NO third-party python packages are needed - the checker is standard library only.
#           On O2: module load gcc/14.2.0 cuda/12.8 first.
set -uo pipefail

BIN="${1:-iqtree3}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALN="${2:-}"
CHECK="$DIR/rf_logl_check.py"   # stdlib-only RF + logL gate (no dendropy needed)
PY="${IQTREE_GPU_PY:-python3}"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

echo "==============================================================="
echo " GPU-accelerated IQ-TREE 3 — self-verification"
echo "==============================================================="
command -v "$BIN" >/dev/null 2>&1 || { echo "ERROR: iqtree3 not found: $BIN"; exit 1; }
echo " binary : $(command -v "$BIN")"
"$BIN" --version 2>&1 | head -1 | sed 's/^/ version: /'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/^/ GPU    : /'
else
    echo " GPU    : nvidia-smi not found (is a GPU allocated?)"
fi

# a tiny built-in 6-taxon AA alignment if the user did not supply one
if [ -z "$ALN" ]; then
    ALN="$WORK/demo.faa"
    cat > "$ALN" <<'FASTA'
>taxonA
MKAILVLTLAVLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF
>taxonB
MKAILVLSLAVLFSSAYSRGVFRRDAHKSEIAHRFNDLGEQHFKGLVLIAFSQYLQQCPF
>taxonC
MKGILVLTLAVLFSSAYSRGVFRRDTHKSEVAHRFKDLGEEHFKGLVLVAFSQYLQQCPF
>taxonD
MRAILVLTLAVLFSSAYSRGVYRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF
>taxonE
MKAILVLTLAVLLSSAYSRGVFRRDTHKGEIAHRFKDLGEEHFKGLVLIAFSQYLQHCPF
>taxonF
MKAILILTLAVLFSSAYSRGVFRRDTHKSEIAHRFKELGEEHFKGLVLIAFSQYLQQCPY
FASTA
    echo " input  : built-in 6-taxon demo alignment"
else
    echo " input  : $ALN"
fi
echo "---------------------------------------------------------------"

echo " [1/3] CPU run  (iqtree3 -m LG+G4) ..."
t0=$SECONDS; "$BIN" -s "$ALN" -m LG+G4 -T 1 -seed 1 --prefix "$WORK/cpu" -redo >/dev/null 2>&1; cpu_t=$((SECONDS-t0))
echo " [2/3] GPU run  (iqtree3 -gpu -m LG+G4) ..."
t0=$SECONDS
"$BIN" -s "$ALN" -m LG+G4 -T 1 -seed 1 -gpu --prefix "$WORK/gpu" -redo > "$WORK/gpu.stdout" 2>&1
gpu_t=$((SECONDS-t0))
grep -E "IQ-TREE3-GPU" "$WORK/gpu.stdout" | head -3

# ENGAGEMENT GATE. A log-likelihood/RF match proves nothing on its own: if the GPU kernel is never
# installed, both runs execute the SAME CPU kernel and agree trivially. A pointer swap placed after
# an early `return` in setLikelihoodKernel() is dead code, and a script that checks only the
# log-likelihood would still report PASS.
# "GPU acceleration ACTIVE" is printed only from inside the swap block, so it is the proof that the
# GPU kernel was actually installed. Absence of it = the run was CPU-only, whatever the logL says.
if ! grep -q "GPU acceleration ACTIVE" "$WORK/gpu.stdout"; then
    echo "---------------------------------------------------------------"
    echo " VERDICT: FAIL — the GPU kernel was NOT engaged."
    echo "   The '-gpu' run never printed 'GPU acceleration ACTIVE', so the likelihood"
    echo "   kernel was never swapped onto the device and this run used the CPU."
    if grep -q "no CUDA device is visible" "$WORK/gpu.stdout"; then
        echo "   Reason: no CUDA device visible to this process (check nvidia-smi / your GPU allocation)."
    else
        echo "   A log-likelihood match below would NOT indicate a working GPU build."
    fi
    exit 1
fi
echo " GPU engagement     : OK (kernel swap applied)"

# ACCOUNTING GATE. The banner proves the kernel was installed, not that it RAN: every entry point
# re-tests eligibility per call and plain -gpu delegates to the CPU on any device failure (for
# example a binary built for another compute capability: "no kernel image is available"). Such a run
# prints the banner, finishes on the CPU, and matches the CPU log-likelihood trivially. The GPUSTATS
# line at exit counts what actually happened, so require: calls on the device, none delegated, no
# device errors. This is the same rule the benchmark harness applies.
stats=$(grep -oE "GPUSTATS gpu_calls=.*" "$WORK/gpu.stdout" | tail -1)
gpu_calls=$(echo "$stats" | grep -oE "gpu_calls=[0-9]+" | cut -d= -f2)
cpu_calls=$(echo "$stats" | grep -oE "cpu_calls=[0-9]+" | cut -d= -f2)
dev_err=$(echo "$stats"   | grep -oE "device_errors=[0-9]+" | cut -d= -f2)
if [ -z "$stats" ] || [ "${gpu_calls:-0}" -eq 0 ] || [ "${cpu_calls:-1}" -ne 0 ] || [ "${dev_err:-1}" -ne 0 ]; then
    echo "---------------------------------------------------------------"
    echo " VERDICT: FAIL — the GPU kernel was installed but did NOT run the likelihood."
    echo "   exit accounting: ${stats:-<no GPUSTATS line>}"
    if grep -q "no kernel image is available" "$WORK/gpu.stdout"; then
        echo "   Reason: this binary was not built for this GPU's compute capability."
        echo "   Rebuild with the default multi-arch setting, or pass this card's architecture"
        echo "   (e.g. ./build_iqtree_gpu.sh ./iqtree3_gpu_build 75) -- see docs/DEPENDENCIES.md."
    else
        grep -E "^\[IQ-TREE3-GPU\] (likelihood calls|  delegated by reason)" "$WORK/gpu.stdout" | sed 's/^/   /'
    fi
    exit 1
fi
echo " GPU accounting     : OK (${gpu_calls} calls on the device, 0 delegated, 0 device errors)"

cll=$(grep -oE "Log-likelihood of the tree: *-?[0-9.]+" "$WORK/cpu.iqtree" | grep -oE "\-?[0-9.]+$")
gll=$(grep -oE "Log-likelihood of the tree: *-?[0-9.]+" "$WORK/gpu.iqtree" | grep -oE "\-?[0-9.]+$")
echo "---------------------------------------------------------------"
echo " CPU log-likelihood : $cll"
echo " GPU log-likelihood : $gll"

echo " [3/3] comparing trees (Robinson-Foulds + log-likelihood) ..."
# Use the stdlib-only checker: it computes RF with NO third-party dependency. The old path here
# required dendropy (absent on most systems, including this one) and silently degraded to
# `[ "$cll" = "$gll" ]` -- an EXACT STRING comparison of two floats. That is a bit-identity test,
# which is the wrong gate: GPU and CPU sum in different orders, so a correct build routinely differs
# in the last digits (a validated +R4 run here showed reldiff 1.7e-07). The correct criterion is
# the project's science gate: RF = 0 AND |logL| relative difference <= 1e-4.
if [ -f "$CHECK" ]; then
    "$PY" "$CHECK" "$WORK/cpu" "$WORK/gpu" demo
    rc=$?
    [ "$rc" -eq 0 ] || echo " VERDICT: FAIL — the GPU tree does not match the CPU tree."
else
    echo " VERDICT: SKIPPED — $CHECK not found; cannot verify equivalence."
    exit 1
fi
echo "---------------------------------------------------------------"
echo " wall time: CPU ${cpu_t}s   GPU ${gpu_t}s"
echo "   (This demo alignment is tiny, so the two times are expected to be similar — it checks"
echo "    correctness, not speed. Do not read a speedup from it.)"
echo " Done. If the verdict is PASS, the GPU build reproduces IQ-TREE exactly on this machine."
