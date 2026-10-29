#!/usr/bin/env bash
# fetch_benchmark_data.sh - download the benchmark input data from Zenodo into this clone.
#
#   bash benchmark/scripts/fetch_benchmark_data.sh alignments   # 35 amino-acid + 26 nucleotide alignments (~0.8 GB): what the harness needs
#   bash benchmark/scripts/fetch_benchmark_data.sh genomes      # the 435 source genomes (~0.3 GB): only to audit the alignments' provenance
#   bash benchmark/scripts/fetch_benchmark_data.sh all          # both
#
# Record: https://doi.org/10.5281/zenodo.22942159. Each archive unpacks to the repository path its
# name spells (benchmark_data_protein.tar.gz -> benchmark/data/protein/). Archives are verified
# against the record's ARCHIVES.sha256 before unpacking; one already present and correct is reused.
# Figures and tables regenerate from benchmark/runs/ without any of this (see benchmark/REPRODUCING.md).
set -euo pipefail
_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "$PWD") || { echo "FATAL: cannot locate repository root" >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"
cd "$REPO"

RECORD="${ZENODO_RECORD:-22942159}"
BASE="${ZENODO_FILES_URL:-https://zenodo.org/records/$RECORD/files}"   # override with file:///dir for an offline test
DL="$REPO/benchmark/data/.zenodo"; mkdir -p "$DL"
case "${1:-}" in
  alignments) SETS="benchmark_data_protein benchmark_data_dna" ;;
  genomes)    SETS="benchmark_data_genomes" ;;
  all)        SETS="benchmark_data_protein benchmark_data_dna benchmark_data_genomes" ;;
  *) sed -n 2,9p "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

get () {  # get NAME  -> $DL/NAME
  local name="$1" url="$BASE/$1"
  case "$url" in file://*) cp -f "${url#file://}" "$DL/$name" ;;
    *) url="$url?download=1"
       curl -fL --retry 3 -o "$DL/$name" "$url" || { echo "download failed: $url" >&2
         echo "If the response was 404, the Zenodo record is not public yet; ask the authors for an access link." >&2; exit 1; } ;;
  esac
}

echo "== $BASE =="
get supporting_files.tar.gz
tar -xzf "$DL/supporting_files.tar.gz" -C "$DL"
SUMS="$DL/supporting_files/ARCHIVES.sha256"
for s in $SETS; do
  a="$s.tar.gz"; want=$(awk -v f="$a" '$2==f{print $1}' "$SUMS")
  [ -n "$want" ] || { echo "no checksum for $a in ARCHIVES.sha256" >&2; exit 1; }
  if [ -f "$DL/$a" ] && [ "$(sha256sum "$DL/$a" | cut -d' ' -f1)" = "$want" ]; then
    echo "   $a: already downloaded, checksum ok"
  else
    echo "   downloading $a"; get "$a"
    [ "$(sha256sum "$DL/$a" | cut -d' ' -f1)" = "$want" ] || { echo "checksum mismatch: $a" >&2; exit 1; }
  fi
  echo "   unpacking $a -> $(echo "$s" | tr '_' '/')/"
  tar -xzf "$DL/$a" -C "$REPO"
done
echo "== verifying =="
python3 "$REPO/benchmark/scripts/verify_data.py"
