#!/usr/bin/env bash
# Install the large benchmark inputs into an IQ-TREE3-GPU clone, then verify them.
#   bash install_into_repo.sh /path/to/iqtree-gpu
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="${1:-}"
if [ -z "$REPO" ] || [ ! -d "$REPO/benchmark/data/dna" ]; then
  echo "usage: bash install_into_repo.sh /path/to/iqtree-gpu" >&2
  echo "  (the target must be a clone containing benchmark/data/dna/)" >&2
  exit 2
fi
REPO="$(cd "$REPO" && pwd)"
echo "== installing large benchmark inputs into $REPO =="
copied=0; skipped=0
for src in "$HERE"/data/dna/*/aln.phy; do
  set_name=$(basename "$(dirname "$src")")
  dst="$REPO/benchmark/data/dna/$set_name/aln.phy"
  mkdir -p "$(dirname "$dst")"
  if [ -f "$dst" ] && [ "$(sha256sum < "$dst" | cut -d' ' -f1)" = "$(sha256sum < "$src" | cut -d' ' -f1)" ]; then
    printf "  present and correct  %s\n" "$set_name"; skipped=$((skipped+1)); continue
  fi
  cp "$src" "$dst" && { printf "  installed            %s\n" "$set_name"; copied=$((copied+1)); }
  # the recipe and guide tree ship in the repo already, but copy them if absent
  for extra in PROVENANCE.md aln.treefile; do
    [ -f "$HERE/data/dna/$set_name/$extra" ] && [ ! -f "$REPO/benchmark/data/dna/$set_name/$extra" ] \
      && cp "$HERE/data/dna/$set_name/$extra" "$REPO/benchmark/data/dna/$set_name/"
  done
done
echo "== $copied installed, $skipped already correct =="
echo "== verifying every alignment against CHECKSUMS.sha256 =="
( cd "$REPO" && sha256sum -c benchmark/data/dna/CHECKSUMS.sha256 ) | tail -20
