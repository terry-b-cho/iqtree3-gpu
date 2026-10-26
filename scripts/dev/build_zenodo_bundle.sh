#!/usr/bin/env bash
# =============================================================================
#  Build the Zenodo deposit folder for the large benchmark inputs.
#
#  14 simulated DNA alignments total ~2.4 GB and 11 exceed GitHub's hard 100 MB per-file limit, so
#  they cannot ship in the repository. This assembles them into ONE self-contained folder that can be
#  uploaded to Zenodo as a unit and installed into any clone with the included installer.
#
#  Uses hard links, so it costs no additional disk on the same filesystem.
#  Output is gitignored: it is a staging area, not a tracked artifact.
#
#  Usage: bash scripts/dev/build_zenodo_bundle.sh [version]
# =============================================================================
# Paths: the repository root is resolved at run time by scripts/lib/paths.sh. Nothing is hardcoded.
set -uo pipefail
_i3g_try () { local d="${1:-}"; [ -n "$d" ] || return 1; d=$(cd "$d" 2>/dev/null && pwd) || return 1
  while [ "$d" != "/" ]; do [ -f "$d/scripts/lib/paths.sh" ] && { printf '%s' "$d"; return 0; }
    d=$(dirname "$d"); done; return 1; }
_d=$(_i3g_try "${REPO_DIR:-}") || _d=$(_i3g_try "$(dirname "${BASH_SOURCE[0]:-$0}")") \
  || _d=$(_i3g_try "$PWD") || { echo "FATAL: cannot locate repository root" >&2; exit 78; }
. "$_d/scripts/lib/paths.sh"
cd "$REPO" || exit 1

VER="${1:-$(cat VERSION 2>/dev/null || echo 0.1.0-dev)}"
B="zenodo_upload_bundle/iqtree3-gpu-benchmark-inputs-v${VER}"
rm -rf zenodo_upload_bundle; mkdir -p "$B/data/dna"

echo "== staging alignments too large for GitHub =="
n=0
for f in data/dna/*/aln.phy; do
  # an alignment belongs in the deposit exactly when git refuses to track it
  git -C "$REPO" check-ignore -q "$f" || continue
  set_name=$(basename "$(dirname "$f")")
  mkdir -p "$B/data/dna/$set_name"
  ln "$f" "$B/data/dna/$set_name/aln.phy" 2>/dev/null || cp "$f" "$B/data/dna/$set_name/aln.phy"
  for extra in PROVENANCE.md aln.treefile; do
    [ -f "data/dna/$set_name/$extra" ] && cp "data/dna/$set_name/$extra" "$B/data/dna/$set_name/"
  done
  printf "  %-20s %s\n" "$set_name" "$(du -h "$f" | cut -f1)"
  n=$((n+1))
done
cp data/dna/CHECKSUMS.sha256 "$B/" 2>/dev/null

# README + installer are written by hand and kept beside this script's output; regenerate them only
# if absent so local edits survive a rebuild.
for asset in README.md install_into_repo.sh; do
  if [ -f "scripts/dev/zenodo_assets/$asset" ]; then cp "scripts/dev/zenodo_assets/$asset" "$B/"; fi
done
chmod +x "$B/install_into_repo.sh" 2>/dev/null

echo "== $n alignments staged into $B =="
echo "   apparent size: $(du -sh --apparent-size "$B" | cut -f1)   (hard-linked: no extra disk)"
echo "   upload the folder '$(basename "$B")' to Zenodo as a single deposit,"
echo "   then put the DOI into REPRODUCING.md and USER_GUIDE.md."
