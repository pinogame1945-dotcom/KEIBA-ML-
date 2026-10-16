#!/usr/bin/env bash
set -euo pipefail
# Official existing historical score sources, verified against GitHub Contents metadata.
# No Git clone and no paid storage; 404 is fatal, no alternative filenames.
YEAR="${1:?Usage: bash scripts/fetch-saved-council-github.sh YEAR CACHE_DIR}"
OUT="${2:?cache directory required}"
[[ "$YEAR" =~ ^202[1-5]$ ]] || { echo "2026 sealed; only 2021..2025" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
ROOT="research-results"
RACE="$ROOT/l1-council-rebuild-v1/run-37183263875/races/y$YEAR.jsonl.gz"
BALLOTS="$ROOT/l16-outsider-support-safe-fast-v2/run-36724124925/y$YEAR/ballots.csv.gz"
REF="research/l1-council-rebuild-v1-20261004"
REPO="pinogame1945-dotcom/KEIBA-ML-"
dst1="$OUT/council-y$YEAR.jsonl.gz"
dst2="$OUT/outsider-ballots-y$YEAR.csv.gz"
# Each call lists exact GitHub metadata before downloading. Any local cache
# is verified by its Git blob digest and source size; corrupt local files are replaced.
python3 scripts/github-readonly-archive.py \
  --repo "$REPO" --ref "$REF" --path "$RACE" --dest "$dst1" --max-bytes 2000000
python3 scripts/github-readonly-archive.py \
  --repo "$REPO" --ref "$REF" --path "$BALLOTS" --dest "$dst2" --max-bytes 2000000
test -s "$dst1" && test -s "$dst2"
echo "ARCHIVED_GITHUB_SAVED_SCORE_READY $YEAR $dst1 $dst2"
