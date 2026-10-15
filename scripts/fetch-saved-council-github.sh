#!/usr/bin/env bash
set -euo pipefail
# Read-only source for already-committed saved historical council matrices.
# Requires git and free local disk. No Kaggle, API keys or artifact uploads.
YEAR="${1:?Usage: bash scripts/fetch-saved-council-github.sh YEAR CACHE_DIR}"
OUT="${2:?cache directory required}"
[[ "$YEAR" =~ ^202[1-5]$ ]] || { echo "2026 is sealed; year must be 2021..2025" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
ROOT="research-results"
RACE="$ROOT/l1-council-rebuild-v1/run-37183263875/races/y$YEAR.jsonl.gz"
BALLOTS="$ROOT/l16-outsider-support-safe-fast-v2/run-36724124925/y$YEAR/ballots.csv.gz"
dst1="$OUT/council-y$YEAR.jsonl.gz"
dst2="$OUT/outsider-ballots-y$YEAR.csv.gz"
if [[ -s "$dst1" && -s "$dst2" ]];then
  echo "ARCHIVED_SAVED_SCORE_SESSION_REUSE $YEAR $dst1 $dst2"
  exit 0
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git clone --quiet --depth=1 --filter=blob:none --sparse \
  --branch research/l1-council-rebuild-v1-20261004 \
  https://github.com/pinogame1945-dotcom/KEIBA-ML-.git "$tmp/repo"
cd "$tmp/repo"
python3 - "$RACE" "$BALLOTS" <<'PY'
import subprocess,sys
for path in sys.argv[1:]:
    data=subprocess.check_output(
        ["git","ls-tree","-r","-l","HEAD","--",path],text=True).strip()
    fields=data.split(None,4)
    if len(fields)!=5 or fields[4]!=path or fields[1]!="blob":
        raise SystemExit("ARCHIVE_INDEX_MISSING: "+path)
    try:size=int(fields[3])
    except ValueError:raise SystemExit("ARCHIVE_FILE_SIZE_UNKNOWN: "+path)
    if not 1<=size<=2_000_000:
        raise SystemExit("ARCHIVE_STORAGE_GUARD: "+path+" "+str(size))
    print("ARCHIVE_SOURCE_METADATA_OK",path,size)
PY
git sparse-checkout set --no-cone "/$RACE" "/$BALLOTS"
test -s "$RACE"
test -s "$BALLOTS"
# Copy to temp names first to avoid partial files treated as valid cache.
cp "$RACE" "$dst1.tmp.$$"
cp "$BALLOTS" "$dst2.tmp.$$"
mv "$dst1.tmp.$$" "$dst1"
mv "$dst2.tmp.$$" "$dst2"
echo "ARCHIVED_GITHUB_SAVED_SCORE_READY $YEAR $dst1 $dst2"
