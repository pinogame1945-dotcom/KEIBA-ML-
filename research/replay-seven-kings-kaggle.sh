#!/usr/bin/env bash
set -euo pipefail
# Replays seven EXISTING horse-level scores. Does not train or save datasets.
# Only free standard CPU + ephemeral runner-local storage.
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN is required}"
RACE="${1:?Usage: bash research/replay-seven-kings-kaggle.sh RACE_ID [OUTPUT_JSON]}"
OUT="${2:-out/seven-king-one-race-$RACE.json}"
[[ "$RACE" =~ ^[0-9]{12}$ ]] || { echo "Invalid canonical race id" >&2; exit 2; }
YEAR="${RACE:0:4}"
(( YEAR >= 2021 && YEAR <= 2025 )) || { echo "Only archived 2021..2025 OOF folds, 2026 sealed" >&2; exit 2; }
ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT

transient() {
  local label="$1";shift
  local try rc delay
  for try in 1 2 3 4 5;do
    local err="$ROOT/request.err"
    set +e
    "$@" 2>"$err"
    rc=$?
    set -e
    if (( rc == 0 ));then rm -f "$err";return 0;fi
    cat "$err" >&2
    if grep -Eqi '(^|[^0-9])404([^0-9]|$)|Not Found for url' "$err";then
      echo "::error::404 means metadata drift or missing dataset: $label" >&2;return "$rc"
    fi
    if ! grep -Eqi '429|502|503|504|timed out|connection reset|temporary failure|Bad Gateway|Service Unavailable|Gateway Timeout' "$err";then
      echo "::error::nontransient Kaggle error: $label" >&2;return "$rc"
    fi
    (( try < 5 )) || return "$rc"
    delay=$((2 ** try))
    echo "::warning::temporary Kaggle error $label retry=$try sleep=${delay}s" >&2
    sleep "$delay"
  done
}

ALIASES=(core4 pedlegacy condition full light pedv1 condrc)
ARGS=(--race "$RACE")
for alias in "${ALIASES[@]}";do
  ref="pino1945/keiba-l15-7k-$alias-y$YEAR-v1"
  base="$ROOT/$alias"
  mkdir -p "$base"
  files="$base/files.json"
  transient "list:$ref" bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" > "$2"' _ "$ref" "$files"
  remote="$(python3 - "$files" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
if not isinstance(rows,list):raise SystemExit("invalid Kaggle metadata")
names=[str(r["name"]) for r in rows]
match=[r for r in rows if r.get("name") in ("score.jsonl.gz","score.jsonl")]
if len(match)!=1:raise SystemExit("missing/ambiguous known score source in metadata")
try: size=int(match[0]["size"])
except (ValueError,TypeError,KeyError):raise SystemExit("unknown score file size; refuse storage risk")
if size<1 or size>100_000_000:raise SystemExit(f"score file not within 100 MB guard: {size}")
print(match[0]["name"])
PY
)"
  test -n "$remote"
  if [[ ! -s "$base/$remote" ]];then
    transient "download:$ref:$remote" kaggle datasets download "$ref" -f "$remote" -p "$base" --unzip --quiet
  fi
  test -s "$base/$remote" || { echo "::error::metadata drift: $remote absent after download" >&2; exit 1; }
  ARGS+=(--expert "$alias=$base/$remote")
  echo "ARCHIVE_SCORE_FOUND alias=$alias remote=$remote dataset=$ref"
done

python3 research/replay-seven-kings.py "${ARGS[@]}" --output "$OUT"
echo "ARCHIVED_SEVEN_KINGS_REPORT $OUT"
