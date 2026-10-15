#!/usr/bin/env bash
set -euo pipefail
# Existing archived race -> 7 independently trained PREVIOUS-YEAR LightGBM
# binaries on strict original FINAL-phase yearly snapshots.
# Origin: research FINAL archive (pre-bet timestamps NOT independently verified).
# All data read-only, cached only in provided local session folder.
: "${KAGGLE_API_TOKEN:?Kaggle API read-only credentials required}"
raw="${1:?Usage: bash scripts/score-known-race-kaggle.sh NETKEIBA_RACE_ID_OR_URL [SESSION_ROOT] [OUTPUT_JSON]}"
RACE="$(python3 - "$raw" <<'PY'
import re,sys,urllib.parse
v=sys.argv[1].strip()
if v.startswith("https://"):
    u=urllib.parse.urlparse(v)
    if u.hostname not in ("race.netkeiba.com","db.netkeiba.com"):
        raise SystemExit("only official netkeiba links accepted")
    v=urllib.parse.parse_qs(u.query).get("race_id",[None])[0] or (
        re.search(r"/race/(\d{12})/?",u.path) or [None,None])[1]
if not isinstance(v,str) or not re.fullmatch(r"202[2-5]\d{8}",v):
    raise SystemExit("2022..2025 saved years only; 2026 sealed")
print(v)
PY
)"
year="${RACE:0:4}"
previous=$((10#$year-1))
session="${2:-${RUNNER_TEMP:-/tmp}/keiba-race-session}"
output="${3:-$session/results/$RACE-prior-year-kings.json}"
mkdir -p "$session/snapshot" "$session/models/y$previous" "$(dirname "$output")"
snap_ref="pino1945/keiba-ml-snapshot-eb13d4096519167b"

transient(){
  local label="$1"; shift
  local attempt rc
  for attempt in 1 2 3 4;do
    set +e
    "$@" 2>"$session/request.err"
    rc=$?
    set -e
    if [ "$rc" -eq 0 ];then rm -f "$session/request.err";return 0;fi
    cat "$session/request.err" >&2
    if grep -Eqi '404|Not Found for url' "$session/request.err";then
      echo "::error::Kaggle metadata drift/nonexistent remote, 404 abort $label" >&2
      return "$rc"
    fi
    if ! grep -Eqi '429|502|503|504|timed out|connection reset|temporary failure' "$session/request.err";then
      echo "::error::nontransient failure $label" >&2
      return "$rc"
    fi
    test "$attempt" -lt 4 || return "$rc"
    sleep $((2**attempt))
  done
}
snap="$session/snapshot"
if [ ! -s "$snap/files.json" ];then
  transient "snapshot:files" bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$snap_ref" "$snap/files.json"
fi
python3 - "$snap/files.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
manifest=[r for r in rows if r.get("name")=="manifest.json"]
if len(manifest)!=1 or not 1<=int(manifest[0]["size"])<=5000000:
    raise SystemExit("snapshot manifest file unavailable or too large")
PY
if [ ! -s "$snap/manifest.json" ];then
  transient "snapshot:manifest" kaggle datasets download "$snap_ref" -f manifest.json -p "$snap" --unzip --quiet
fi
test -s "$snap/manifest.json"
python3 research/resolve_snapshot_manifest_remote_v1.py \
  --manifest "$snap/manifest.json" --files-json "$snap/files.json" \
  --generation-id eb13d4096519167b --years "$year" \
  --output "$snap/resolved-$year.tsv"
remote="$(python3 - "$snap/resolved-$year.tsv" "$snap/files.json" <<'PY'
import json,sys
lines=open(sys.argv[1],encoding="utf-8").read().strip().splitlines()
if len(lines)!=1:raise SystemExit("year resolution must be unique")
name=lines[0].split("\t")[2]
match=[r for r in json.load(open(sys.argv[2],encoding="utf-8")) if r.get("name")==name]
if len(match)!=1 or not 1<=int(match[0]["size"])<=2300000000:
    raise SystemExit("unknown or over-cap snapshot")
print(name)
PY
)"
bytes="$(python3 - "$snap/files.json" "$remote" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
print(next(int(r["size"]) for r in rows if r["name"]==sys.argv[2]))
PY
)"
available_kb="$(df -Pk "$snap" | tail -1 | awk '{print $4}')"
test "$available_kb" -gt 9000000 || { echo "::error::free runner ephemeral disk guard (<9GB), no download" >&2;exit 1; }
# Cache only when actual metadata size is identical. Never choose a path by guessing.
if [ ! -s "$snap/$remote" ] || [ "$(stat -c%s "$snap/$remote")" != "$bytes" ];then
  transient "snapshot:$remote" kaggle datasets download "$snap_ref" -f "$remote" -p "$snap" --unzip --quiet
fi
test -s "$snap/$remote"
if [ "${KEIBA_PREPARE_ONLY:-0}" = "1" ];then
  echo "BATCH_SNAPSHOT_PREPARED $year $snap/$remote"
  exit 0
fi
small="$session/races/$RACE/features.jsonl"
if [ ! -s "$small" ];then
  python3 research/extract-race-snapshot.py --input "$snap/$remote" --race "$RACE" --output "$small"
fi

for alias in core4 pedlegacy condition full light pedv1 condrc;do
  ref="pino1945/keiba-l15-7k-$alias-y$previous-v1"
  d="$session/models/y$previous/$alias"
  mkdir -p "$d"
  if [ ! -s "$d/files.json" ];then
    transient "models:files:$alias" bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$ref" "$d/files.json"
  fi
  python3 - "$d/files.json" "$alias" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
for name,limit in (("model.txt",10000000),("model.json",3000000),("schema.json",3000000)):
    found=[r for r in rows if r.get("name")==name]
    if len(found)!=1 or not 1<=int(found[0]["size"])<=limit:
        raise SystemExit("missing/over-limit source model "+sys.argv[2]+" "+name)
PY
  for filename in model.txt model.json schema.json;do
    expected_bytes="$(python3 - "$d/files.json" "$filename" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding='utf-8'))
print(next(int(x['size']) for x in rows if x['name']==sys.argv[2]))
PY
)"
    if [ ! -s "$d/$filename" ] || [ "$(stat -c%s "$d/$filename" 2>/dev/null || echo 0)" != "$expected_bytes" ];then
      transient "model:$alias:$filename" kaggle datasets download "$ref" -f "$filename" -p "$d" --unzip --quiet
    fi
    test -s "$d/$filename" || { echo "::error::metadata drift $alias $filename"; exit 1; }
  done
done
python3 research/predict-prior-year-seven-kings.py \
  --snapshot "$small" --bundles "$session/models/y$previous" \
  --race "$RACE" --output "$output"
echo "PRIOR_YEAR_KING_SINGLE_RACE_REPORT $output"
