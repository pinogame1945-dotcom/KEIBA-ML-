#!/usr/bin/env bash
set -euo pipefail
# Strict read-only session. One yearly snapshot reused for all seven models.
# This proves original research model reproducibility, not blind pre-race ASOF.
: "${KAGGLE_API_TOKEN:?Kaggle read-only access required}"
YEAR=2025
SNAP_REF="pino1945/keiba-ml-snapshot-eb13d4096519167b"
ROOT="${1:?Usage: bash scripts/prove-existing-trained-kings.sh EPHEMERAL_WORKSPACE}"
mkdir -p "$ROOT/snapshot" "$ROOT/kings"
META="$ROOT/snapshot/files.json"

transient(){
  local label="$1";shift
  local i err rc
  for i in 1 2 3 4;do
    err="$ROOT/request.err"
    set +e
    "$@" 2>"$err"
    rc=$?
    set -e
    if [ "$rc" -eq 0 ];then rm -f "$err"; return 0;fi
    cat "$err" >&2
    if grep -Eqi '404|Not Found for url' "$err";then
      echo "::error::404 metadata drift $label; no retry" >&2; return "$rc"
    fi
    if ! grep -Eqi '429|502|503|504|timed out|connection reset|temporary failure' "$err";then
      echo "::error::nontransient error $label; stop" >&2; return "$rc"
    fi
    test "$i" -lt 4 || return "$rc"
    sleep $((2**i))
  done
}

transient "snapshot:list" bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$SNAP_REF" "$META"
python3 - "$META" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
found=[r for r in rows if r.get("name")=="manifest.json"]
if len(found)!=1:raise SystemExit("snapshot manifest not found in metadata")
if not 1<=int(found[0]["size"])<=5000000:raise SystemExit("snapshot manifest size guard")
print("SNAPSHOT_MANIFEST_REMOTE_RESOLVED",found[0]["name"],found[0]["size"])
PY
transient "snapshot:manifest" kaggle datasets download "$SNAP_REF" -f manifest.json -p "$ROOT/snapshot" --unzip --quiet
test -s "$ROOT/snapshot/manifest.json"
remote="$(python3 - "$ROOT/snapshot/manifest.json" "$META" "$YEAR" <<'PY'
import json,sys
manifest=json.load(open(sys.argv[1],encoding="utf-8"))
files=json.load(open(sys.argv[2],encoding="utf-8"))
year=int(sys.argv[3])
if manifest.get("generation",{}).get("generation_id")!="eb13d4096519167b":
    raise SystemExit("snapshot generation contract drift")
found=[x for x in manifest.get("years",[]) if int(x.get("year",-1))==year]
if len(found)!=1:raise SystemExit("snapshot year not unique")
formal=str(found[0].get("file") or "")
listed=[x for x in files if x.get("name")==formal]
if len(listed)!=1:raise SystemExit("snapshot formal file absent from metadata")
raw=listed[0].get("size")
try:size=int(raw)
except (TypeError,ValueError):raise SystemExit("unknown snapshot file size")
if not 1<=size<=550000000:raise SystemExit("snapshot too large for one ephemeral free runner budget: "+str(size))
if not formal.endswith((".jsonl",".jsonl.gz")):raise SystemExit("invalid manifest snapshot format")
print(formal)
PY
)"
echo "SNAPSHOT_EXACT_REMOTE_SELECTED $remote"
transient "snapshot:year:$YEAR" kaggle datasets download "$SNAP_REF" -f "$remote" -p "$ROOT/snapshot" --unzip --quiet
test -s "$ROOT/snapshot/$remote"
for alias in core4 pedlegacy condition full light pedv1 condrc;do
  ref="pino1945/keiba-l15-7k-$alias-y$YEAR-v1"
  d="$ROOT/kings/$alias"
  mkdir -p "$d"
  transient "king:list:$alias" bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$ref" "$d/files.json"
  python3 - "$d/files.json" "$alias" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
expected={"model.txt":10000000,"model.json":3000000,"schema.json":3000000,
          "candidate.json":500000,"score.jsonl.gz":100000000}
for name,limit in expected.items():
    match=[r for r in rows if r.get("name")==name]
    if len(match)!=1:raise SystemExit("king remote absent in metadata "+sys.argv[2]+" "+name)
    try:size=int(match[0]["size"])
    except (ValueError,TypeError):raise SystemExit("unknown size "+name)
    if not 1<=size<=limit:raise SystemExit("file cap exceeded "+name)
    print("KING_REAL_REMOTE",sys.argv[2],name,size)
PY
  for name in model.txt model.json schema.json candidate.json score.jsonl.gz;do
    test ! -s "$d/$name" || continue
    transient "king:download:$alias:$name" kaggle datasets download "$ref" -f "$name" -p "$d" --unzip --quiet
    test -s "$d/$name" || { echo "::error::metadata drift after download $alias $name" >&2; exit 1; }
  done
done
python3 research/verify-trained-kings-vs-saved-scores.py \
  --snapshot "$ROOT/snapshot/$remote" \
  --bundles "$ROOT/kings" --race 202507010212
