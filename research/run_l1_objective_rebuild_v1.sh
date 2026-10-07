#!/usr/bin/env bash
set -euo pipefail

: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN required}"
DATASET_REF="${DATASET_REF:-pino1945/keiba-ml-snapshot-bf811fa2eab73db0}"
GENERATION_ID="${GENERATION_ID:-bf811fa2eab73db0}"
SNAP_ROOT="out/l1-objective-snapshot"
RESULT_ROOT="research-results/l1-objective-rebuild-v1/run-${GITHUB_RUN_ID:?GITHUB_RUN_ID required}"

kaggle_transient() {
  local label="$1"; shift
  local attempt rc delay tmp
  for attempt in 1 2 3 4 5 6; do
    tmp="$(mktemp)"
    set +e
    "$@" >"$tmp.out" 2>"$tmp.err"
    rc=$?
    set -e
    cat "$tmp.out"
    cat "$tmp.err" >&2
    if [[ "$rc" -eq 0 ]]; then
      rm -f "$tmp" "$tmp.out" "$tmp.err"
      return 0
    fi
    if grep -Eqi '404 Client Error|Not Found for url' "$tmp.out" "$tmp.err"; then
      echo "::error::Kaggle 404 non-retryable label=$label" >&2
      rm -f "$tmp" "$tmp.out" "$tmp.err"
      return "$rc"
    fi
    if ! grep -Eqi '429|502|503|504|Too Many Requests|Bad Gateway|Service Unavailable|Gateway Timeout|timed out|connection reset|temporary failure' "$tmp.out" "$tmp.err"; then
      echo "::error::Kaggle non-transient failure label=$label rc=$rc" >&2
      rm -f "$tmp" "$tmp.out" "$tmp.err"
      return "$rc"
    fi
    rm -f "$tmp" "$tmp.out" "$tmp.err"
    [[ "$attempt" -lt 6 ]] || return "$rc"
    case "$attempt" in
      1) delay=10 ;;
      2) delay=20 ;;
      3) delay=40 ;;
      4) delay=60 ;;
      *) delay=90 ;;
    esac
    sleep "$delay"
  done
}

rm -rf "$SNAP_ROOT" out/l1-objective-proj
mkdir -p "$SNAP_ROOT" "$RESULT_ROOT"

# One metadata listing for the dataset. All subsequent downloads use only resolved real names.
FILES_JSON="$SNAP_ROOT/files.json"
kaggle_transient "files"   bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$DATASET_REF" "$FILES_JSON"

python - "$FILES_JSON" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
names=[str(x.get("name") or "") for x in rows]
if "manifest.json" not in names:
    raise SystemExit("manifest.json missing from Kaggle metadata; metadata drift")
print("KAGGLE_METADATA_OK files=",len(names))
PY

kaggle_transient "download:manifest.json"   kaggle datasets download "$DATASET_REF" -f manifest.json -p "$SNAP_ROOT" --unzip --quiet --force
test -s "$SNAP_ROOT/manifest.json"

python research/resolve_snapshot_manifest_remote_v1.py   --manifest "$SNAP_ROOT/manifest.json"   --files-json "$FILES_JSON"   --generation-id "$GENERATION_ID"   --output "$SNAP_ROOT/requested.tsv"

while IFS=$'\t' read -r year logical remote mode sha rows min_date max_date; do
  [[ -n "$remote" ]] || continue
  kaggle_transient "download:$remote"     kaggle datasets download "$DATASET_REF" -f "$remote" -p "$SNAP_ROOT" --unzip --quiet --force
  test -s "$SNAP_ROOT/$remote"
done < "$SNAP_ROOT/requested.tsv"

python - "$SNAP_ROOT/requested.tsv" "$SNAP_ROOT" <<'PY'
import gzip,hashlib,json,os,sys
req,root=sys.argv[1:]
for line in open(req,encoding="utf-8"):
    y,logical,remote,mode,expected_sha,expected_rows,min_date,max_date=line.rstrip("\n").split("\t")
    path=os.path.join(root,remote)
    if remote.endswith(".gz") and logical==remote and expected_sha:
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):
                h.update(block)
        if h.hexdigest()!=expected_sha:
            raise SystemExit(f"sha256 mismatch {remote}")
    opener=gzip.open if remote.endswith(".gz") else open
    rows=0
    actual_min=None
    actual_max=None
    with opener(path,"rt",encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            row=json.loads(raw)
            d=str(row.get("race_date") or (row.get("features") or {}).get("race_date") or "")[:10]
            rows+=1
            if d:
                actual_min=d if actual_min is None or d<actual_min else actual_min
                actual_max=d if actual_max is None or d>actual_max else actual_max
    if expected_rows and rows!=int(expected_rows):
        raise SystemExit(f"row mismatch {remote}: expected={expected_rows} actual={rows}")
    if min_date and actual_min!=min_date:
        raise SystemExit(f"min_date mismatch {remote}: expected={min_date} actual={actual_min}")
    if max_date and actual_max!=max_date:
        raise SystemExit(f"max_date mismatch {remote}: expected={max_date} actual={actual_max}")
    print(f"SNAPSHOT_METADATA_RESOLVED_OK year={y} logical={logical} remote={remote} mode={mode} rows={rows}")
PY

run_variant() {
  local variant="$1"
  local feature_sets="$2"
  local actor_prefixes="$3"
  local proj="out/l1-objective-proj/$variant"
  local result="$RESULT_ROOT/$variant"
  mkdir -p "$proj" "$result"
  local args=()
  while IFS=$'\t' read -r year logical remote mode sha rows min_date max_date; do
    local cmd=(
      node research/project-yearly-snapshots.mjs
      --inputs "$SNAP_ROOT/$remote"
      --output "$proj/y$year.jsonl.gz"
      --feature-sets "$feature_sets"
      --prediction-phase FINAL
    )
    if [[ -n "$actor_prefixes" ]]; then
      cmd+=(--actor-prefixes "$actor_prefixes")
    fi
    "${cmd[@]}"
    args+=(--year-file "$year:$proj/y$year.jsonl.gz")
  done < "$SNAP_ROOT/requested.tsv"

  python research/run_l1_objective_rebuild_v1.py     --variant "$variant"     "${args[@]}"     --out-dir "$result"

  rm -rf "$proj"
  rm -f "$result/oos-predictions.csv.gz"
}

run_variant   core4   BASE,OPPONENT,AUTO,ACTOR,TIME_PACE   ""

run_variant   structural   BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,BACKFILL,ACTOR,TIME_PACE   actor_jockey_,actor_trainer_

python - "$RESULT_ROOT" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for variant in ("core4","structural"):
    p=root/variant/"pooled-metrics.csv"
    with p.open(newline="",encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
fields=[]
for row in rows:
    for key in row:
        if key not in fields:
            fields.append(key)
with (root/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
summary={
    "contract":"L1_OBJECTIVE_REBUILD_V1_AGGREGATE",
    "question":"Does a new horse-only L1 learning objective materially improve strict walk-forward ranking?",
    "variants":{
        "core4":"BASE+OPPONENT+AUTO+ACTOR+TIME_PACE",
        "structural":"BASE+OPPONENT+NETWORK+LAP+STYLE+DISTANCE+BACKFILL+ACTOR(jockey,trainer)+TIME_PACE"
    },
    "heads":["WIN_BINARY","TOP3_BINARY","RANK_GRADED","BLEND_EQUAL","BLEND_TOP3_RANK","BLEND_WIN_RANK"],
    "walk_forward":"2 prior years -> next year, 2021-2025",
    "ability_uses_odds":False,
    "2026_locked":True,
    "promotion":False
}
(root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print((root/"comparison.csv").read_text())
PY

echo "L1_OBJECTIVE_REBUILD_V1_COMPLETE result=$RESULT_ROOT"
