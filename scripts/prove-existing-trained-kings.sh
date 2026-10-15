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
python3 research/resolve_snapshot_manifest_remote_v1.py \
  --manifest "$ROOT/snapshot/manifest.json" --files-json "$META" \
  --generation-id eb13d4096519167b --years "$YEAR" \
  --output "$ROOT/snapshot/resolved.tsv"
remote="$(python3 - "$ROOT/snapshot/resolved.tsv" "$META" <<'PY'
import json,sys
from pathlib import Path
lines=Path(sys.argv[1]).read_text(encoding="utf-8").strip().splitlines()
if len(lines)!=1:raise SystemExit("metadata resolver must identify exactly one year")
fields=lines[0].split("\t")
if len(fields)!=8:raise SystemExit("invalid manifest resolution row")
name=fields[2]
files=json.load(open(sys.argv[2],encoding="utf-8"))
rows=[row for row in files if row.get("name")==name]
if len(rows)!=1:raise SystemExit("snapshot resolved name not uniquely in Kaggle metadata")
try:size=int(rows[0]["size"])
except (TypeError,ValueError):raise SystemExit("snapshot size missing")
if not 1<=size<=2300000000:raise SystemExit("snapshot exceeds ephemeral free runner quota: "+str(size))
print(name)
PY
)"
# The exact 2025 snapshot is 2.12 GB, and will be read only once.
# Refuse unless runner-local included ephemeral disk is clearly sufficient.
available_kb=$(df -Pk "$ROOT" | tail -1 | awk '{print $4}')
test "$available_kb" -gt 9000000 || { echo "::error::insufficient free ephemeral disk for known 2.12GB 2025 file, stop before download";exit 1; }
echo "SNAPSHOT_EXACT_REMOTE_SELECTED $remote"
transient "snapshot:year:$YEAR" kaggle datasets download "$SNAP_REF" -f "$remote" -p "$ROOT/snapshot" --unzip --quiet
test -s "$ROOT/snapshot/$remote"
python3 research/extract-race-snapshot.py \
  --input "$ROOT/snapshot/$remote" --race 202507010212 \
  --output "$ROOT/one-race-features.jsonl"
# Compact original Council matrix contains all seven corrected rankings
# for the same race, so downloading ~700MB of full-year score.jsonl is wasteful.
COUNCIL_PATH="research-results/l1-council-rebuild-v1/run-37183263875/races/y2025.jsonl.gz"
python3 scripts/github-readonly-archive.py \
  --repo pinogame1945-dotcom/KEIBA-ML- \
  --ref research/l1-council-rebuild-v1-20261004 \
  --path "$COUNCIL_PATH" \
  --dest "$ROOT/council-y2025.jsonl.gz" --max-bytes 2000000
for alias in core4 pedlegacy condition full light pedv1 condrc;do
  ref="pino1945/keiba-l15-7k-$alias-y$YEAR-v1"
  d="$ROOT/kings/$alias"
  mkdir -p "$d"
  transient "king:list:$alias" bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$ref" "$d/files.json"
  python3 - "$d/files.json" "$alias" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
expected={"model.txt":10000000,"model.json":3000000,"schema.json":3000000,
          "candidate.json":500000}
for name,limit in expected.items():
    match=[r for r in rows if r.get("name")==name]
    if len(match)!=1:raise SystemExit("king remote absent in metadata "+sys.argv[2]+" "+name)
    try:size=int(match[0]["size"])
    except (ValueError,TypeError):raise SystemExit("unknown size "+name)
    if not 1<=size<=limit:raise SystemExit("file cap exceeded "+name)
    print("KING_REAL_REMOTE",sys.argv[2],name,size)
PY
  for name in model.txt model.json schema.json candidate.json;do
    test ! -s "$d/$name" || continue
    transient "king:download:$alias:$name" kaggle datasets download "$ref" -f "$name" -p "$d" --unzip --quiet
    test -s "$d/$name" || { echo "::error::metadata drift after download $alias $name" >&2; exit 1; }
  done
done
python3 research/verify-trained-kings-vs-saved-scores.py \
  --snapshot "$ROOT/one-race-features.jsonl" \
  --bundles "$ROOT/kings" --race 202507010212 \
  --council-file "$ROOT/council-y2025.jsonl.gz"
# Separately prove a model selection timeline with no 2025 labels: each
# 2024 model was trained on 2022-23 and early-stopped on 2024, *before*
# the 2025 target race. Historical FINAL input is still NOT timestamped live.
mkdir -p "$ROOT/previous-year-kings"
for alias in core4 pedlegacy condition full light pedv1 condrc;do
  year=2024
  ref="pino1945/keiba-l15-7k-$alias-y$year-v1"
  d="$ROOT/previous-year-kings/$alias"
  mkdir -p "$d"
  transient "prev:list:$alias" bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$ref" "$d/files.json"
  python3 - "$d/files.json" "$alias" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
for name,limit in (("model.txt",10000000),("model.json",3000000),("schema.json",3000000)):
    candidates=[r for r in rows if r.get("name")==name]
    if len(candidates)!=1:raise SystemExit("missing previous-year model metadata "+sys.argv[2]+" "+name)
    try:size=int(candidates[0]["size"])
    except (TypeError,ValueError):raise SystemExit("missing previous-year model file size")
    if not 1<=size<=limit:raise SystemExit("previous-year model size cap exceeded")
    print("PREVIOUS_YEAR_MODEL_REMOTE",sys.argv[2],name,size)
PY
  for name in model.txt model.json schema.json;do
    test ! -s "$d/$name" || continue
    transient "prev:download:$alias:$name" kaggle datasets download "$ref" -f "$name" -p "$d" --unzip --quiet
    test -s "$d/$name" || { echo "::error::download metadata drift $alias $name"; exit 1; }
  done
done
python3 research/predict-prior-year-seven-kings.py \
  --snapshot "$ROOT/one-race-features.jsonl" \
  --bundles "$ROOT/previous-year-kings" --race 202507010212 \
  --output "$ROOT/previous-year-new-prediction.json"
# Join newly computed 7-way 2024-model scores to the previously proven
# historical saved 7King + 13 Outsider + L1.7 OOS signal on SAME 16 horses.
# BACKFILL has only two real days for integration smoke; standalone prior-year
# model scores above use the original 2025 full-history FINAL snapshot.
python3 scripts/prepare-proven-real-race.py "$ROOT/backfill"
python3 research/analyze-one-race.py \
  --race 202507010212 --source-root "$ROOT/backfill" --history-from 2025-01-05 \
  --github-saved --prior-year-models "$ROOT/previous-year-new-prediction.json" \
  --require-scored --output-dir "$ROOT/integrated"
python3 - "$ROOT/integrated/analysis.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r["race_id"]=="202507010212" and r["starters"]==16
assert r["saved_seven_kings"]["status"]=="GITHUB_ARCHIVED_7KING_REPLAY"
assert r["saved_seven_kings"]["outsider_model_count"]==13
assert r["saved_l17_podium"]["status"]=="ARCHIVED_OOS_P3"
assert r["prior_year_new_model_inference"]["status"]=="ORIGINAL_MODELS_SCORED_NOT_VERIFIED_PREBET"
assert len(r["horses"])==16
for h in r["horses"]:
    assert len(h["prior_year_king_ranks"])==7
    assert len(h["saved_seven_king_ranks"])==7
    assert "saved_outsider_points" in h
print("ALL_REGISTERED_EXISTING_AND_PRIOR_YEAR_TRAINED_MODELS_INTEGRATED_PASS",
      r["race_id"],r["starters"],"7 prior-year new inferences",
      "7 original archived rankings","13 outsider ballots","10 L1.7 OOS probabilities")
for h in sorted(r["horses"],key=lambda a:a["prior_year_king_rank_sum"])[:6]:
    print("NEW_PRIOR_YEAR_RANK",h["prior_year_king_rank_sum"],
          h.get("horse_number"),h.get("horse_name"),h["horse_id"],
          "archived_original_rank",h.get("archived_rank_sum_comparator_rank"))
PY

