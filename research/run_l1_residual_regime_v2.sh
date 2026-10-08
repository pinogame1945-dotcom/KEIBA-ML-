#!/usr/bin/env bash
set -euo pipefail

: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN required}"
DATASET_REF="${DATASET_REF:-pino1945/keiba-ml-snapshot-bf811fa2eab73db0}"
GENERATION_ID="${GENERATION_ID:-bf811fa2eab73db0}"
ROOT="out/l1-residual-regime-v2"
PROJ="$ROOT/projected"
RESULT="research-results/l1-residual-regime-v2-v1/run-${GITHUB_RUN_ID:?GITHUB_RUN_ID required}"
CPU="${L1_THREADS:-$(nproc)}"
DL_PARALLEL="${KAGGLE_DOWNLOAD_PARALLEL:-2}"
PROJ_PARALLEL="${PROJECTION_PARALLEL:-$CPU}"
(( DL_PARALLEL > 7 )) && DL_PARALLEL=7
(( DL_PARALLEL < 1 )) && DL_PARALLEL=1
(( PROJ_PARALLEL > 7 )) && PROJ_PARALLEL=7
(( PROJ_PARALLEL < 1 )) && PROJ_PARALLEL=1

export L1_THREADS="$CPU"
export OMP_NUM_THREADS="$CPU"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

echo "REGIME_V2_RUNTIME cpu=$CPU download_parallel=$DL_PARALLEL projection_parallel=$PROJ_PARALLEL"

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

wait_oldest() {
  local -n arr=$1
  wait "${arr[0]}"
  arr=("${arr[@]:1}")
}

rm -rf "$ROOT"
mkdir -p "$ROOT" "$PROJ" "$RESULT"

FILES="$ROOT/files.json"
kaggle_transient "files"   bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _   "$DATASET_REF" "$FILES"

python - "$FILES" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
names=[str(x.get("name") or "") for x in rows]
if "manifest.json" not in names:
    raise SystemExit("manifest.json missing from Kaggle metadata; metadata drift")
print("KAGGLE_METADATA_OK files=",len(names))
PY

kaggle_transient "download:manifest.json"   kaggle datasets download "$DATASET_REF" -f manifest.json -p "$ROOT" --unzip --quiet --force
test -s "$ROOT/manifest.json"

python research/resolve_snapshot_manifest_remote_v1.py   --manifest "$ROOT/manifest.json"   --files-json "$FILES"   --generation-id "$GENERATION_ID"   --output "$ROOT/requested.tsv"

download_pids=()
while IFS=$'\t' read -r year logical remote mode sha rows min_date max_date; do
  [[ -n "$remote" ]] || continue
  (
    kaggle_transient "download:$remote"       kaggle datasets download "$DATASET_REF" -f "$remote" -p "$ROOT" --unzip --quiet --force
    test -s "$ROOT/$remote"
    echo "SNAPSHOT_DOWNLOAD_OK year=$year remote=$remote"
  ) &
  download_pids+=("$!")
  if (( ${#download_pids[@]} >= DL_PARALLEL )); then wait_oldest download_pids; fi
done < "$ROOT/requested.tsv"
for pid in "${download_pids[@]}"; do wait "$pid"; done

python - "$ROOT/requested.tsv" "$ROOT" "$CPU" <<'PY'
import concurrent.futures,gzip,hashlib,json,os,sys
req,root,workers=sys.argv[1:]
items=[line.rstrip("\n").split("\t") for line in open(req,encoding="utf-8") if line.strip()]
def check(x):
    y,logical,remote,mode,sha,expected_rows,min_date,max_date=x
    path=os.path.join(root,remote)
    if remote.endswith(".gz") and logical==remote and sha:
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
        if h.hexdigest()!=sha: raise RuntimeError(f"sha256 mismatch {remote}")
    opener=gzip.open if remote.endswith(".gz") else open
    rows=0; lo=None; hi=None
    with opener(path,"rt",encoding="utf-8") as f:
        for raw in f:
            if not raw.strip(): continue
            row=json.loads(raw)
            d=str(row.get("race_date") or (row.get("features") or {}).get("race_date") or "")[:10]
            rows+=1
            if d:
                lo=d if lo is None or d<lo else lo
                hi=d if hi is None or d>hi else hi
    if expected_rows and rows!=int(expected_rows): raise RuntimeError(f"row mismatch {remote}")
    if min_date and lo!=min_date: raise RuntimeError(f"min_date mismatch {remote}")
    if max_date and hi!=max_date: raise RuntimeError(f"max_date mismatch {remote}")
    return f"SNAPSHOT_VALID_OK year={y} remote={remote} rows={rows}"
with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items),max(1,int(workers)))) as ex:
    for msg in ex.map(check,items): print(msg,flush=True)
PY

projection_pids=()
while IFS=$'\t' read -r year logical remote mode sha rows min_date max_date; do
  (
    node research/project-yearly-snapshots.mjs       --inputs "$ROOT/$remote"       --output "$PROJ/y$year.jsonl.gz"       --feature-sets BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,BACKFILL,AUTO,ACTOR,TIME_PACE       --prediction-phase FINAL
    test -s "$PROJ/y$year.jsonl.gz"
    echo "UNION_PROJECTION_OK year=$year"
  ) &
  projection_pids+=("$!")
  if (( ${#projection_pids[@]} >= PROJ_PARALLEL )); then wait_oldest projection_pids; fi
done < "$ROOT/requested.tsv"
for pid in "${projection_pids[@]}"; do wait "$pid"; done

while IFS=$'\t' read -r year logical remote mode sha rows min_date max_date; do
  rm -f "$ROOT/$remote"
done < "$ROOT/requested.tsv"

args=()
for year in 2019 2020 2021 2022 2023 2024 2025; do
  test -s "$PROJ/y$year.jsonl.gz"
  args+=(--year-file "$year:$PROJ/y$year.jsonl.gz")
done

python research/run_l1_residual_regime_v2.py   "${args[@]}"   --out-dir "$RESULT"

for f in summary.json fold-metrics.csv pooled-metrics.csv feature-importance.csv feature-counts.csv state-coverage.csv; do
  test -s "$RESULT/$f"
done

rm -rf "$ROOT"
echo "L1_RESIDUAL_REGIME_V2_DONE result=$RESULT"
