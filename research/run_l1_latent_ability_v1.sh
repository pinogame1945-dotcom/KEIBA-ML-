#!/usr/bin/env bash
set -euo pipefail

: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN required}"
DATASET_REF="${DATASET_REF:-pino1945/keiba-ml-snapshot-bf811fa2eab73db0}"
GENERATION_ID="${GENERATION_ID:-bf811fa2eab73db0}"
ROOT="out/l1-latent-ability"
RESULT_ROOT="research-results/l1-latent-ability-v1/run-${GITHUB_RUN_ID:?GITHUB_RUN_ID required}"
CPU="${L1_THREADS:-$(nproc)}"
DL_PARALLEL="${KAGGLE_DOWNLOAD_PARALLEL:-2}"
(( DL_PARALLEL > 7 )) && DL_PARALLEL=7
(( DL_PARALLEL < 1 )) && DL_PARALLEL=1

export L1_THREADS="$CPU"
export OMP_NUM_THREADS="$CPU"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

echo "LATENT_RUNTIME cpu=$CPU download_parallel=$DL_PARALLEL"

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
  local pid="${arr[0]}"
  wait "$pid"
  arr=("${arr[@]:1}")
}

rm -rf "$ROOT"
mkdir -p "$ROOT" "$RESULT_ROOT"

FILES_JSON="$ROOT/files.json"
kaggle_transient "files"   bash -c 'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _   "$DATASET_REF" "$FILES_JSON"

python - "$FILES_JSON" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
names=[str(x.get("name") or "") for x in rows]
if "manifest.json" not in names:
    raise SystemExit("manifest.json missing from Kaggle metadata; metadata drift")
print("KAGGLE_METADATA_OK files=",len(names))
PY

kaggle_transient "download:manifest.json"   kaggle datasets download "$DATASET_REF" -f manifest.json -p "$ROOT" --unzip --quiet --force
test -s "$ROOT/manifest.json"

python research/resolve_snapshot_manifest_remote_v1.py   --manifest "$ROOT/manifest.json"   --files-json "$FILES_JSON"   --generation-id "$GENERATION_ID"   --output "$ROOT/requested.tsv"

download_pids=()
while IFS=$'\t' read -r year logical remote mode sha rows min_date max_date; do
  [[ -n "$remote" ]] || continue
  (
    kaggle_transient "download:$remote"       kaggle datasets download "$DATASET_REF" -f "$remote" -p "$ROOT" --unzip --quiet --force
    test -s "$ROOT/$remote"
    echo "SNAPSHOT_DOWNLOAD_OK year=$year remote=$remote"
  ) &
  download_pids+=("$!")
  if (( ${#download_pids[@]} >= DL_PARALLEL )); then
    wait_oldest download_pids
  fi
done < "$ROOT/requested.tsv"
for pid in "${download_pids[@]}"; do
  wait "$pid"
done

python - "$ROOT/requested.tsv" "$ROOT" "$CPU" <<'PY'
import concurrent.futures,gzip,hashlib,json,os,sys
req,root,workers=sys.argv[1:]
items=[line.rstrip("\n").split("\t") for line in open(req,encoding="utf-8") if line.strip()]

def check(fields):
    y,logical,remote,mode,expected_sha,expected_rows,min_date,max_date=fields
    path=os.path.join(root,remote)
    if remote.endswith(".gz") and logical==remote and expected_sha:
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):
                h.update(block)
        if h.hexdigest()!=expected_sha:
            raise RuntimeError(f"sha256 mismatch {remote}")
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
        raise RuntimeError(f"row mismatch {remote}: expected={expected_rows} actual={rows}")
    if min_date and actual_min!=min_date:
        raise RuntimeError(f"min_date mismatch {remote}: expected={min_date} actual={actual_min}")
    if max_date and actual_max!=max_date:
        raise RuntimeError(f"max_date mismatch {remote}: expected={max_date} actual={actual_max}")
    return f"SNAPSHOT_VALID_OK year={y} logical={logical} remote={remote} mode={mode} rows={rows}"

with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(items),max(1,int(workers)))) as ex:
    for msg in ex.map(check,items):
        print(msg,flush=True)
PY

args=()
while IFS=$'\t' read -r year logical remote mode sha rows min_date max_date; do
  args+=(--year-file "$year:$ROOT/$remote")
done < "$ROOT/requested.tsv"

python research/run_l1_latent_ability_v1.py   "${args[@]}"   --out-dir "$RESULT_ROOT"

test -s "$RESULT_ROOT/summary.json"
test -s "$RESULT_ROOT/fold-metrics.csv"
test -s "$RESULT_ROOT/pooled-metrics.csv"

rm -rf "$ROOT"
echo "L1_LATENT_ABILITY_V1_DONE result=$RESULT_ROOT"
