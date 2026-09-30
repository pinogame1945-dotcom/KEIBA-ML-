#!/usr/bin/env bash
set -euo pipefail

cmd="${1:?usage: kaggle-snapshot-session.sh init|fetch GENERATION_ID OUT_DIR [YEAR...]}"
generation_id="${2:?generation id required}"
out_dir="${3:?out dir required}"
shift 3

slug="keiba-ml-snapshot-${generation_id}"
manifest="$out_dir/manifest.json"
dataset_ref_file="$out_dir/.dataset-ref"
files_json="$out_dir/.files.json"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

mkdir -p "$out_dir"

retry_kaggle() {
  local label="$1"
  shift
  local attempt delay rc tmp
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
      echo "::error::Kaggle 404 is non-retryable label=$label" >&2
      rm -f "$tmp" "$tmp.out" "$tmp.err"
      return "$rc"
    fi
    if ! grep -Eqi '429|502|503|504|Too Many Requests|Bad Gateway|Service Unavailable|Gateway Timeout|timed out|connection reset|temporary failure' "$tmp.out" "$tmp.err"; then
      echo "::error::Kaggle non-transient failure label=$label rc=$rc" >&2
      rm -f "$tmp" "$tmp.out" "$tmp.err"
      return "$rc"
    fi
    rm -f "$tmp" "$tmp.out" "$tmp.err"
    if [[ "$attempt" -eq 6 ]]; then
      echo "::error::Kaggle transient operation failed label=$label after $attempt attempts" >&2
      return "$rc"
    fi
    case "$attempt" in
      1) delay=15 ;;
      2) delay=30 ;;
      3) delay=60 ;;
      4) delay=90 ;;
      *) delay=120 ;;
    esac
    echo "::warning::Kaggle transient retry label=$label attempt=$attempt/6 sleep=${delay}s" >&2
    sleep "$delay"
  done
}
resolve_ref() {
  if [[ -n "${KAGGLE_DATASET_REF:-}" ]]; then
    local ref="$KAGGLE_DATASET_REF"
    [[ "$ref" == */"$slug" ]] || { echo "dataset ref mismatch: $ref" >&2; exit 3; }
    printf '%s' "$ref"
    return
  fi
  echo "KAGGLE_DATASET_REF is required for session mode" >&2
  exit 3
}

validate_one() {
  local path="$1" manifest_name="$2"
  python - "$manifest" "$path" "$manifest_name" <<'PY'
import gzip,hashlib,json,sys
manifest,path,name=sys.argv[1:]
m=json.load(open(manifest,encoding="utf-8"))
by_name={str(y.get("file") or ""):y for y in m.get("years",[])}
if name not in by_name:
    raise SystemExit(f"{name} not in manifest")
expected=by_name[name]
if path.endswith(".gz"):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    if h.hexdigest()!=expected["sha256"]:
        raise SystemExit(f"sha256 mismatch: {path}")
    opener=lambda: gzip.open(path,"rt",encoding="utf-8")
else:
    opener=lambda: open(path,"rt",encoding="utf-8")
rows=0
min_date=None
max_date=None
with opener() as f:
    for line in f:
        if not line.strip():
            continue
        row=json.loads(line)
        date=str(row.get("race_date") or (row.get("features") or {}).get("race_date") or "")[:10]
        rows+=1
        if date:
            min_date=date if min_date is None or date<min_date else min_date
            max_date=date if max_date is None or date>max_date else max_date
if rows!=int(expected["rows"]):
    raise SystemExit(f"row mismatch {path}: expected={expected['rows']} actual={rows}")
if min_date!=expected["min_date"] or max_date!=expected["max_date"]:
    raise SystemExit(f"date mismatch {path}: expected={expected['min_date']}..{expected['max_date']} actual={min_date}..{max_date}")
print(f"SESSION_SNAPSHOT_OK path={path} rows={rows} dates={min_date}..{max_date}")
PY
}

manifest_name_for_year() {
  local year="$1"
  python - "$manifest" "$year" <<'PY'
import json,sys
manifest,year=sys.argv[1:]
m=json.load(open(manifest,encoding="utf-8"))
matches=[str(x.get("file") or "") for x in m.get("years",[]) if str(x.get("year"))==year]
if len(matches)!=1:
    raise SystemExit(f"manifest year missing/ambiguous: year={year} matches={matches}")
print(matches[0])
PY
}

case "$cmd" in
  init)
    dataset_ref="$(resolve_ref)"
    retry_kaggle "session-manifest" kaggle datasets download "$dataset_ref" -f manifest.json -p "$out_dir" --unzip --quiet --force
    retry_kaggle "session-files" bash -c 'kaggle datasets files "$1" --page-size 500 --format "json(name,size)" >"$2"' _ "$dataset_ref" "$files_json"
    python - "$manifest" "$generation_id" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
actual=m.get("generation",{}).get("generation_id")
if actual!=sys.argv[2]:
    raise SystemExit(f"generation mismatch: expected {sys.argv[2]}, got {actual}")
PY
    printf '%s\n' "$dataset_ref" > "$dataset_ref_file"
    echo "KAGGLE_SNAPSHOT_SESSION_INIT_OK"
    ;;
  fetch)
    [[ -s "$manifest" && -s "$dataset_ref_file" && -s "$files_json" ]] || {
      echo "session not initialized" >&2
      exit 4
    }
    dataset_ref="$(cat "$dataset_ref_file")"
    [[ "$#" -ge 1 ]] || { echo "at least one year required" >&2; exit 5; }

    for year in "$@"; do
      [[ "$year" =~ ^[0-9]{4}$ ]] || { echo "invalid year: $year" >&2; exit 5; }
      manifest_name="$(manifest_name_for_year "$year")"
      gz_path="$out_dir/$manifest_name"
      plain_name="${manifest_name%.gz}"
      plain_path="$out_dir/$plain_name"

      if [[ -s "$gz_path" ]]; then
        echo "SESSION_SNAPSHOT_REUSE year=$year path=$gz_path"
        validate_one "$gz_path" "$manifest_name"
        continue
      fi
      if [[ "$plain_name" != "$manifest_name" && -s "$plain_path" ]]; then
        echo "SESSION_SNAPSHOT_REUSE year=$year path=$plain_path"
        validate_one "$plain_path" "$manifest_name"
        continue
      fi

      remote_name="$(python - "$files_json" "$manifest_name" "$plain_name" <<'PY'
import json,sys
files_path,manifest_name,plain_name=sys.argv[1:]
rows=json.load(open(files_path,encoding="utf-8"))
names=[str(x.get("name") or "") for x in rows]
if manifest_name in names:
    print(manifest_name); raise SystemExit
if plain_name != manifest_name and plain_name in names:
    print(plain_name); raise SystemExit
nested=[x for x in names if x.endswith("/"+manifest_name) or (plain_name!=manifest_name and x.endswith("/"+plain_name))]
if len(nested)==1:
    print(nested[0]); raise SystemExit
raise SystemExit(f"snapshot file missing/ambiguous manifest={manifest_name} plain={plain_name}")
PY
)"
      echo "SESSION_REMOTE_RESOLVED year=$year file=$remote_name"
      retry_kaggle "session-snapshot:$year" kaggle datasets download "$dataset_ref" -f "$remote_name" -p "$out_dir" --unzip --quiet --force

      if [[ -s "$gz_path" ]]; then
        validate_one "$gz_path" "$manifest_name"
      elif [[ -s "$plain_path" ]]; then
        validate_one "$plain_path" "$manifest_name"
      else
        echo "::error::download succeeded but snapshot file is missing year=$year expected=$manifest_name" >&2
        exit 6
      fi
      sleep 8
    done
    ;;
  *)
    echo "unknown command: $cmd" >&2
    exit 7
    ;;
esac
