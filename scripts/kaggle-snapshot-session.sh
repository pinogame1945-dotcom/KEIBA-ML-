#!/usr/bin/env bash
set -euo pipefail

cmd="${1:?usage: kaggle-snapshot-session.sh init|fetch GENERATION_ID OUT_DIR [YEAR...]}"
generation_id="${2:?generation id required}"
out_dir="${3:?out dir required}"
shift 3

slug="keiba-ml-snapshot-${generation_id}"
manifest="$out_dir/manifest.json"
remote_csv="$out_dir/.remote-files.csv"
dataset_ref_file="$out_dir/.dataset-ref"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

mkdir -p "$out_dir"

retry_kaggle() {
  local label="$1"
  shift
  local attempt delay
  for attempt in 1 2 3 4 5 6; do
    if "$@"; then
      return 0
    fi
    if [[ "$attempt" -eq 6 ]]; then
      echo "::error::Kaggle operation failed label=$label after $attempt attempts" >&2
      return 1
    fi
    case "$attempt" in
      1) delay=15 ;;
      2) delay=30 ;;
      3) delay=60 ;;
      4) delay=90 ;;
      *) delay=120 ;;
    esac
    echo "::warning::Kaggle retry label=$label attempt=$attempt/6 sleep=${delay}s" >&2
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
  local path="$1" remote_name="$2"
  python - "$manifest" "$path" "$remote_name" <<'PY'
import gzip,hashlib,json,os,sys
manifest,path,name=sys.argv[1:]
m=json.load(open(manifest,encoding="utf-8"))
gz_name=name if name.endswith(".gz") else name+".gz"
by_gz={y["file"]:y for y in m["years"]}
if gz_name not in by_gz:
    raise SystemExit(f"{name} not in manifest")
expected=by_gz[gz_name]
if name.endswith(".gz"):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    if h.hexdigest()!=expected["sha256"]:
        raise SystemExit(f"sha256 mismatch: {name}")
    opener=lambda: gzip.open(path,"rt",encoding="utf-8")
else:
    opener=lambda: open(path,"rt",encoding="utf-8")
rows=0; min_date=None; max_date=None
with opener() as f:
    for line in f:
        if not line.strip(): continue
        row=json.loads(line)
        date=str(row.get("race_date") or (row.get("features") or {}).get("race_date") or "")[:10]
        rows+=1
        if date:
            min_date=date if min_date is None or date<min_date else min_date
            max_date=date if max_date is None or date>max_date else max_date
if rows!=int(expected["rows"]):
    raise SystemExit(f"row mismatch {name}: expected={expected['rows']} actual={rows}")
if min_date!=expected["min_date"] or max_date!=expected["max_date"]:
    raise SystemExit(f"date mismatch {name}: {min_date}..{max_date}")
print(f"SESSION_SNAPSHOT_OK name={name} rows={rows} dates={min_date}..{max_date}")
PY
}

case "$cmd" in
  init)
    dataset_ref="$(resolve_ref)"
    retry_kaggle "session-manifest" kaggle datasets download "$dataset_ref" -f manifest.json -p "$out_dir" --unzip --quiet --force
    python - "$manifest" "$generation_id" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
actual=m.get("generation",{}).get("generation_id")
if actual!=sys.argv[2]:
    raise SystemExit(f"generation mismatch: expected {sys.argv[2]}, got {actual}")
PY
    list_files() {
      kaggle datasets files "$dataset_ref" --page-size 200 -v > "$remote_csv"
    }
    retry_kaggle "session-files-list" list_files
    printf '%s
' "$dataset_ref" > "$dataset_ref_file"
    echo "KAGGLE_SNAPSHOT_SESSION_INIT_OK"
    ;;
  fetch)
    [[ -s "$manifest" && -s "$remote_csv" && -s "$dataset_ref_file" ]] || {
      echo "session not initialized" >&2
      exit 4
    }
    dataset_ref="$(cat "$dataset_ref_file")"
    [[ "$#" -ge 1 ]] || { echo "at least one year required" >&2; exit 5; }
    for year in "$@"; do
      [[ "$year" =~ ^[0-9]{4}$ ]] || { echo "invalid year: $year" >&2; exit 5; }
      wanted="snapshot-${year}.jsonl.gz"
      remote_name="$(python - "$remote_csv" "$wanted" <<'PY'
import csv,sys
csv_path,wanted=sys.argv[1:]
with open(csv_path,newline="",encoding="utf-8") as f:
    names=[str(r.get("name") or "") for r in csv.DictReader(f)]
if wanted in names:
    print(wanted)
elif wanted.endswith(".gz") and wanted[:-3] in names:
    print(wanted[:-3])
else:
    raise SystemExit(f"remote snapshot missing: {wanted}")
PY
)"
      local_path="$out_dir/$remote_name"
      if [[ -s "$local_path" ]]; then
        echo "SESSION_SNAPSHOT_REUSE year=$year path=$local_path"
        validate_one "$local_path" "$remote_name"
        continue
      fi
      retry_kaggle "session-snapshot:$year" kaggle datasets download "$dataset_ref" -f "$remote_name" -p "$out_dir" --unzip --quiet --force
      validate_one "$local_path" "$remote_name"
      sleep 8
    done
    ;;
  *)
    echo "unknown command: $cmd" >&2
    exit 6
    ;;
esac
