#!/usr/bin/env bash
set -euo pipefail

generation_id="${1:?usage: kaggle-fetch-snapshot.sh GENERATION_ID [YEAR|YEAR1,YEAR2,...|all] [OUT_DIR]}"
year_spec="${2:-all}"
out_dir="${3:-out/restored-snapshots/$generation_id}"
slug="keiba-ml-snapshot-${generation_id}"

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

if [[ -n "${KAGGLE_DATASET_REF:-}" ]]; then
  dataset_ref="$KAGGLE_DATASET_REF"
  if [[ "$dataset_ref" != */"$slug" ]]; then
    echo "KAGGLE_DATASET_REF does not match generation: $dataset_ref" >&2
    exit 3
  fi
else
  results="$(mktemp)"
  printf '[]\n' > "$results"
  page=1
  while :; do
    page_json="$(mktemp)"
    kaggle datasets list --mine --search "$slug" --page "$page" --format "json(ref)" >"$page_json"
    count="$(python - "$results" "$page_json" <<'PY'
import json,sys
dst_path,page_path=sys.argv[1:]
with open(dst_path,encoding="utf-8") as f:
    dst=json.load(f)
raw=open(page_path,encoding="utf-8").read().strip()
src=[] if not raw else json.loads(raw)
if not isinstance(src,list):
    raise SystemExit("Kaggle dataset list response must be a JSON array")
dst.extend(src)
with open(dst_path,"w",encoding="utf-8") as f:
    json.dump(dst,f)
print(len(src))
PY
)"
    [[ "$count" == "0" ]] && break
    page=$((page + 1))
    [[ "$page" -le 1000 ]] || { echo "Kaggle dataset pagination safety stop" >&2; exit 3; }
  done

  dataset_ref="$(python - "$results" "$slug" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
slug=sys.argv[2]
matches=[str(r.get("ref","")) for r in rows if str(r.get("ref","")).endswith("/"+slug)]
if len(matches)!=1:
    raise SystemExit(f"expected exactly one dataset for {slug}, found {matches}")
print(matches[0])
PY
)"
fi

retry_kaggle "manifest" kaggle datasets download "$dataset_ref" -f manifest.json -p "$out_dir" --unzip --quiet --force

python - "$out_dir/manifest.json" "$generation_id" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
actual=m.get("generation",{}).get("generation_id")
if actual != sys.argv[2]:
    raise SystemExit(f"generation mismatch: expected {sys.argv[2]}, got {actual}")
PY

remote_csv="$(mktemp)"
list_files() {
  kaggle datasets files "$dataset_ref" --page-size 200 -v > "$remote_csv"
}
retry_kaggle "files-list" list_files

if [[ "$year_spec" == "all" ]]; then
  mapfile -t requested < <(python - "$out_dir/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
for y in m["years"]: print(y["file"])
PY
)
else
  mapfile -t requested < <(YEAR_SPEC="$year_spec" python - <<'PY'
import os
raw=os.environ["YEAR_SPEC"]
years=[x.strip() for x in raw.split(",") if x.strip()]
if not years:
    raise SystemExit("empty year specification")
for y in years:
    if not (len(y)==4 and y.isdigit()):
        raise SystemExit(f"invalid year in specification: {y!r}")
    print(f"snapshot-{y}.jsonl.gz")
PY
)
fi

resolved=()
for manifest_name in "${requested[@]}"; do
  remote_name="$(python - "$remote_csv" "$manifest_name" <<'PY'
import csv,sys
csv_path,wanted=sys.argv[1:]
with open(csv_path,newline="",encoding="utf-8") as f:
    names=[str(r.get("name") or "") for r in csv.DictReader(f)]
if wanted in names:
    print(wanted)
elif wanted.endswith(".gz") and wanted[:-3] in names:
    print(wanted[:-3])
else:
    raise SystemExit(f"remote snapshot file missing: wanted {wanted}; available={names}")
PY
)"
  retry_kaggle "snapshot:$remote_name" kaggle datasets download "$dataset_ref" -f "$remote_name" -p "$out_dir" --unzip --quiet --force
  resolved+=("$remote_name")
  sleep 8
done

python - "$out_dir/manifest.json" "$out_dir" "${resolved[@]}" <<'PY'
import gzip,hashlib,json,os,sys
manifest,out_dir,*files=sys.argv[1:]
m=json.load(open(manifest,encoding="utf-8"))
by_gz={y["file"]:y for y in m["years"]}
for name in files:
    gz_name=name if name.endswith(".gz") else name+".gz"
    if gz_name not in by_gz:
        raise SystemExit(f"{name} is not represented in manifest")
    expected=by_gz[gz_name]
    path=os.path.join(out_dir,name)
    if name.endswith(".gz"):
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
        if h.hexdigest()!=expected["sha256"]:
            raise SystemExit(f"sha256 mismatch: {name}")
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
        raise SystemExit(f"row mismatch {name}: expected={expected['rows']} actual={rows}")
    if min_date!=expected["min_date"] or max_date!=expected["max_date"]:
        raise SystemExit(
            f"date-range mismatch {name}: expected={expected['min_date']}..{expected['max_date']} "
            f"actual={min_date}..{max_date}"
        )
    print(f"RESTORED_FILE_OK name={name} rows={rows} dates={min_date}..{max_date}")
print("KAGGLE_SNAPSHOT_RESTORE_OK")
PY

echo "dataset_ref=$dataset_ref"
echo "generation_id=$generation_id"
printf 'restored_file=%s\n' "${resolved[@]}"
