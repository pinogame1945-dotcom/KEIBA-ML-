#!/usr/bin/env bash
set -euo pipefail

generation_id="${1:?usage: kaggle-fetch-snapshot.sh GENERATION_ID [YEAR|all] [OUT_DIR]}"
year="${2:-all}"
out_dir="${3:-out/restored-snapshots/$generation_id}"
slug="keiba-ml-snapshot-${generation_id}"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

mkdir -p "$out_dir"
results="$(mktemp)"
printf '[]\n' > "$results"
page=1
while :; do
  page_json="$(mktemp)"
  kaggle datasets list --mine --search "$slug" --page "$page" --format "json(ref)" >"$page_json"
  count="$(python - "$results" "$page_json" <<'PY'
import json,sys
dst=json.load(open(sys.argv[1],encoding="utf-8"))
src=json.load(open(sys.argv[2],encoding="utf-8"))
dst.extend(src)
json.dump(dst,open(sys.argv[1],"w",encoding="utf-8"))
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

kaggle datasets download "$dataset_ref" -f manifest.json -p "$out_dir" --unzip --quiet --force

python - "$out_dir/manifest.json" "$generation_id" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
actual=m.get("generation",{}).get("generation_id")
if actual != sys.argv[2]:
    raise SystemExit(f"generation mismatch: expected {sys.argv[2]}, got {actual}")
PY

if [[ "$year" == "all" ]]; then
  mapfile -t files < <(python - "$out_dir/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
for y in m["years"]: print(y["file"])
PY
)
else
  files=("snapshot-${year}.jsonl.gz")
fi

for file in "${files[@]}"; do
  kaggle datasets download "$dataset_ref" -f "$file" -p "$out_dir" --unzip --quiet --force
done

python - "$out_dir/manifest.json" "$out_dir" "${files[@]}" <<'PY'
import hashlib,json,os,sys
manifest,out_dir,*files=sys.argv[1:]
m=json.load(open(manifest,encoding="utf-8"))
expected={y["file"]:y["sha256"] for y in m["years"]}
for name in files:
    if name not in expected:
        raise SystemExit(f"{name} is not in manifest")
    h=hashlib.sha256()
    with open(os.path.join(out_dir,name),"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    if h.hexdigest()!=expected[name]:
        raise SystemExit(f"sha256 mismatch: {name}")
print("KAGGLE_SNAPSHOT_RESTORE_OK")
PY

echo "dataset_ref=$dataset_ref"
echo "generation_id=$generation_id"
