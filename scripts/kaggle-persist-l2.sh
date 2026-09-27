#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${1:-out/l2-bundles}"
EPHEMERAL="${KAGGLE_L2_EPHEMERAL:-0}"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi
if [[ "${KAGGLE_STORAGE_PREFLIGHT_OK:-0}" != "1" ]]; then
  echo "Kaggle storage preflight marker missing; run scripts/kaggle-storage-preflight.sh before compute." >&2
  exit 12
fi

manifest="$(find "$BUNDLE_ROOT" -mindepth 2 -maxdepth 2 -name manifest.json -print | sort | tail -n 1)"
if [[ -z "$manifest" || ! -f "$manifest" ]]; then
  echo "L2 bundle manifest not found under $BUNDLE_ROOT" >&2
  exit 3
fi
bundle_dir="$(dirname "$manifest")"

readarray -t meta < <(python - "$manifest" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
print(m["bundle_id"])
print(m["snapshot_generation"])
print(m["total_bytes"])
for row in m["expert_outputs"]:
    print("FILE\t"+row["file"]+"\t"+row["sha256"]+"\t"+str(row["bytes"]))
s=m["summary"]
print("FILE\t"+s["file"]+"\t"+s["sha256"]+"\t"+str(s["bytes"]))
PY
)

bundle_id="${meta[0]}"
snapshot_generation="${meta[1]}"
bundle_bytes="${meta[2]}"
slug="keiba-ml-l2-${bundle_id}"

echo "L2 bundle: $bundle_id"
echo "Snapshot generation: $snapshot_generation"
echo "Bundle bytes: $bundle_bytes"

while IFS=$'\t' read -r kind file expected_sha expected_bytes; do
  [[ "$kind" == "FILE" ]] || continue
  actual_bytes="$(stat -c %s "$bundle_dir/$file")"
  actual_sha="$(sha256sum "$bundle_dir/$file" | awk '{print $1}')"
  [[ "$actual_bytes" == "$expected_bytes" ]] || { echo "size mismatch: $file" >&2; exit 4; }
  [[ "$actual_sha" == "$expected_sha" ]] || { echo "sha256 mismatch: $file" >&2; exit 5; }
done < <(printf '%s\n' "${meta[@]:3}")

stage="$(mktemp -d)"
created_here=0
dataset_ref=""
cleanup() {
  if [[ "$EPHEMERAL" == "1" && "$created_here" == "1" && -n "$dataset_ref" ]]; then
    kaggle datasets delete "$dataset_ref" --yes >/dev/null 2>&1 || true
  fi
  rm -rf "$stage"
}
trap cleanup EXIT

python - "$manifest" "$bundle_dir" "$stage" <<'PY'
import json,shutil,sys
manifest_path,bundle_dir,stage=sys.argv[1:]
m=json.load(open(manifest_path,encoding="utf-8"))
for row in m["expert_outputs"]:
    shutil.copy2(bundle_dir+"/"+row["file"],stage+"/"+row["file"])
shutil.copy2(bundle_dir+"/"+m["summary"]["file"],stage+"/"+m["summary"]["file"])
shutil.copy2(manifest_path,stage+"/manifest.json")
PY
(
  cd "$stage"
  sha256sum *.jsonl.gz comparison-summary.json manifest.json > SHA256SUMS
)

kaggle datasets init -p "$stage" >/dev/null
python - "$stage/dataset-metadata.json" "$slug" "$bundle_id" "$snapshot_generation" <<'PY'
import json,sys
p,slug,bundle_id,snapshot_generation=sys.argv[1:]
d=json.load(open(p,encoding="utf-8"))
dataset_id=str(d.get("id",""))
if "INSERT_SLUG_HERE" not in dataset_id:
    raise SystemExit(f"unexpected Kaggle init dataset id: {dataset_id!r}")
d["id"]=dataset_id.replace("INSERT_SLUG_HERE",slug)
d["title"]=f"KEIBA ML L2 {bundle_id}"[:50]
d["licenses"]=[{"name":"other"}]
d["description"]=(
    "Private KEIBA-ML L2 expert-output companion dataset. "
    f"Bundle {bundle_id}; snapshot generation {snapshot_generation}; "
    "L1_TO_L2_OUTPUT_CONTRACT_V1; pre-race only."
)
json.dump(d,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
open(p,"a",encoding="utf-8").write("\n")
print(d["id"])
PY

dataset_ref="$(python - "$stage/dataset-metadata.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding="utf-8"))["id"])
PY
)"
if [[ "$dataset_ref" == *"INSERT_"* || "$dataset_ref" != */"$slug" ]]; then
  echo "Kaggle dataset ref could not be resolved safely: $dataset_ref" >&2
  exit 6
fi

if kaggle datasets status "$dataset_ref" >/dev/null 2>&1; then
  echo "Dataset already exists; verifying remote manifest instead of uploading again."
else
  # Capacity was checked once before compute. Do not re-check here.
  kaggle datasets create -p "$stage" --quiet --keep-tabular --dir-mode skip
  created_here=1
fi

verify_dir="$(mktemp -d)"
verified=0
VERIFY_ATTEMPTS="${KAGGLE_L2_VERIFY_ATTEMPTS:-60}"
VERIFY_SLEEP_SECONDS="${KAGGLE_L2_VERIFY_SLEEP_SECONDS:-10}"
for attempt in $(seq 1 "$VERIFY_ATTEMPTS"); do
  rm -f "$verify_dir/manifest.json"
  if kaggle datasets download "$dataset_ref" -f manifest.json -p "$verify_dir" --unzip --quiet --force; then
    if cmp "$stage/manifest.json" "$verify_dir/manifest.json"; then
      verified=1
      echo "Kaggle L2 manifest verification succeeded on attempt $attempt."
      break
    fi
    echo "Remote L2 manifest did not match local manifest." >&2
    exit 10
  fi
  if [[ "$attempt" -lt "$VERIFY_ATTEMPTS" ]]; then sleep "$VERIFY_SLEEP_SECONDS"; fi
done
if [[ "$verified" != "1" ]]; then
  echo "Kaggle L2 dataset did not become readable within the verification window." >&2
  exit 11
fi

if [[ "$EPHEMERAL" == "1" ]]; then
  if [[ "$created_here" != "1" ]]; then
    echo "Ephemeral test unexpectedly reused an existing dataset; refusing delete." >&2
    exit 9
  fi
  kaggle datasets delete "$dataset_ref" --yes >/dev/null
  created_here=0
  echo "KAGGLE_L2_EPHEMERAL_DELETE_OK"
fi

echo "KAGGLE_L2_PERSIST_OK"
echo "dataset_ref=$dataset_ref"
echo "bundle_id=$bundle_id"
echo "snapshot_generation=$snapshot_generation"
