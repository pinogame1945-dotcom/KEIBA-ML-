#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_ROOT="${1:-out/yearly-snapshots}"
SAFETY_LIMIT_BYTES="${KAGGLE_SNAPSHOT_SAFETY_LIMIT_BYTES:-193273528320}" # 180 GiB
EPHEMERAL="${KAGGLE_SNAPSHOT_EPHEMERAL:-0}"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

manifest="$(find "$SNAPSHOT_ROOT" -mindepth 2 -maxdepth 2 -name manifest.json -print | sort | tail -n 1)"
if [[ -z "$manifest" || ! -f "$manifest" ]]; then
  echo "snapshot manifest not found under $SNAPSHOT_ROOT" >&2
  exit 3
fi

generation_dir="$(dirname "$manifest")"

readarray -t meta < <(python - "$manifest" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
g=m["generation"]
print(g["generation_id"])
print(g["feature_schema_version"])
print(m["total_bytes"])
for y in m["years"]:
    print(f'FILE\t{y["file"]}\t{y["sha256"]}\t{y["bytes"]}')
PY
)

generation_id="${meta[0]}"
schema="${meta[1]}"
snapshot_bytes="${meta[2]}"
slug="keiba-ml-snapshot-${generation_id}"

echo "Snapshot generation: $generation_id"
echo "Feature schema: $schema"
echo "Snapshot bytes: $snapshot_bytes"

# Verify every yearly file against the authoritative manifest before any upload.
while IFS=$'\t' read -r kind file expected_sha expected_bytes; do
  [[ "$kind" == "FILE" ]] || continue
  actual_bytes="$(stat -c %s "$generation_dir/$file")"
  actual_sha="$(sha256sum "$generation_dir/$file" | awk '{print $1}')"
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
cp "$generation_dir"/snapshot-*.jsonl.gz "$stage"/
cp "$manifest" "$stage/manifest.json"
(
  cd "$stage"
  sha256sum snapshot-*.jsonl.gz manifest.json > SHA256SUMS
)

kaggle datasets init -p "$stage" >/dev/null

python - "$stage/dataset-metadata.json" "$slug" "$generation_id" "$schema" <<'PY'
import json,sys
p,slug,generation_id,schema=sys.argv[1:]
d=json.load(open(p,encoding="utf-8"))
dataset_id=str(d.get("id",""))
if "INSERT_SLUG_HERE" not in dataset_id:
    raise SystemExit(f"unexpected Kaggle init dataset id: {dataset_id!r}")
d["id"]=dataset_id.replace("INSERT_SLUG_HERE",slug)
d["title"]=f"KEIBA ML Snapshot {generation_id}"[:50]
d["licenses"]=[{"name":"other"}]
d["description"]=(
    "Private KEIBA-ML research snapshot. "
    f"Generation {generation_id}; feature schema {schema}; "
    "STRICT_PRIOR_DATE_ONLY. Generated automatically by GitHub Actions."
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

echo "Kaggle dataset ref: $dataset_ref"

# Idempotency: if this exact generation already exists, verify it rather than duplicating it.
if kaggle datasets status "$dataset_ref" >/dev/null 2>&1; then
  echo "Dataset already exists; verifying remote manifest instead of uploading again."
else
  # Conservative free-tier guard: sum ALL datasets owned by this account, not just KEIBA datasets.
  # If quota information cannot be read, stop rather than risk crossing a free storage boundary.
  usage_json="$(mktemp)"
  printf '[]\n' > "$usage_json"
  page=1
  while :; do
    page_json="$(mktemp)"
    if ! kaggle datasets list --mine --page "$page" --format "json(ref,totalBytes)" >"$page_json"; then
      echo "Could not verify Kaggle storage usage; refusing upload." >&2
      exit 7
    fi
    # Kaggle CLI exits 0 but may emit an empty body when the account owns no datasets.
    # Treat that as an empty page instead of attempting to JSON-decode an empty file.
    if [[ ! -s "$page_json" ]]; then
      count=0
    else
      count="$(python - "$usage_json" "$page_json" <<'PY'
import json,sys
dst=json.load(open(sys.argv[1],encoding="utf-8"))
with open(sys.argv[2],encoding="utf-8") as fh:
    raw=fh.read().strip()
src=[] if not raw else json.loads(raw)
if not isinstance(src,list):
    raise SystemExit("unexpected Kaggle datasets list payload")
dst.extend(src)
json.dump(dst,open(sys.argv[1],"w",encoding="utf-8"))
print(len(src))
PY
)"
    fi
    [[ "$count" == "0" ]] && break
    page=$((page + 1))
    [[ "$page" -le 1000 ]] || { echo "Kaggle dataset pagination safety stop" >&2; exit 7; }
  done

  current_bytes="$(python - "$usage_json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
print(sum(int(r.get("totalBytes") or 0) for r in rows))
PY
)"

  projected=$(( current_bytes + snapshot_bytes ))
  echo "Owned Kaggle dataset bytes: $current_bytes"
  echo "Projected bytes after upload: $projected"
  echo "Safety ceiling: $SAFETY_LIMIT_BYTES"

  if (( projected > SAFETY_LIMIT_BYTES )); then
    echo "Projected Kaggle storage exceeds the 180 GiB safety ceiling; refusing upload." >&2
    exit 8
  fi

  kaggle datasets create -p "$stage" --quiet --keep-tabular --dir-mode skip
  created_here=1
fi

verify_dir="$(mktemp -d)"
kaggle datasets download "$dataset_ref" -f manifest.json -p "$verify_dir" --unzip --quiet --force
cmp "$stage/manifest.json" "$verify_dir/manifest.json"

if [[ "$EPHEMERAL" == "1" ]]; then
  if [[ "$created_here" != "1" ]]; then
    echo "Ephemeral persistence test unexpectedly reused an existing dataset; refusing to delete it." >&2
    exit 9
  fi
  kaggle datasets delete "$dataset_ref" --yes >/dev/null
  created_here=0
  echo "KAGGLE_SNAPSHOT_EPHEMERAL_DELETE_OK"
fi

echo "KAGGLE_SNAPSHOT_PERSIST_OK"
echo "dataset_ref=$dataset_ref"
echo "generation_id=$generation_id"
