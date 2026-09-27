#!/usr/bin/env bash
set -euo pipefail

SAFETY_LIMIT_BYTES="${KAGGLE_STORAGE_SAFETY_LIMIT_BYTES:-193273528320}" # 180 GiB

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

usage_json="$(mktemp)"
trap 'rm -f "$usage_json"' EXIT
printf '[]\n' > "$usage_json"

page=1
while :; do
  page_json="$(mktemp)"
  if ! kaggle datasets list --mine --page "$page" --format "json(ref,totalBytes)" >"$page_json"; then
    rm -f "$page_json"
    echo "Could not verify Kaggle storage usage; refusing to start research run." >&2
    exit 3
  fi
  count="$(python - "$usage_json" "$page_json" <<'PY'
import json,sys
dst=json.load(open(sys.argv[1],encoding="utf-8"))
raw=open(sys.argv[2],encoding="utf-8").read().strip()
if not raw or raw.lower().startswith("no datasets found"):
    src=[]
else:
    src=json.loads(raw)
if not isinstance(src,list):
    raise SystemExit("unexpected Kaggle datasets list payload")
dst.extend(src)
json.dump(dst,open(sys.argv[1],"w",encoding="utf-8"))
print(len(src))
PY
)"
  rm -f "$page_json"
  [[ "$count" == "0" ]] && break
  page=$((page+1))
  [[ "$page" -le 1000 ]] || { echo "Kaggle dataset pagination safety stop" >&2; exit 4; }
done

current_bytes="$(python - "$usage_json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
print(sum(int(r.get("totalBytes") or 0) for r in rows))
PY
)"

echo "Owned Kaggle dataset bytes: $current_bytes"
echo "Research storage ceiling: $SAFETY_LIMIT_BYTES"

if (( current_bytes >= SAFETY_LIMIT_BYTES )); then
  echo "Owned Kaggle storage is already at/above the 180 GiB research ceiling; refusing to start." >&2
  exit 5
fi

echo "KAGGLE_STORAGE_PREFLIGHT_OK current_bytes=$current_bytes limit_bytes=$SAFETY_LIMIT_BYTES"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "KAGGLE_STORAGE_PREFLIGHT_OK=1" >> "$GITHUB_ENV"
  echo "KAGGLE_STORAGE_PREFLIGHT_BYTES=$current_bytes" >> "$GITHUB_ENV"
fi
