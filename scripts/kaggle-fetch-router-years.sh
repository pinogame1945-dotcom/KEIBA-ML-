#!/usr/bin/env bash
set -euo pipefail

years_spec="${1:?usage: kaggle-fetch-router-years.sh YEAR1,YEAR2,... [OUT_DIR]}"
out_dir="${2:-out/router-seven}"

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

list_files_json() {
  local ref="$1" tmp
  tmp="$(mktemp)"
  if retry_kaggle "router-files:$ref" bash -c     'kaggle datasets files "$1" --page-size 200 --format "json(name,size)" >"$2"' _ "$ref" "$tmp"; then
    cat "$tmp"
    rm -f "$tmp"
    return 0
  fi
  rm -f "$tmp"
  return 1
}

IFS=',' read -r -a years <<< "$years_spec"
for raw_year in "${years[@]}"; do
  year="$(echo "$raw_year" | xargs)"
  [[ "$year" =~ ^[0-9]{4}$ ]] || { echo "invalid year: $year" >&2; exit 3; }

  ref="pino1945/keiba-l15-rolecmp-y${year}-v1"
  dst="$out_dir/$year"
  mkdir -p "$dst"

  existing="$(find "$dst" -type f \( -name 'router-7k.jsonl.gz' -o -name 'router-7k.jsonl' \) -print -quit)"
  if [[ -n "$existing" && -s "$existing" ]]; then
    echo "ROUTER_YEAR_REUSE year=$year path=$existing"
    continue
  fi

  raw="$(list_files_json "$ref")"
  remote="$(RAW="$raw" python - <<'PY'
import json,os
rows=json.loads(os.environ["RAW"])
names=[str(x.get("name") or "") for x in rows]
for wanted in ("router-7k.jsonl.gz","router-7k.jsonl"):
    if wanted in names:
        print(wanted)
        raise SystemExit
nested=[x for x in names if x.endswith("/router-7k.jsonl.gz") or x.endswith("/router-7k.jsonl")]
if len(nested)==1:
    print(nested[0])
    raise SystemExit
raise SystemExit(f"router-7k missing/ambiguous: {names}")
PY
)"

  echo "ROUTER_REMOTE_RESOLVED year=$year file=$remote"
  retry_kaggle "router-download:$year" kaggle datasets download "$ref" -f "$remote" -p "$dst" --unzip --quiet --force

  router="$(find "$dst" -type f \( -name 'router-7k.jsonl.gz' -o -name 'router-7k.jsonl' \) -print -quit)"
  test -n "$router"
  test -s "$router"
  echo "ROUTER_YEAR_OK year=$year path=$router"
  sleep 8
done

echo "KAGGLE_ROUTER_YEARS_RESTORE_OK"
