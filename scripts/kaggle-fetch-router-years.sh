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

IFS=',' read -r -a years <<< "$years_spec"
for raw_year in "${years[@]}"; do
  year="$(echo "$raw_year" | xargs)"
  [[ "$year" =~ ^[0-9]{4}$ ]] || { echo "invalid year: $year" >&2; exit 3; }

  ref="pino1945/keiba-l15-rolecmp-y${year}-v1"
  dst="$out_dir/$year"
  mkdir -p "$dst"

  # Fast path: known canonical file name.
  if retry_kaggle "router:${year}:gz" kaggle datasets download "$ref" -f router-7k.jsonl.gz -p "$dst" --unzip --quiet --force; then
    :
  else
    # Fallback for datasets that store the uncompressed name.
    retry_kaggle "router:${year}:jsonl" kaggle datasets download "$ref" -f router-7k.jsonl -p "$dst" --unzip --quiet --force
  fi

  router="$(find "$dst" -type f \( -name 'router-7k.jsonl.gz' -o -name 'router-7k.jsonl' \) -print -quit)"
  test -n "$router"
  test -s "$router"
  echo "ROUTER_YEAR_OK year=$year path=$router"
  sleep 8
done

echo "KAGGLE_ROUTER_YEARS_RESTORE_OK"
