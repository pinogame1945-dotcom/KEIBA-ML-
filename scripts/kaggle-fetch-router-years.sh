#!/usr/bin/env bash
set -euo pipefail

years_spec="${1:?usage: kaggle-fetch-router-years.sh YEAR1,YEAR2,... [OUT_DIR]}"
out_dir="${2:-out/router-seven}"

[[ -n "${KAGGLE_API_TOKEN:-}" ]] || {
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
}

mkdir -p "$out_dir"
session_script="scripts/kaggle-dataset-session-v2.sh"
[[ -s "$session_script" ]] || {
  echo "missing metadata-first Kaggle session helper: $session_script" >&2
  exit 3
}

IFS=',' read -r -a years <<< "$years_spec"
for raw_year in "${years[@]}"; do
  year="$(echo "$raw_year" | xargs)"
  [[ "$year" =~ ^[0-9]{4}$ ]] || { echo "invalid year: $year" >&2; exit 4; }

  ref="pino1945/keiba-l15-rolecmp-y${year}-v1"
  dst="$out_dir/$year"
  mkdir -p "$dst"

  router="$(find "$dst" -type f \( -name 'router-7k.jsonl' -o -name 'router-7k.jsonl.gz' \) -print -quit)"
  if [[ -n "$router" && -s "$router" ]]; then
    echo "ROUTER_YEAR_REUSE year=$year path=$router"
  else
    bash "$session_script" init "$ref" "$dst"
    bash "$session_script" fetch "$dst" router-7k.jsonl
    router="$(find "$dst" -type f \( -name 'router-7k.jsonl' -o -name 'router-7k.jsonl.gz' \) -print -quit)"
  fi

  [[ -n "$router" ]] || { echo "::error::router missing after exact metadata fetch year=$year" >&2; exit 5; }
  test -s "$router"
  echo "ROUTER_YEAR_OK year=$year path=$router"
done

echo "KAGGLE_ROUTER_YEARS_RESTORE_OK"
