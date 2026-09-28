#!/usr/bin/env bash
set -euo pipefail

years_spec="${1:?usage: kaggle-fetch-router-years.sh YEAR1,YEAR2,... [OUT_DIR]}"
out_dir="${2:-out/router-seven}"

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

mkdir -p "$out_dir"

download_once_classified() {
  local ref="$1" remote="$2" dst="$3"
  local err
  err="$(mktemp)"
  if kaggle datasets download "$ref" -f "$remote" -p "$dst" --unzip --quiet --force 2>"$err"; then
    rm -f "$err"
    return 0
  fi
  cat "$err" >&2 || true
  if grep -Eqi '(^|[^0-9])404([^0-9]|$)|not[[:space:]]+found' "$err"; then
    rm -f "$err"
    return 44
  fi
  rm -f "$err"
  return 1
}

retry_remote() {
  local label="$1" ref="$2" remote="$3" dst="$4"
  local attempt delay rc
  for attempt in 1 2 3 4 5 6; do
    if download_once_classified "$ref" "$remote" "$dst"; then
      return 0
    fi
    rc=$?
    if [[ "$rc" -eq 44 ]]; then
      echo "::notice::Kaggle 404 label=$label remote=$remote; switching name immediately" >&2
      return 44
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

  if [[ -s "$dst/router-7k.jsonl" || -s "$dst/router-7k.jsonl.gz" ]]; then
    router="$(find "$dst" -type f \( -name 'router-7k.jsonl' -o -name 'router-7k.jsonl.gz' \) -print -quit)"
    echo "ROUTER_YEAR_REUSE year=$year path=$router"
  else
    if retry_remote "router:${year}:jsonl" "$ref" router-7k.jsonl "$dst"; then
      :
    else
      rc=$?
      if [[ "$rc" -eq 44 ]]; then
        retry_remote "router:${year}:gz" "$ref" router-7k.jsonl.gz "$dst"
      else
        exit "$rc"
      fi
    fi
    router="$(find "$dst" -type f \( -name 'router-7k.jsonl' -o -name 'router-7k.jsonl.gz' \) -print -quit)"
  fi

  test -n "$router"
  test -s "$router"
  echo "ROUTER_YEAR_OK year=$year path=$router"
  sleep 8
done

echo "KAGGLE_ROUTER_YEARS_RESTORE_OK"
