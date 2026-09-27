#!/usr/bin/env bash
set -euo pipefail

refs_file="$(mktemp)"
trap 'rm -f "$refs_file"' EXIT
: > "$refs_file"

page=1
while :; do
  page_csv="$(mktemp)"
  if ! kaggle datasets list --mine --page "$page" --csv >"$page_csv" 2>&1; then
    echo "Kaggle owned-dataset listing failed on page $page" >&2
    cat "$page_csv" >&2
    rm -f "$page_csv"
    exit 7
  fi

  page_refs="$(python - "$page_csv" <<'PY'
import csv,sys
p=sys.argv[1]
raw=open(p,encoding="utf-8",errors="replace").read().strip()
if not raw:
    raise SystemExit(0)
low=raw.lower()
if low.startswith("no datasets") or low.startswith("no dataset"):
    raise SystemExit(0)
rows=list(csv.DictReader(raw.splitlines()))
if not rows:
    raise SystemExit(0)
if "ref" not in rows[0]:
    raise SystemExit(f"unexpected Kaggle dataset-list CSV columns: {list(rows[0])}")
for row in rows:
    ref=(row.get("ref") or "").strip()
    if not ref:
        continue
    if "/" not in ref:
        raise SystemExit(f"unexpected Kaggle dataset ref: {ref!r}")
    print(ref)
PY
)"
  rm -f "$page_csv"

  [[ -n "$page_refs" ]] || break
  before="$(sort -u "$refs_file" | wc -l)"
  printf '%s\n' "$page_refs" >> "$refs_file"
  sort -u "$refs_file" -o "$refs_file"
  after="$(wc -l < "$refs_file")"
  [[ "$after" -gt "$before" ]] || break

  page=$((page + 1))
  [[ "$page" -le 1000 ]] || {
    echo "Kaggle dataset pagination safety stop" >&2
    exit 7
  }
done

total=0
while IFS= read -r ref; do
  [[ -n "$ref" ]] || continue
  files_csv="$(mktemp)"
  if ! kaggle datasets files "$ref" --csv >"$files_csv" 2>&1; then
    echo "Kaggle file listing failed for $ref" >&2
    cat "$files_csv" >&2
    rm -f "$files_csv"
    exit 7
  fi

  bytes="$(python - "$files_csv" <<'PY'
import csv,sys
p=sys.argv[1]
raw=open(p,encoding="utf-8",errors="replace").read().strip()
if not raw:
    print(0)
    raise SystemExit(0)
rows=list(csv.DictReader(raw.splitlines()))
if not rows:
    print(0)
    raise SystemExit(0)
if "size" not in rows[0]:
    raise SystemExit(f"unexpected Kaggle files CSV columns: {list(rows[0])}")
total=0
for row in rows:
    value=(row.get("size") or "").strip()
    if not value:
        raise SystemExit("missing Kaggle file size")
    try:
        total += int(value)
    except ValueError:
        raise SystemExit(f"unparseable Kaggle file size: {value!r}")
print(total)
PY
)"
  rm -f "$files_csv"
  echo "KAGGLE_DATASET_BYTES ref=$ref bytes=$bytes" >&2
  total=$(( total + bytes ))
done < "$refs_file"

printf '%s\n' "$total"
