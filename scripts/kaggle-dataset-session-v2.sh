#!/usr/bin/env bash
set -euo pipefail

cmd="${1:?usage: kaggle-dataset-session-v2.sh init|fetch ...}"

retry_metadata() {
  local ref="$1" out="$2"
  local attempt delay log
  log="$(mktemp)"
  for attempt in 1 2 3 4 5 6; do
    : > "$log"
    if kaggle datasets files "$ref" --page-size 1000 --format "json(name,size)" >"$out.tmp" 2>"$log"; then
      mv "$out.tmp" "$out"
      rm -f "$log"
      return 0
    fi
    cat "$log" >&2 || true
    if grep -Eqi '(^|[^0-9])404([^0-9]|$)|not[[:space:]]+found' "$log"; then
      echo "::error::dataset metadata not found ref=$ref" >&2
      rm -f "$log" "$out.tmp"
      return 44
    fi
    if grep -Eqi '(^|[^0-9])4[0-9][0-9]([^0-9]|$)' "$log" && ! grep -Eqi '(^|[^0-9])429([^0-9]|$)' "$log"; then
      echo "::error::non-retryable dataset metadata client error ref=$ref" >&2
      rm -f "$log" "$out.tmp"
      return 43
    fi
    if [[ "$attempt" -eq 6 ]]; then
      echo "::error::metadata failed ref=$ref after $attempt attempts" >&2
      rm -f "$log" "$out.tmp"
      return 1
    fi
    case "$attempt" in
      1) delay=10 ;;
      2) delay=20 ;;
      3) delay=40 ;;
      4) delay=60 ;;
      *) delay=90 ;;
    esac
    echo "::warning::metadata retry ref=$ref attempt=$attempt/6 sleep=${delay}s" >&2
    sleep "$delay"
  done
}

resolve_remote() {
  local files_json="$1" logical="$2"
  python - "$files_json" "$logical" <<'PY'
import json,os,sys
path,logical=sys.argv[1:]
rows=json.load(open(path,encoding="utf-8"))
names=[str(x.get("name") or "") for x in rows if str(x.get("name") or "")]
name_set=set(names)

candidates=[]
def add(x):
    if x and x not in candidates:
        candidates.append(x)

add(logical)
if logical.endswith(".gz"):
    add(logical[:-3])
else:
    add(logical+".gz")

base=os.path.basename(logical)
if base.endswith(".gz"):
    alt_base=base[:-3]
else:
    alt_base=base+".gz"

# Exact metadata matches only.  No speculative download is attempted.
for x in candidates:
    if x in name_set:
        print(x)
        raise SystemExit(0)

basename_matches=[x for x in names if os.path.basename(x) in {base,alt_base}]
if len(basename_matches)==1:
    print(basename_matches[0])
    raise SystemExit(0)

raise SystemExit(
    f"remote file missing/ambiguous logical={logical} "
    f"basename_matches={basename_matches}"
)
PY
}

retry_exact_download() {
  local ref="$1" remote="$2" out_dir="$3"
  local attempt delay log
  log="$(mktemp)"
  for attempt in 1 2 3 4 5 6; do
    : > "$log"
    if kaggle datasets download "$ref" -f "$remote" -p "$out_dir" --unzip --quiet --force >"$log" 2>&1; then
      cat "$log"
      rm -f "$log"
      return 0
    fi
    cat "$log" >&2 || true

    # A 404 here means the metadata changed between list and download.
    # Do not guess another filename; fail so the caller can re-init metadata.
    if grep -Eqi '(^|[^0-9])404([^0-9]|$)|not[[:space:]]+found' "$log"; then
      echo "::error::metadata drift: exact remote disappeared ref=$ref remote=$remote" >&2
      rm -f "$log"
      return 44
    fi
    if grep -Eqi '(^|[^0-9])4[0-9][0-9]([^0-9]|$)' "$log" && ! grep -Eqi '(^|[^0-9])429([^0-9]|$)' "$log"; then
      echo "::error::non-retryable exact-download client error ref=$ref remote=$remote" >&2
      rm -f "$log"
      return 43
    fi

    if [[ "$attempt" -eq 6 ]]; then
      echo "::error::exact download failed ref=$ref remote=$remote after $attempt attempts" >&2
      rm -f "$log"
      return 1
    fi
    case "$attempt" in
      1) delay=10 ;;
      2) delay=20 ;;
      3) delay=40 ;;
      4) delay=60 ;;
      *) delay=90 ;;
    esac
    echo "::warning::exact download retry ref=$ref remote=$remote attempt=$attempt/6 sleep=${delay}s" >&2
    sleep "$delay"
  done
}

case "$cmd" in
  init)
    ref="${2:?dataset ref required}"
    out_dir="${3:?out dir required}"
    [[ -n "${KAGGLE_API_TOKEN:-}" ]] || { echo "KAGGLE_API_TOKEN required" >&2; exit 2; }
    mkdir -p "$out_dir"
    retry_metadata "$ref" "$out_dir/.files.json"
    printf '%s\n' "$ref" > "$out_dir/.dataset-ref"
    python - "$out_dir/.files.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))
print(f"KAGGLE_EXACT_SESSION_INIT_OK files={len(rows)}")
PY
    ;;

  fetch)
    out_dir="${2:?out dir required}"
    shift 2
    [[ "$#" -ge 1 ]] || { echo "at least one logical file required" >&2; exit 3; }
    [[ -s "$out_dir/.files.json" && -s "$out_dir/.dataset-ref" ]] || {
      echo "exact session not initialized: $out_dir" >&2
      exit 4
    }
    ref="$(cat "$out_dir/.dataset-ref")"
    for logical in "$@"; do
      remote="$(resolve_remote "$out_dir/.files.json" "$logical")"
      remote_base="$(basename "$remote")"
      logical_base="$(basename "$logical")"
      plain_logical="${logical_base%.gz}"

      existing="$(find "$out_dir" -type f \( -name "$remote_base" -o -name "$logical_base" -o -name "$plain_logical" \) -print -quit)"
      if [[ -n "$existing" ]]; then
        echo "KAGGLE_EXACT_REUSE logical=$logical remote=$remote path=$existing"
        continue
      fi

      echo "KAGGLE_EXACT_FETCH logical=$logical remote=$remote"
      retry_exact_download "$ref" "$remote" "$out_dir"

      downloaded="$(find "$out_dir" -type f \( -name "$remote_base" -o -name "$logical_base" -o -name "$plain_logical" \) -print -quit)"
      [[ -n "$downloaded" ]] || {
        echo "::error::exact download returned no local file logical=$logical remote=$remote" >&2
        exit 5
      }
      echo "KAGGLE_EXACT_FETCH_OK logical=$logical remote=$remote path=$downloaded"
    done
    ;;

  *)
    echo "unknown command: $cmd" >&2
    exit 6
    ;;
esac
