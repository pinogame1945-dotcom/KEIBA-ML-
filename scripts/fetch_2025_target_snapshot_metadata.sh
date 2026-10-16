#!/usr/bin/env bash
# Fetch metadata-known 2025 snapshot only; no name guess, 404 no retries.
set -euo pipefail
: "${KAGGLE_API_TOKEN:?KAGGLE_API_TOKEN_REQUIRED}"
year="${1:?YEAR}"
out="${2:?DESTINATION_DIRECTORY}"
test "$year" = "2025" || { echo "ONLY_TARGET_YEAR_2025_ALLOWED" >&2; exit 2; }
dataset="pino1945/keiba-ml-snapshot-bf811fa2eab73db0"
generation="bf811fa2eab73db0"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
list="$out/kaggle-files.json"
manifest="$out/manifest.json"
resolved="$out/resolved-2025.tsv"
transient() {
  local label="$1"; shift
  local n rc logfile="$out/tmp.$BASHPID.$RANDOM"
  for n in 1 2 3; do
    rc=0
    "$@" > "$logfile.out" 2> "$logfile.err" || rc=$?
    if [ "$rc" -eq 0 ]; then
      if [ "$label" = "list" ]; then cat "$logfile.out" > "$list"; else cat "$logfile.out"; fi
      rm -f "$logfile.out" "$logfile.err"
      return 0
    fi
    if grep -Eiq '404|not found' "$logfile.out" "$logfile.err"; then
      echo "KAGGLE_METADATA_DRIFT_404_STOP label=$label" >&2
      rm -f "$logfile.out" "$logfile.err"; return "$rc"
    fi
    if ! grep -Eiq '429|502|503|504|too many requests|bad gateway|service unavailable|gateway timeout|timed out|connection reset|temporary failure' "$logfile.out" "$logfile.err"; then
      echo "KAGGLE_NON_TRANSIENT_STOP label=$label rc=$rc" >&2
      rm -f "$logfile.out" "$logfile.err"; return "$rc"
    fi
    rm -f "$logfile.out" "$logfile.err"
    if [ "$n" -ge 3 ]; then echo "KAGGLE_RETRIES_EXHAUSTED label=$label" >&2; return "$rc"; fi
    sleep $(( 8 * (2 ** (n-1)) ))
  done
}
if [ ! -s "$list" ]; then
  transient list kaggle datasets files "$dataset" --page-size 200 --format 'json(name,size)'
fi
python3 - "$list" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
assert isinstance(rows,list) and all(isinstance(x,dict) for x in rows)
names=[r.get("name") for r in rows]
assert len(names)==len(set(names)) and "manifest.json" in names, "INVALID_KAGGLE_METADATA"
print("KAGGLE_FILE_METADATA_VERIFIED",len(rows),flush=True)
PY
if [ ! -s "$manifest" ]; then
  transient manifest kaggle datasets download "$dataset" -f manifest.json -p "$out" --unzip --quiet
fi
test -s "$manifest"
python3 research/resolve_snapshot_manifest_remote_v1.py --manifest "$manifest" \
    --files-json "$list" --generation-id "$generation" --years 2025 --output "$resolved"
IFS=$'\t' read -r actualyear logical remote mode sha rows mindate maxdate < "$resolved"
test "$actualyear" = "2025" && test -n "$remote"
python3 - "$list" "$remote" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
names=[x for x in data if x["name"]==sys.argv[2]]
assert len(names)==1
raw_size=names[0].get("size")
print("KAGGLE_2025_REPORTED_SIZE",repr(raw_size),type(raw_size).__name__,flush=True)
import re,shutil
if type(raw_size) is int:
    size=raw_size
elif isinstance(raw_size,str):
    value=raw_size.strip().replace(",","")
    match=re.fullmatch(r"(?i)([0-9]+(?:\\.[0-9]+)?)\\s*(B|KB|KIB|MB|MIB|GB|GIB)?",value)
    assert match is not None, "UNRECOGNIZED_KAGGLE_SIZE_STOP"
    unit=(match[2] or "B").upper()
    multiplier={"B":1,"KB":1000,"KIB":1024,"MB":1_000_000,"MIB":1048576,
                "GB":1_000_000_000,"GIB":1073741824}[unit]
    size=int(float(match[1])*multiplier)
else:
    raise ValueError("KAGGLE_SIZE_NOT_REPORTED_STOP")
free=shutil.disk_usage(sys.argv[1] and "/tmp").free
assert 0<size<=3_500_000_000, "SOURCE_EXCEEDS_3_5GB_FREE_EPHEMERAL_CAP"
assert size+3_000_000_000<free, "SOURCE_EXCEEDS_FREE_RUNNER_CAPACITY"
print("KAGGLE_2025_REMOTE_SIZE_WITHIN_GUARD",size,flush=True)
PY
if [ ! -s "$out/$remote" ]; then
  transient snapshot kaggle datasets download "$dataset" -f "$remote" -p "$out" --unzip --quiet
else
  echo "KAGGLE_2025_RUNNER_FILE_REUSED"
fi
test -s "$out/$remote"
python3 - "$out/$remote" "$logical" "$remote" "$sha" <<'PY'
import hashlib,sys
path,logical,remote,sha=sys.argv[1:]
if logical==remote and remote.endswith(".gz") and sha:
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for data in iter(lambda:f.read(1024*1024),b""):digest.update(data)
    assert digest.hexdigest()==sha, "2025_SNAPSHOT_SHA256_DRIFT"
    print("KAGGLE_SNAPSHOT_2025_SHA_VERIFIED",flush=True)
else:
    print("KAGGLE_SNAPSHOT_2025_COMPRESSION_EQUIVALENCE_NEEDS_FULL_ROW_AUDIT",flush=True)
PY
printf '%s\n' "$out/$remote"
