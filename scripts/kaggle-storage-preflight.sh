#!/usr/bin/env bash
set -euo pipefail

SAFETY_LIMIT_BYTES="${KAGGLE_STORAGE_SAFETY_LIMIT_BYTES:-193273528320}" # 180 GiB

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

python - "$SAFETY_LIMIT_BYTES" <<'PY'
import json,os,subprocess,sys,time
limit=int(sys.argv[1])

def run(args,label):
    delays=(0,5,10,20,40,60)
    last=""
    for attempt,delay in enumerate(delays,1):
        if delay:
            time.sleep(delay)
        p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode==0:
            return p.stdout.strip()
        last=(p.stderr or p.stdout or "").strip()
        retryable=("429" in last or "Too Many Requests" in last or "502" in last or "503" in last)
        print(f"KAGGLE_PREFLIGHT_RETRY label={label} attempt={attempt} retryable={retryable}",flush=True)
        if not retryable:
            break
    raise SystemExit(f"Could not verify Kaggle storage usage; refusing to start research run.\n{last}")

refs=[]
for page in range(1,1001):
    raw=run(["kaggle","datasets","list","--mine","--page",str(page),"--format","json(ref)"],f"list-page-{page}")
    if not raw or raw.lower().startswith("no datasets found"):
        break
    rows=json.loads(raw)
    if not isinstance(rows,list):
        raise SystemExit("unexpected Kaggle dataset list payload")
    page_refs=[str(x.get("ref") or "").strip() for x in rows if str(x.get("ref") or "").strip()]
    refs.extend(page_refs)
    if not page_refs:
        break
else:
    raise SystemExit("Kaggle dataset pagination safety stop")

grand=0
for ref in sorted(set(refs)):
    raw=run(["kaggle","datasets","files",ref,"--page-size","1000","--format","json(name,size)"],f"files-{ref}")
    if not raw or raw.lower().startswith("no files found"):
        rows=[]
    else:
        rows=json.loads(raw)
    if not isinstance(rows,list):
        raise SystemExit(f"unexpected Kaggle files payload for {ref}")
    subtotal=0
    for row in rows:
        try:
            subtotal += int(row.get("size") or 0)
        except (TypeError,ValueError):
            raise SystemExit(f"invalid file size in {ref}")
    grand += subtotal

print(f"Owned Kaggle dataset count: {len(set(refs))}")
print(f"Owned Kaggle dataset bytes: {grand}")
print(f"Research storage ceiling: {limit}")
if grand>=limit:
    raise SystemExit("Owned Kaggle storage is already at/above the 180 GiB research ceiling; refusing to start.")
print(f"KAGGLE_STORAGE_PREFLIGHT_OK current_bytes={grand} limit_bytes={limit}")
gh=os.environ.get("GITHUB_ENV")
if gh:
    with open(gh,"a",encoding="utf-8") as fh:
        fh.write("KAGGLE_STORAGE_PREFLIGHT_OK=1\n")
        fh.write(f"KAGGLE_STORAGE_PREFLIGHT_BYTES={grand}\n")
go=os.environ.get("GITHUB_OUTPUT")
if go:
    with open(go,"a",encoding="utf-8") as fh:
        fh.write("ok=1\n")
        fh.write(f"bytes={grand}\n")
PY
