#!/usr/bin/env bash
set -euo pipefail

SAFETY_LIMIT_BYTES="${KAGGLE_STORAGE_SAFETY_LIMIT_BYTES:-193273528320}" # 180 GiB

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "KAGGLE_API_TOKEN is required" >&2
  exit 2
fi

python - "$SAFETY_LIMIT_BYTES" <<'PY'
import json,subprocess,sys
limit=int(sys.argv[1])

def run(args):
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode:
        raise SystemExit("Could not verify Kaggle storage usage; refusing to start research run.\n"+(p.stderr or p.stdout))
    return p.stdout.strip()

refs=[]
for page in range(1,1001):
    raw=run(["kaggle","datasets","list","--mine","--page",str(page),"--format","json(ref)"])
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
    raw=run(["kaggle","datasets","files",ref,"--page-size","1000","--format","json(name,size)"])
    if not raw or raw.lower().startswith("no files found"):
        rows=[]
    else:
        rows=json.loads(raw)
    if not isinstance(rows,list):
        raise SystemExit(f"unexpected Kaggle files payload for {ref}")
    subtotal=0
    for row in rows:
        try: subtotal+=int(row.get("size") or 0)
        except (TypeError,ValueError): raise SystemExit(f"invalid file size in {ref}")
    grand+=subtotal

print(f"Owned Kaggle dataset count: {len(set(refs))}")
print(f"Owned Kaggle dataset bytes: {grand}")
print(f"Research storage ceiling: {limit}")
if grand>=limit:
    raise SystemExit("Owned Kaggle storage is already at/above the 180 GiB research ceiling; refusing to start.")
print(f"KAGGLE_STORAGE_PREFLIGHT_OK current_bytes={grand} limit_bytes={limit}")
with open(__import__("os").environ["GITHUB_ENV"],"a",encoding="utf-8") as fh:
    fh.write("KAGGLE_STORAGE_PREFLIGHT_OK=1\n")
    fh.write(f"KAGGLE_STORAGE_PREFLIGHT_BYTES={grand}\n")
PY
