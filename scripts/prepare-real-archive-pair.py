#!/usr/bin/env python3
"""Prepare a bounded real BACKFILL smoke input, preserving original race bytes.

Usage: python3 scripts/prepare-real-archive-pair.py PATH_TO_PARTIAL_CLONE OUTPUT_DIR
The origin clone must be blobless sparse: only two dated race gzip packs and
one manifest are materialized, with a remote blob-size guard before checkout.
"""
import datetime,gzip,json,os,re,shutil,subprocess,sys
from pathlib import Path

DATE=re.compile(r"data/daily/(2025-\d\d-\d\d)\.jsonl\.gz$")
MAX_DAY=35_000_000
MAX_MANIFEST=100_000_000

def tree(repo,*items):
    return subprocess.check_output(
        ["git","-C",str(repo),"ls-tree","-r","-l","HEAD",*items],
        text=True)

def parse_tree_item(line):
    seg=line.split(None,4)
    if len(seg)!=5:return None
    kind=seg[1]
    try:size=int(seg[3])
    except ValueError:return None
    return {"kind":kind,"sha":seg[2],"size":size,"path":seg[4]}

def main():
    if len(sys.argv)!=3:raise SystemExit("usage: prepare-real-archive-pair.py CLONE OUT")
    root=Path(sys.argv[1]).resolve()
    output=Path(sys.argv[2]).resolve()
    items=[parse_tree_item(line) for line in tree(root,"data/daily").splitlines()]
    items=[x for x in items if x and x["kind"]=="blob" and DATE.fullmatch(x["path"]) and x["size"]<=MAX_DAY]
    dates={DATE.fullmatch(x["path"]).group(1):x for x in items}
    pair=None
    for date in sorted(dates):
        tomorrow=(datetime.date.fromisoformat(date)+datetime.timedelta(days=1)).isoformat()
        if tomorrow in dates:
            pair=(date,tomorrow)
            break
    if pair is None:raise SystemExit("NO_CONSECUTIVE_REAL_ARCHIVE_DAYS_2025")
    mf=[parse_tree_item(line) for line in tree(root,"data/manifest.json").splitlines()]
    if len(mf)!=1 or mf[0] is None or mf[0]["size"]>MAX_MANIFEST:
        raise SystemExit("MANIFEST_EXCEEDS_BLOB_GUARD")
    source_date,target_date=pair
    patterns=["/data/manifest.json",
              f"/data/daily/{source_date}.jsonl.gz",
              f"/data/daily/{target_date}.jsonl.gz"]
    subprocess.run(["git","-C",str(root),"sparse-checkout","set","--no-cone","--stdin"],
        input="\n".join(patterns)+"\n",text=True,check=True)
    manifest=json.loads((root/"data/manifest.json").read_text(encoding="utf-8"))
    for d in pair:
        item=manifest.get("days",{}).get(d)
        if not item or item.get("status")!="SUCCESS":
            raise SystemExit("REAL_MANIFEST_DAY_NOT_SUCCESS "+d)
        if item.get("file")!=f"data/daily/{d}.jsonl.gz":
            raise SystemExit("REAL_MANIFEST_PACK_DRIFT "+d)
    raw=(root/f"data/daily/{target_date}.jsonl.gz").read_bytes()
    races=[]
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        if line.strip():races.append(json.loads(line))
    candidates=[r for r in races if re.fullmatch(r"\d{12}",str(r.get("race",{}).get("race_id","")))
                and sum(1 for e in r.get("entries",[]) if str(e.get("entry_status","")).upper()
                not in ("SCRATCHED","EXCLUDED"))>=3]
    if not candidates:raise SystemExit("REAL_TARGET_RACE_WITH_STARTERS_NOT_FOUND")
    target=sorted(candidates,key=lambda r:r["race"]["race_id"])[-1]
    rid=str(target["race"]["race_id"])
    output.joinpath("data","daily").mkdir(parents=True,exist_ok=True)
    for d in pair:
        shutil.copy2(root/f"data/daily/{d}.jsonl.gz",output/f"data/daily/{d}.jsonl.gz")
    selected={"days":{d:manifest["days"][d] for d in pair},
              "non_meeting_days":{},
              "schedule_exception_days":{}}
    output.joinpath("data","manifest.json").write_text(
        json.dumps(selected,ensure_ascii=False),encoding="utf-8")
    print("REAL_BACKFILL_PAIR_PREPARED",json.dumps({
        "source_date":source_date,"target_date":target_date,
        "race_id":rid,"expected_starters":len(target.get("entries",[])),
        "source_pack_bytes":dates[source_date]["size"],
        "target_pack_bytes":dates[target_date]["size"],
        "source_git_sha":subprocess.check_output(
            ["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    }))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"],"a",encoding="utf-8") as f:
            f.write(f"race_id={rid}\nsource_date={source_date}\ntarget_date={target_date}\n")

if __name__=="__main__":main()
