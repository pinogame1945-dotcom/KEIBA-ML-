#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
from pathlib import Path

YEARS=(2022,2023,2024,2025)
EXPECTED_ALERTS_PER_YEAR=346
EXPECTED_ALERTS_TOTAL=1384
EXPECTED_NOVEL_TOTAL=5577
ALLOWED_OUTSIDERS={
    "当日傾向型","レース構造型","枠・コース型","メンバー構成型","騎手型"
}
REQUIRED={
    "contract","l15_version","year","race_id","gate_alert","gate_score","gate_action",
    "seven_consensus_order","seven_anchor_horse_ids","seven_union_horse_ids",
    "selected_outsiders","novel_horse_ids","candidate_horse_ids",
}


def parse_args():
    p=argparse.ArgumentParser(description="Validate and manifest frozen L1.5 FIX V1 ledgers.")
    p.add_argument("--contract",default="contracts/l15-fixed-v1.json")
    p.add_argument("--ledger-dir",required=True)
    p.add_argument("--manifest-out",required=True)
    return p.parse_args()


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as fh:
        while True:
            b=fh.read(1024*1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rows(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main():
    a=parse_args()
    contract=json.loads(Path(a.contract).read_text(encoding="utf-8"))
    if contract.get("contract")!="L15_FIXED_V1" or contract.get("status")!="FROZEN":
        raise SystemExit("contract must be frozen L15_FIXED_V1")
    if contract.get("immutability",{}).get("mutate_v1") is not False:
        raise SystemExit("immutability guard broken")
    if contract.get("historical_scope",{}).get("locked_years")!=[2026]:
        raise SystemExit("2026 lock guard broken")

    root=Path(a.ledger_dir)
    files=[]
    total_rows=0
    total_novel=0
    seen=set()
    outsider_usage={name:0 for name in sorted(ALLOWED_OUTSIDERS)}

    for year in YEARS:
        path=root/f"y{year}.jsonl.gz"
        if not path.exists():
            raise SystemExit(f"missing fixed ledger: {path}")
        count=0
        novel=0
        for row in rows(path):
            count+=1
            total_rows+=1
            rid=str(row.get("race_id") or "")
            if not rid or rid in seen:
                raise SystemExit(f"missing/duplicate race_id: {rid}")
            seen.add(rid)
            if set(row)!=REQUIRED:
                raise SystemExit(
                    f"output schema drift race={rid}: missing={sorted(REQUIRED-set(row))} "
                    f"extra={sorted(set(row)-REQUIRED)}"
                )
            if row["contract"]!="L15_FIXED_OUTPUT_V1" or row["l15_version"]!="1.0.0":
                raise SystemExit(f"version drift race={rid}")
            if int(row["year"])!=year or row["gate_alert"] is not True:
                raise SystemExit(f"fixed ledger must contain Gate alerts only race={rid}")
            if row["gate_action"]!="INTERVENE_FULL_K2":
                raise SystemExit(f"unexpected Gate action race={rid}")
            selected=list(row["selected_outsiders"])
            if len(selected)!=2 or len(set(selected))!=2:
                raise SystemExit(f"K2 selection drift race={rid}: {selected}")
            if not set(selected).issubset(ALLOWED_OUTSIDERS):
                raise SystemExit(f"unknown outsider race={rid}: {selected}")
            for label in selected:
                outsider_usage[label]+=1
            seven=list(row["seven_consensus_order"])
            anchors=list(row["seven_anchor_horse_ids"])
            union=set(row["seven_union_horse_ids"])
            novel_ids=list(row["novel_horse_ids"])
            candidates=list(row["candidate_horse_ids"])
            if len(seven)!=len(set(seven)) or set(seven)!=union:
                raise SystemExit(f"seven consensus/union drift race={rid}")
            if anchors!=seven[:2] or len(anchors)!=2:
                raise SystemExit(f"anchor drift race={rid}")
            if set(novel_ids)&union:
                raise SystemExit(f"novel horse overlaps seven union race={rid}")
            if len(novel_ids)!=len(set(novel_ids)):
                raise SystemExit(f"duplicate novel horse race={rid}")
            if candidates!=seven+novel_ids or len(candidates)!=len(set(candidates)):
                raise SystemExit(f"candidate ordering/uniqueness drift race={rid}")
            if row["gate_score"] is None:
                raise SystemExit(f"alert gate score missing race={rid}")
            novel+=len(novel_ids)
            total_novel+=len(novel_ids)

        if count!=EXPECTED_ALERTS_PER_YEAR:
            raise SystemExit(f"year alert count regression y{year}: {count}")
        files.append({
            "year":year,
            "file":path.name,
            "rows":count,
            "novel_horses":novel,
            "bytes":path.stat().st_size,
            "sha256":sha256(path),
        })

    if total_rows!=EXPECTED_ALERTS_TOTAL:
        raise SystemExit(f"total alert count regression: {total_rows}")
    if total_novel!=EXPECTED_NOVEL_TOTAL:
        raise SystemExit(f"candidate inflation regression: novel={total_novel} expected={EXPECTED_NOVEL_TOTAL}")

    manifest={
        "contract":"L15_FIXED_LEDGER_MANIFEST_V1",
        "l15_contract":"L15_FIXED_V1",
        "l15_version":"1.0.0",
        "status":"FROZEN",
        "years":list(YEARS),
        "locked_years":[2026],
        "files":files,
        "total_alert_rows":total_rows,
        "total_novel_horses":total_novel,
        "selected_outsider_calls":total_rows*2,
        "outsider_usage":outsider_usage,
        "non_alert_semantics":"Race IDs absent from the yearly alert ledger are PASS_SEVEN_ONLY.",
        "downstream_semantics":"L2 must consume candidate_horse_ids plus seven_anchor_horse_ids; L1.5 must not use odds/popularity/payout.",
        "frozen_sources":contract["frozen_sources"],
    }
    out=Path(a.manifest_out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L15_FIXED_V1_VALID")
    print(json.dumps({
        "alerts":total_rows,
        "novel_horses":total_novel,
        "selected_outsider_calls":total_rows*2,
        "manifest":str(out),
    },ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()
