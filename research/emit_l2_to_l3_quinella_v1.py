#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import re
from pathlib import Path

CONTRACT="L2_TO_L3_QUINELLA_V1"
VERSION=1
FORBIDDEN={
    "stake_yen","recommended_stake_yen","bet_amount_yen","budget_share","kelly_fraction",
    "finish_position","winner","hit","target","return_yen","return_yen_per100",
    "payout","payouts","profit_yen","roi_pct"
}
REQUIRED_INPUT={
    "race_id","race_date","pair_horse_ids","pair_numbers","law_id",
    "market_snapshot_id","market_snapshot_kind","odds","market_rank",
    "model_rank","rank_upgrade","l17_rank_score","field_size",
    "l2_model_version"
}
OPTIONAL_COPY=[
    "market_observed_at_utc","seconds_to_post","market_q_norm","market_aware_score",
    "pair_prob_product","surface","distance_m","venue","race_number",
    "scheduled_post_time_utc","source_ref","source_sha"
]
CONSISTENT_FIELDS=[
    "race_date","market_snapshot_kind","odds","market_rank","model_rank",
    "rank_upgrade","l17_rank_score","field_size","l2_model_version",
    *OPTIONAL_COPY
]

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--input",action="append",required=True,help="CSV; may repeat. One row per law-selected ticket.")
    p.add_argument("--law-catalog",default="contracts/l2-quinella-law-catalog-v1.json")
    p.add_argument("--output",required=True)
    p.add_argument("--generated-at-utc",required=True)
    return p.parse_args()

def load_catalog(path):
    c=json.load(open(path,encoding="utf-8"))
    if c.get("contract")!="L2_QUINELLA_LAW_CATALOG_V1":
        raise SystemExit("bad law catalog contract")
    enabled={x["law_id"] for x in c["laws"] if x.get("enabled_for_handoff") is True}
    all_ids={x["law_id"] for x in c["laws"]}
    rejected={x["law_id"] for x in c["laws"] if x.get("status","").startswith("REJECTED")}
    return c,enabled,all_ids,rejected

def read_csvs(paths):
    rows=[]
    for path in paths:
        with open(path,newline="",encoding="utf-8") as f:
            r=csv.DictReader(f)
            fields=set(r.fieldnames or [])
            missing=sorted(REQUIRED_INPUT-fields)
            if missing:
                raise SystemExit(f"missing columns path={path}: {missing}")
            bad=sorted(fields & FORBIDDEN)
            if bad:
                raise SystemExit(f"forbidden outcome/stake columns path={path}: {bad}")
            rows.extend(dict(x) for x in r)
    if not rows:
        raise SystemExit("no input rows")
    return rows

def num(v,name,integer=False):
    if v is None or str(v).strip()=="":
        return None
    try:
        x=float(v)
    except Exception:
        raise SystemExit(f"invalid numeric {name}={v!r}")
    return int(x) if integer else x

def law_sort_key(x):
    m=re.fullmatch(r"LAW(\d+)",x)
    return (0,int(m.group(1))) if m else (1,x)

def parse_horses(row):
    ids=[x.strip() for x in str(row["pair_horse_ids"]).split("|")]
    nums=[x.strip() for x in str(row["pair_numbers"]).split("-")]
    if len(ids)!=2 or len(nums)!=2 or not all(ids) or not all(nums):
        raise SystemExit(f"bad pair encoding race={row['race_id']}")
    pairs=[(ids[i],int(float(nums[i]))) for i in range(2)]
    if pairs[0][0]==pairs[1][0] or pairs[0][1]==pairs[1][1]:
        raise SystemExit(f"duplicate horse in pair race={row['race_id']}")
    pairs.sort(key=lambda x:x[0])
    return [{"horse_id":hid,"horse_number":n} for hid,n in pairs]

def canonical_key(row,horses):
    return f"QUINELLA:{row['race_id']}:{horses[0]['horse_id']}:{horses[1]['horse_id']}"

def normalize_field(name,value):
    if value is None or str(value).strip()=="":
        return None
    if name in {"odds","market_q_norm","market_aware_score","pair_prob_product"}:
        return num(value,name)
    if name in {"market_rank","model_rank","rank_upgrade","l17_rank_score","field_size",
                "seconds_to_post","distance_m","race_number"}:
        return num(value,name,integer=True)
    return str(value)

def compatible(base,row):
    for k in CONSISTENT_FIELDS:
        a=normalize_field(k,base.get(k))
        b=normalize_field(k,row.get(k))
        if a!=b:
            raise SystemExit(f"duplicate ticket signal drift key={base['_merge_key']} field={k} {a!r}!={b!r}")

def validate_record(rec):
    if rec["contract"]!=CONTRACT or rec["version"]!=VERSION:
        raise SystemExit("contract/version drift")
    if rec["bet_type"]!="QUINELLA":
        raise SystemExit("bet_type drift")
    if len(rec["horses"])!=2:
        raise SystemExit("quinella requires two horses")
    if rec["horses"]!=sorted(rec["horses"],key=lambda h:h["horse_id"]):
        raise SystemExit("horses not canonical")
    if rec["law_count"]!=len(rec["selected_by_laws"]) or len(set(rec["selected_by_laws"]))!=rec["law_count"]:
        raise SystemExit("law merge drift")
    if rec["market_snapshot_kind"]=="LIVE" and not rec.get("market_observed_at_utc"):
        raise SystemExit("LIVE market snapshot requires market_observed_at_utc")
    if rec["market_snapshot_kind"]=="HISTORICAL_UNKNOWN_TIME" and rec.get("market_observed_at_utc"):
        raise SystemExit("HISTORICAL_UNKNOWN_TIME must not pretend to know observed_at")
    if rec["market_snapshot_kind"] not in {"LIVE","HISTORICAL_FINAL","HISTORICAL_UNKNOWN_TIME"}:
        raise SystemExit("invalid market_snapshot_kind")
    if float(rec["odds"])<=0:
        raise SystemExit("odds must be positive")
    if int(rec["market_rank"])<1 or int(rec["model_rank"])<1 or int(rec["field_size"])<2:
        raise SystemExit("invalid positive rank/field_size")
    if int(rec["rank_upgrade"])!=int(rec["market_rank"])-int(rec["model_rank"]):
        raise SystemExit("rank_upgrade mismatch")
    if any(k in rec for k in FORBIDDEN):
        raise SystemExit("forbidden field emitted")

def main():
    a=args()
    catalog,enabled,all_ids,rejected=load_catalog(a.law_catalog)
    rows=read_csvs(a.input)

    merged={}
    for row in rows:
        law=str(row["law_id"]).strip()
        if law not in all_ids:
            raise SystemExit(f"unknown law_id={law}")
        if law in rejected:
            raise SystemExit(f"rejected law cannot be handed to L3: {law}")
        if law not in enabled:
            raise SystemExit(f"law not enabled for handoff: {law}")
        horses=parse_horses(row)
        ticket_key=canonical_key(row,horses)
        merge_key=(str(row["race_id"]),str(row["market_snapshot_id"]),ticket_key)
        if merge_key not in merged:
            base=dict(row)
            base["_merge_key"]=merge_key
            base["_horses"]=horses
            base["_ticket_key"]=ticket_key
            base["_laws"]={law}
            merged[merge_key]=base
        else:
            compatible(merged[merge_key],row)
            merged[merge_key]["_laws"].add(law)

    records=[]
    for _,row in sorted(merged.items(),key=lambda kv:kv[0]):
        rec={
            "contract":CONTRACT,
            "version":VERSION,
            "generated_at_utc":a.generated_at_utc,
            "race_id":str(row["race_id"]),
            "race_date":str(row["race_date"]),
            "market_snapshot_id":str(row["market_snapshot_id"]),
            "market_snapshot_kind":str(row["market_snapshot_kind"]),
            "bet_type":"QUINELLA",
            "ticket_key":row["_ticket_key"],
            "horses":row["_horses"],
            "selected_by_laws":sorted(row["_laws"],key=law_sort_key),
            "law_count":len(row["_laws"]),
            "odds":num(row["odds"],"odds"),
            "market_rank":num(row["market_rank"],"market_rank",True),
            "model_rank":num(row["model_rank"],"model_rank",True),
            "rank_upgrade":num(row["rank_upgrade"],"rank_upgrade",True),
            "l17_rank_score":num(row["l17_rank_score"],"l17_rank_score",True),
            "field_size":num(row["field_size"],"field_size",True),
            "l2_model_version":str(row["l2_model_version"]),
            "law_catalog_version":int(catalog["version"]),
        }
        for k in OPTIONAL_COPY:
            v=normalize_field(k,row.get(k))
            if v is not None:
                rec[k]=v
        validate_record(rec)
        records.append(rec)

    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if out.suffix==".gz" else open
    with opener(out,"wt",encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n")

    print(json.dumps({
        "contract":CONTRACT,
        "input_rows":len(rows),
        "output_tickets":len(records),
        "duplicates_merged":len(rows)-len(records),
        "enabled_laws":sorted(enabled,key=law_sort_key),
        "output":str(out),
    },ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
