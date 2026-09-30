#!/usr/bin/env python3
import argparse,csv,gzip,json,math,re
from collections import Counter,defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import final_odds_tuple,payout_map

YEARS=(2022,2023,2024,2025)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def num(v):
    try:
        if isinstance(v,str): v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else None
    except: return None

def parse_key(k):
    s=str(k)
    if not re.fullmatch(r"\d{6}",s): return None
    t=tuple(int(s[i:i+2]) for i in (0,2,4))
    if any(x<=0 for x in t) or len(set(t))!=3: return None
    return t

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def main():
    a=parse_args(); root=Path(a.backfill_root); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    details=[]; yearly=[]; samples=[]
    for year in YEARS:
        c=Counter()
        for day in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=day.name[:10]
            op=root/"data"/"odds"/"daily"/day.name
            if not op.exists():
                c["missing_odds_day"]+=1; continue
            odds_by={str(x.get("race_id") or ""):x for x in read_jsonl_gz(op)}
            for pack in read_jsonl_gz(day):
                rid=str((pack.get("race") or {}).get("race_id") or "")
                c["races"]+=1
                o=odds_by.get(rid)
                if o is None:
                    c["missing_odds_row"]+=1; continue
                entries=pack.get("entries") or []
                started={int(e["horse_number"]) for e in entries if str(e.get("entry_status") or "STARTED")=="STARTED" and e.get("horse_number")}
                nonstarted={int(e["horse_number"]) for e in entries if str(e.get("entry_status") or "STARTED")!="STARTED" and e.get("horse_number")}
                g=(o.get("odds") or {}).get("8") or {}
                valid_keys=[]; positive={}
                comma_positive=0
                for k,raw in g.items():
                    t=parse_key(k)
                    if t is None: continue
                    valid_keys.append(t)
                    ft=final_odds_tuple(raw)
                    if not ft: continue
                    v=num(ft[0])
                    if v is not None and v>0:
                        positive[t]=v
                        if isinstance(ft[0],str) and "," in ft[0]: comma_positive+=1
                pos_union={n for t in positive for n in t}
                expected_union=len(pos_union)*(len(pos_union)-1)*(len(pos_union)-2) if len(pos_union)>=3 else 0
                current_complete=(len(positive)==expected_union and expected_union>0)
                expected_started=len(started)*(len(started)-1)*(len(started)-2) if len(started)>=3 else 0
                positive_started={t:v for t,v in positive.items() if set(t)<=started}
                started_complete=(len(positive_started)==expected_started and expected_started>0)
                positive_nonstarted={t:v for t,v in positive.items() if not set(t)<=started}
                raw_started={t for t in valid_keys if set(t)<=started}
                missing_started=max(0,expected_started-len(positive_started))
                raw_missing_started=max(0,expected_started-len(raw_started))
                payouts,present=payout_map(pack)
                win_keys=[k[1] for k,v in payouts.items() if k[0]=="TRIFECTA" and v>0]
                win=win_keys[0] if win_keys else None
                winner_priced=bool(win in positive) if win else False
                group_summary=(o.get("group_summary") or {}).get("8") or {}
                reason=(
                    "CURRENT_COMPLETE" if current_complete else
                    "FALSE_REJECT_NONSTARTED_RESIDUAL" if started_complete and positive_nonstarted else
                    "STARTED_COMPLETE_OTHER" if started_complete else
                    "STARTED_PARTIAL"
                )
                c[reason]+=1
                if winner_priced: c["winner_priced"]+=1
                if comma_positive: c["races_with_comma"]+=1
                c["comma_positive_rows"]+=comma_positive
                c["positive_rows"]+=len(positive)
                c["positive_started_rows"]+=len(positive_started)
                c["positive_nonstarted_rows"]+=len(positive_nonstarted)
                c["missing_started_rows"]+=missing_started
                c["raw_missing_started_rows"]+=raw_missing_started
                row={
                    "year":year,"date":date,"race_id":rid,"reason":reason,
                    "started_horses":len(started),"nonstarted_horses":len(nonstarted),
                    "raw_group8_rows":len(g),"valid_key_rows":len(valid_keys),
                    "summary_rows":group_summary.get("rows"),"summary_priced_rows":group_summary.get("priced_rows"),
                    "positive_rows":len(positive),"comma_positive_rows":comma_positive,
                    "positive_union_horses":len(pos_union),"current_expected_rows":expected_union,
                    "current_complete":int(current_complete),
                    "expected_started_rows":expected_started,"positive_started_rows":len(positive_started),
                    "missing_started_rows":missing_started,"raw_started_rows":len(raw_started),
                    "raw_missing_started_rows":raw_missing_started,
                    "positive_nonstarted_rows":len(positive_nonstarted),
                    "winner_priced":int(winner_priced),
                }
                details.append(row)
                if reason!="CURRENT_COMPLETE" and len(samples)<200:
                    samples.append(row)
        yearly.append({"year":year,**dict(c)})
    write_csv(out/"race-detail.csv",details)
    write_csv(out/"yearly-summary.csv",yearly)
    write_csv(out/"noncomplete-samples.csv",samples)
    summary={
        "contract":"TRIFECTA_FULLFIELD_COVERAGE_AUDIT_V1",
        "years":list(YEARS),
        "yearly":yearly,
        "definitions":{
            "current_complete":"positive decoded trifecta rows equals P(number of horses appearing in any positive row,3)",
            "started_complete":"positive decoded trifecta rows restricted to race-pack entry_status STARTED equals P(number of STARTED horses,3)",
            "false_reject_nonstarted_residual":"current completeness fails but STARTED-only completeness passes and positive odds still reference non-started horses"
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__": main()
