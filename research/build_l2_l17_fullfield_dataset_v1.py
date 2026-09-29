#!/usr/bin/env python3
import argparse,gzip,json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    YEARS,TEMPLATE_TO_BET,load_fixed_ledgers,load_router,load_day,load_odds_day,
    horse_number_map,decode_odds,payout_map,canonical_numbers,generate_templates
)

BET_TYPES=("QUINELLA","EXACTA","TRIO","TRIFECTA")
TEMPLATES=tuple(t for t,b in TEMPLATE_TO_BET.items() if b in BET_TYPES)

def parse_args():
    p=argparse.ArgumentParser(description="Build L2 ticket dataset from L1.7 full-field consensus.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--l17-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out

def open_text(path):
    return gzip.open(path,"rt",encoding="utf-8") if str(path).endswith(".gz") else open(path,"rt",encoding="utf-8")

def load_l17(path,year):
    out={}
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("contract")!="L17_SEVEN_KING_FULLFIELD_OUTPUT_V1":
                raise ValueError(f"bad L1.7 contract y={year}")
            if int(r.get("year"))!=year:
                raise ValueError(f"L1.7 year drift expected={year} row={r.get('year')}")
            rid=str(r.get("race_id") or "")
            if not rid or rid in out:
                raise ValueError(f"L1.7 bad/duplicate race_id y={year} rid={rid}")
            horses=r.get("horses") or []
            order=[str(x) for x in (r.get("consensus_order") or [])]
            if len(horses)!=int(r.get("field_size") or 0) or len(order)!=len(horses):
                raise ValueError(f"L1.7 field size drift race={rid}")
            if len(set(order))!=len(order):
                raise ValueError(f"L1.7 duplicate horse race={rid}")
            if {str(x.get("horse_id")) for x in horses}!=set(order):
                raise ValueError(f"L1.7 horse/order mismatch race={rid}")
            out[rid]=r
    return out

def main():
    a=parse_args()
    router_paths=parse_paths(a.router_year)
    l17_paths=parse_paths(a.l17_year)
    if set(router_paths)!=set(YEARS) or set(l17_paths)!=set(YEARS):
        raise SystemExit("year path mismatch")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(router_paths[y],y) for y in YEARS}
    l17={y:load_l17(l17_paths[y],y) for y in YEARS}

    race_owner={}
    date_to_races=defaultdict(list)
    expected={}
    for y in YEARS:
        alerts=set(fixed[y])
        normal=[rid for rid in routers[y] if rid not in alerts]
        expected[y]=len(normal)
        missing=set(normal)-set(l17[y])
        if missing:
            raise SystemExit(f"L1.7 missing normal races y={y} sample={sorted(missing)[:10]}")
        for rid in normal:
            date=str(routers[y][rid].get("race_date") or "")[:10]
            race_owner[rid]=y
            date_to_races[date].append(rid)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    template_rows=defaultdict(int)
    template_hits=defaultdict(int)
    template_races=defaultdict(set)
    missing_odds=defaultdict(int)
    processed=set()
    field_hist=defaultdict(int)
    outside6_ticket_rows=defaultdict(int)

    with ExitStack() as stack:
        handles={
            t:stack.enter_context(gzip.open(out/f"{t}.jsonl.gz","wt",encoding="utf-8"))
            for t in TEMPLATES
        }
        root=Path(a.backfill_root)
        for di,date in enumerate(sorted(date_to_races),1):
            wanted=set(date_to_races[date])
            day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
            odds_day=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
            for rid in sorted(wanted):
                y=race_owner[rid]
                pack=day.get(rid); odds_record=odds_day.get(rid)
                if pack is None or odds_record is None:
                    raise SystemExit(f"missing market/race row race={rid}")
                rec=l17[y][rid]
                order=[str(x) for x in rec["consensus_order"]]
                if len(order)<2:
                    raise SystemExit(f"too-small field race={rid}")
                field_hist[len(order)]+=1
                horse_no=horse_number_map(pack)
                if any(h not in horse_no for h in order):
                    raise SystemExit(f"L1.7 horse missing from race pack race={rid}")
                l17_num={str(x["horse_id"]):x.get("horse_number") for x in rec["horses"]}
                for h in order:
                    try: n=int(l17_num.get(h))
                    except (TypeError,ValueError): n=0
                    if n>0 and n!=horse_no[h]:
                        raise SystemExit(f"horse number drift race={rid} horse={h} l17={n} pack={horse_no[h]}")
                rank={h:i+1 for i,h in enumerate(order)}
                odds_map=decode_odds(odds_record)
                payouts,present=payout_map(pack)
                templates=generate_templates(order,[],order)

                for t in TEMPLATES:
                    bet=TEMPLATE_TO_BET[t]
                    if bet not in present:
                        continue
                    for hids in templates.get(t,[]):
                        nums=[horse_no[h] for h in hids]
                        key=(bet,canonical_numbers(bet,nums))
                        odd=odds_map.get(key)
                        if odd is None:
                            missing_odds[t]+=1
                            continue
                        ret=float(payouts.get(key,0.0))
                        ranks=[rank[h] for h in hids]
                        row={
                            "contract":"L2_L17_FULLFIELD_TICKET_V1",
                            "year":y,
                            "race_id":rid,
                            "race_date":date,
                            "bet_type":bet,
                            "template":t,
                            "selection_key":"-".join(map(str,key[1])),
                            "selection_numbers":"-".join(map(str,nums)),
                            "selection_horse_ids":"|".join(hids),
                            "hit":ret>0,
                            "return_yen_per100":ret,
                            "odds":odd,
                            "gate_alert":0,
                            "ticket_novel_count":0,
                            "field_size":len(order),
                            "max_l17_rank":max(ranks),
                            "ticket_outside_top6_count":sum(r>6 for r in ranks),
                        }
                        handles[t].write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
                        template_rows[t]+=1
                        template_hits[t]+=int(ret>0)
                        template_races[t].add(rid)
                        outside6_ticket_rows[t]+=int(any(r>6 for r in ranks))
                processed.add(rid)
            if di%50==0:
                print(f"L17_FULLFIELD_TICKETS_PROGRESS dates={di}/{len(date_to_races)} races={len(processed)}",flush=True)

    expected_total=sum(expected.values())
    if len(processed)!=expected_total:
        raise SystemExit(f"normal coverage drift actual={len(processed)} expected={expected_total}")
    manifest={
        "contract":"L2_L17_FULLFIELD_DATASET_V1",
        "source_l17":"L17_SEVEN_KING_FULLFIELD_OUTPUT_V1",
        "years":list(YEARS),
        "normal_races":{str(y):expected[y] for y in YEARS},
        "races":len(processed),
        "templates":{},
        "field_size_histogram":{str(k):v for k,v in sorted(field_hist.items())},
        "missing_odds_by_template":dict(sorted(missing_odds.items())),
        "odds_used_as_model_feature":False,
        "market_price_stage":"FINAL_ODDS_AFTER_PREDICTION",
        "locked_years":[2026],
    }
    for t in TEMPLATES:
        if template_rows[t]==0:
            raise SystemExit(f"empty fullfield template={t}")
        manifest["templates"][t]={
            "bet_type":TEMPLATE_TO_BET[t],
            "priced_rows":template_rows[t],
            "hits":template_hits[t],
            "races":len(template_races[t]),
            "outside_top6_priced_rows":outside6_ticket_rows[t],
            "file":f"{t}.jsonl.gz",
        }
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_L17_FULLFIELD_DATASET_V1_READY")
    print(json.dumps({
        "races":len(processed),
        "templates":len(TEMPLATES),
        "priced_rows":sum(template_rows.values()),
        "outside_top6_priced_rows":sum(outside6_ticket_rows.values()),
        "field_size_histogram":manifest["field_size_histogram"],
    },ensure_ascii=False,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
