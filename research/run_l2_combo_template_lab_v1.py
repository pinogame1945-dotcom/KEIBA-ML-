#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    canonical_numbers, horse_number_map, payout_map, seven_stats
)

YEARS=(2024,2025)
BETS=("EXACTA","TRIFECTA")
EXPECTED_PER_YEAR=3456

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--selections",required=True)
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def open_text(p):
    return gzip.open(p,"rt",encoding="utf-8") if str(p).endswith(".gz") else open(p,"rt",encoding="utf-8")

def paths(items):
    out={}
    for x in items:
        y,p=x.split(":",1); out[int(y)]=p
    return out

def read_csv(p):
    with open(p,newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def load_routers(ps):
    out={}
    for y in YEARS:
        d={}
        with open_text(ps[y]) as f:
            for line in f:
                if line.strip():
                    r=json.loads(line); d[str(r["race_id"])]=r
        if len(d)!=EXPECTED_PER_YEAR: raise SystemExit(f"router count y{y}={len(d)}")
        out[y]=d
    return out

def load_fixed(root):
    out={}
    for y in YEARS:
        d={}
        with open(Path(root)/f"y{y}.jsonl",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r=json.loads(line); d[str(r["race_id"])]=r
        out[y]=d
    return out

def load_selected(p):
    rows=[]; dates=defaultdict(set)
    for r in read_csv(p):
        if r.get("router")!="DIRECT_BET_TYPE": continue
        y=int(float(r.get("test_year") or 0))
        if y not in YEARS or r.get("bet_type") not in BETS: continue
        rows.append(r); dates[r["race_date"][:10]].add(r["race_id"])
    return rows,dates

def load_packs(root,wanted):
    out={}; base=Path(root)/"data"/"daily"
    for d,rids in sorted(wanted.items()):
        p=base/f"{d}.jsonl.gz"
        if not p.exists(): raise SystemExit(f"missing {p}")
        with gzip.open(p,"rt",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r=json.loads(line); rid=str((r.get("race") or {}).get("race_id") or "")
                    if rid in rids: out[rid]=r
    missing=sorted({x for s in wanted.values() for x in s}-set(out))
    if missing: raise SystemExit(f"missing packs {missing[:5]}")
    return out

def uniq(rows,bet):
    seen=set(); out=[]
    for t in rows:
        if len(set(t))!=len(t): continue
        key=canonical_numbers(bet,t)
        if key in seen: continue
        seen.add(key); out.append(tuple(t))
    return out

def exacta_templates(order):
    out={}
    a1=order[0] if order else None
    if not a1: return out
    for n in (3,4,5,6,8):
        top=order[:min(n,len(order))]
        rest=[x for x in top if x!=a1]
        out[f"EX_A1_TOP{n}_FWD"]=[(a1,x) for x in rest]
        out[f"EX_A1_TOP{n}_REV"]=[(x,a1) for x in rest]
        out[f"EX_A1_TOP{n}_MULTI"]=[z for x in rest for z in ((a1,x),(x,a1))]
    for n in (3,4,5,6):
        top=order[:min(n,len(order))]
        out[f"EX_TOP{n}_BOX"]=list(itertools.permutations(top,2))
    return {k:uniq(v,"EXACTA") for k,v in out.items()}

def trifecta_templates(order):
    out={}
    if len(order)<3: return out
    a1,a2=order[:2]
    for n in (3,4,5,6,8):
        top=order[:min(n,len(order))]
        rest=[x for x in top if x!=a1]
        out[f"TF_A1_FIRST_TOP{n}"]=[(a1,b,c) for b,c in itertools.permutations(rest,2)]
        mates=[x for x in top if x not in {a1,a2}]
        out[f"TF_A12_MULTI_TOP{n}"]=[p for m in mates for p in itertools.permutations((a1,a2,m),3)]
        out[f"TF_A12_TOP2_MATE_TOP{n}"]=[z for m in mates for z in ((a1,a2,m),(a2,a1,m))]
    for n in (3,4,5,6):
        top=order[:min(n,len(order))]
        out[f"TF_TOP{n}_BOX"]=list(itertools.permutations(top,3))
    return {k:uniq(v,"TRIFECTA") for k,v in out.items()}

def candidate_order(router,fixed):
    seven,_=seven_stats(router)
    if fixed:
        cand=[str(x) for x in fixed.get("candidate_horse_ids") or []]
        if not cand: cand=seven
    else: cand=seven
    return cand

def evaluate(template_rows,pack,horse_no,bet):
    payouts,present=payout_map(pack)
    if bet not in present: return None
    wins={nums for (bt,nums),v in payouts.items() if bt==bet and float(v)>0}
    tickets=[]
    for hs in template_rows:
        if any(h not in horse_no for h in hs): continue
        nums=canonical_numbers(bet,[horse_no[h] for h in hs])
        tickets.append(nums)
    tickets=list(dict.fromkeys(tickets))
    if not tickets: return None
    ret=sum(float(payouts.get((bet,t),0.0)) for t in tickets)
    hit=any(t in wins for t in tickets)
    return len(tickets),ret,hit

def write_csv(p,rows):
    if not rows: return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    a=args(); ps=paths(a.router_year)
    if set(ps)!=set(YEARS): raise SystemExit("need 2024,2025 routers")
    selected,wanted=load_selected(a.selections)
    routers=load_routers(ps); fixed=load_fixed(a.fixed_ledger_dir); packs=load_packs(a.backfill_root,wanted)
    stats=defaultdict(lambda:{"races":0,"tickets":0,"hits":0,"stake":0.0,"ret":0.0})
    oracle=defaultdict(lambda:{"races":0,"candidate_hits":0})
    details=[]
    for i,s in enumerate(selected,1):
        y=int(float(s["test_year"])); rid=s["race_id"]; bet=s["bet_type"]
        order=candidate_order(routers[y][rid],fixed[y].get(rid))
        pack=packs[rid]; horse_no=horse_number_map(pack)
        payouts,present=payout_map(pack)
        if bet not in present: continue
        wins={nums for (bt,nums),v in payouts.items() if bt==bet and float(v)>0}
        cand_nums={horse_no[h] for h in order if h in horse_no}
        can=any(set(w).issubset(cand_nums) for w in wins)
        oracle[(y,bet)]["races"]+=1; oracle[(y,bet)]["candidate_hits"]+=int(can)
        templates=exacta_templates(order) if bet=="EXACTA" else trifecta_templates(order)
        for name,rows in templates.items():
            ev=evaluate(rows,pack,horse_no,bet)
            if ev is None: continue
            nt,ret,hit=ev; k=(y,bet,name); d=stats[k]
            d["races"]+=1; d["tickets"]+=nt; d["hits"]+=int(hit); d["stake"]+=100.0*nt; d["ret"]+=ret
            details.append({"year":y,"race_id":rid,"bet_type":bet,"template":name,"tickets":nt,"hit":int(hit),"return_yen":ret})
        if i%250==0: print(f"COMBO_TEMPLATE_PROGRESS {i}/{len(selected)}",flush=True)
    rows=[]
    for (y,bet,name),d in sorted(stats.items()):
        oc=oracle[(y,bet)]
        rows.append({
            "year":y,"bet_type":bet,"template":name,
            "races":d["races"],"tickets":d["tickets"],"avg_tickets_per_race":d["tickets"]/d["races"] if d["races"] else None,
            "race_hits":d["hits"],"race_hit_rate_pct":100*d["hits"]/d["races"] if d["races"] else None,
            "candidate_oracle_hits":oc["candidate_hits"],
            "template_retention_of_candidate_pct":100*d["hits"]/oc["candidate_hits"] if oc["candidate_hits"] else None,
            "stake_yen":d["stake"],"return_yen":d["ret"],"profit_yen":d["ret"]-d["stake"],
            "roi_pct":100*d["ret"]/d["stake"] if d["stake"] else None,
        })
    # cross-year stability table
    by=defaultdict(dict)
    for r in rows: by[(r["bet_type"],r["template"])][r["year"]]=r
    stable=[]
    for (bet,name),yr in sorted(by.items()):
        if not all(y in yr for y in YEARS): continue
        a24,a25=yr[2024],yr[2025]
        stake=a24["stake_yen"]+a25["stake_yen"]; ret=a24["return_yen"]+a25["return_yen"]
        stable.append({
            "bet_type":bet,"template":name,
            "roi_2024":a24["roi_pct"],"roi_2025":a25["roi_pct"],
            "hit_rate_2024":a24["race_hit_rate_pct"],"hit_rate_2025":a25["race_hit_rate_pct"],
            "avg_tickets_2024":a24["avg_tickets_per_race"],"avg_tickets_2025":a25["avg_tickets_per_race"],
            "retention_2024":a24["template_retention_of_candidate_pct"],"retention_2025":a25["template_retention_of_candidate_pct"],
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,
            "min_year_roi_pct":min(a24["roi_pct"],a25["roi_pct"]),
        })
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"template-year-metrics.csv",rows)
    write_csv(out/"template-stability.csv",stable)
    write_csv(out/"race-template-details.csv",details)
    summary={
        "contract":"L2_COMBO_TEMPLATE_LAB_RESULT_V1",
        "source_router_run":36511322402,
        "analysis_years":[2024,2025],
        "selected_exacta_races":sum(1 for x in selected if x["bet_type"]=="EXACTA"),
        "selected_trifecta_races":sum(1 for x in selected if x["bet_type"]=="TRIFECTA"),
        "2026_locked":True,
        "production_promotion":False,
        "note":"Exploratory template search on already-observed 2024-2025. Results identify structures for future walk-forward testing, not production winners."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text("# L2 Combo Template Lab V1\n\nExploratory EXACTA/TRIFECTA template search on DIRECT-selected 2024-2025 races. 2026 remains sealed.\n",encoding="utf-8")
    print("L2_COMBO_TEMPLATE_LAB_V1_READY",flush=True)

if __name__=="__main__": main()
