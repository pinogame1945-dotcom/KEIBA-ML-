#!/usr/bin/env python3
import argparse,csv,gzip,itertools,json,math
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import decode_odds,payout_map,horse_number_map
from build_l2_l17_fullfield_dataset_v1 import load_l17

BET_TYPES=("WIN","QUINELLA","EXACTA","TRIO")
BETA_GRID=(0.0,0.05,0.10,0.20,0.35,0.50,0.75,1.0)
EDGE_GRID=(0.05,0.15,0.30,0.50)
KELLY_MULT_GRID=(0.05,0.15,0.30)
RACE_CAP_GRID=(0.01,0.02)
DAY_CAP_GRID=(0.03,0.06)
HORSE_CAP_MULT_GRID=(0.50,1.00)
MAX_TICKETS_GRID=(4,10)
TYPE_DAY_CAP_SHARE=0.50
INITIAL_BANKROLL=100000.0
MIN_CAL_RACE_COVERAGE=0.20
MAX_CAL_DRAWDOWN_PCT=50.0
MIN_HALF_ROI_FOR_QUALIFY=95.0
CANDIDATE_PER_TYPE=20
EPS=1e-12
EXPECTED_RACES=3456

def finite(v):
    try:
        if isinstance(v,str): v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def keystr(vals):
    return "-".join(str(int(x)) for x in vals)

def parse_key(s):
    return tuple(int(x) for x in str(s).split("-") if str(x))

def norm(xs):
    s=sum(max(EPS,float(x)) for x in xs)
    return [max(EPS,float(x))/s for x in xs]

def blend(l1,win_odds,beta):
    q=norm([1.0/max(EPS,float(o)) for o in win_odds])
    p=norm(l1)
    if beta<=0:return q
    if beta>=1:return p
    vals=[(max(EPS,q[i])**(1.0-beta))*(max(EPS,p[i])**beta) for i in range(len(p))]
    return norm(vals)

def exacta_prob(p,i,j):
    if i==j:return 0.0
    d=1.0-p[i]
    return p[i]*(p[j]/d) if d>EPS else 0.0

def ticket_prob(bet,key,no_index,p):
    try:
        idx=[no_index[int(x)] for x in key]
    except KeyError:
        return 0.0
    if bet=="WIN":
        return p[idx[0]]
    if bet=="EXACTA":
        return exacta_prob(p,idx[0],idx[1])
    if bet=="QUINELLA":
        i,j=idx
        return exacta_prob(p,i,j)+exacta_prob(p,j,i)
    if bet=="TRIO":
        total=0.0
        for a,b,c in itertools.permutations(idx,3):
            d1=1.0-p[a]
            d2=1.0-p[a]-p[b]
            if d1>EPS and d2>EPS:
                total+=p[a]*(p[b]/d1)*(p[c]/d2)
        return total
    raise ValueError(bet)

def read_jsonl_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def load_odds_day(path,wanted):
    out={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line); rid=str(r.get("race_id") or "")
            if rid in wanted: out[rid]=r
    return out

def prepare_year(year,l17_path,backfill_root,out_path):
    l17=load_l17(l17_path,year)
    wanted=set(l17)
    root=Path(backfill_root)
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True)
    written=0; skipped=defaultdict(int)
    with gzip.open(out,"wt",encoding="utf-8") as fo:
        for dpath in sorted((root/"data"/"daily").glob(f"{year}-*.jsonl.gz")):
            date=dpath.name[:10]
            opath=root/"data"/"odds"/"daily"/dpath.name
            if not opath.exists(): continue
            day_rows={}
            with gzip.open(dpath,"rt",encoding="utf-8") as f:
                for line in f:
                    if not line.strip():continue
                    r=json.loads(line); rid=str((r.get("race") or {}).get("race_id") or "")
                    if rid in wanted: day_rows[rid]=r
            if not day_rows: continue
            odds_rows=load_odds_day(opath,set(day_rows))
            for rid,pack in day_rows.items():
                if rid not in odds_rows:
                    skipped["odds_row_missing"]+=1; continue
                rec=l17[rid]
                hn=horse_number_map(pack)
                hrows=rec.get("horses") or []
                p_by_no={}
                for h in hrows:
                    hid=str(h.get("horse_id") or "")
                    no=hn.get(hid)
                    p=finite(h.get("mean_probability"))
                    if no and p is not None and p>0:
                        p_by_no[int(no)]=p
                omap=decode_odds(odds_rows[rid])
                win={int(k[1][0]):float(v) for k,v in omap.items() if k[0]=="WIN"}
                eligible=sorted(set(p_by_no)&set(win))
                if len(eligible)<5:
                    skipped["too_few_win_priced"]+=1; continue
                l1=norm([p_by_no[n] for n in eligible])
                win_odds=[win[n] for n in eligible]
                es=set(eligible)
                tickets={b:[] for b in BET_TYPES}
                for (bet,key),odd in omap.items():
                    if bet not in BET_TYPES: continue
                    k=tuple(int(x) for x in key)
                    if all(x in es for x in k):
                        tickets[bet].append([keystr(k),float(odd)])
                payouts,present=payout_map(pack)
                pay={b:[] for b in BET_TYPES}
                for (bet,key),value in payouts.items():
                    if bet in BET_TYPES and all(int(x) in es for x in key):
                        pay[bet].append([keystr(key),float(value)])
                if not tickets["WIN"] or not pay["WIN"]:
                    skipped["win_contract_missing"]+=1; continue
                row={
                    "year":year,"race_id":rid,"race_date":date,
                    "horse_numbers":eligible,"l1_probability":l1,"win_odds":win_odds,
                    "tickets":tickets,"payouts":pay,
                }
                fo.write(json.dumps(row,separators=(",",":"))+"\n")
                written+=1
    if written<EXPECTED_RACES-10:
        raise SystemExit(f"prepared coverage too low y={year} races={written}")
    return {"year":year,"races":written,"skipped":dict(skipped)}

def select_betas(paths,years):
    stats={b:{beta:[0.0,0] for beta in BETA_GRID} for b in BET_TYPES}
    for y in years:
        for rec in read_jsonl_gz(paths[y]):
            no_index={int(n):i for i,n in enumerate(rec["horse_numbers"])}
            for beta in BETA_GRID:
                p=blend(rec["l1_probability"],rec["win_odds"],beta)
                for bet in BET_TYPES:
                    winners=rec["payouts"].get(bet) or []
                    if not winners: continue
                    mass=0.0
                    for ks,_ in winners:
                        mass+=ticket_prob(bet,parse_key(ks),no_index,p)
                    if mass>0:
                        stats[bet][beta][0]+=-math.log(max(EPS,min(1.0,mass)))
                        stats[bet][beta][1]+=1
    chosen={}; rows=[]
    for bet in BET_TYPES:
        vals=[]
        for beta in BETA_GRID:
            s,n=stats[bet][beta]
            nll=s/n if n else float("inf")
            rows.append({"bet_type":bet,"beta":beta,"nll":nll,"samples":n})
            vals.append((nll,beta))
        vals.sort()
        chosen[bet]=vals[0][1]
    return chosen,rows

def candidate_days(path,betas):
    days=defaultdict(lambda:defaultdict(list))
    source_races=0
    for rec in read_jsonl_gz(path):
        source_races+=1
        date=rec["race_date"]; rid=rec["race_id"]
        no_index={int(n):i for i,n in enumerate(rec["horse_numbers"])}
        strengths={b:blend(rec["l1_probability"],rec["win_odds"],betas[b]) for b in BET_TYPES}
        payout_lookup={b:{ks:float(v) for ks,v in rec["payouts"].get(b,[])} for b in BET_TYPES}
        merged=[]
        for bet in BET_TYPES:
            local=[]
            pvec=strengths[bet]
            for ks,odd0 in rec["tickets"].get(bet,[]):
                odd=float(odd0)
                if odd<=1.0: continue
                key=parse_key(ks)
                p=ticket_prob(bet,key,no_index,pvec)
                if p<=0: continue
                edge=p*odd-1.0
                if edge<=0: continue
                kelly=edge/(odd-1.0)
                if kelly<=0: continue
                local.append({
                    "race_id":rid,"race_date":date,"bet_type":bet,"key":ks,
                    "horses":key,"p":p,"odds":odd,"edge":edge,"kelly":kelly,
                    "return_per100":payout_lookup[bet].get(ks,0.0),
                })
            local.sort(key=lambda x:(x["kelly"],x["edge"],x["p"]),reverse=True)
            merged.extend(local[:CANDIDATE_PER_TYPE])
        merged.sort(key=lambda x:(x["kelly"],x["edge"],x["p"]),reverse=True)
        days[date][rid]=merged
    return days,source_races

def policy_grid():
    rows=[]
    for edge in EDGE_GRID:
        for km in KELLY_MULT_GRID:
            for rc in RACE_CAP_GRID:
                for dc in DAY_CAP_GRID:
                    for hm in HORSE_CAP_MULT_GRID:
                        for mt in MAX_TICKETS_GRID:
                            rows.append({
                                "edge_min":edge,"kelly_mult":km,"race_cap":rc,
                                "day_cap":dc,"horse_cap_mult":hm,"max_tickets":mt,
                            })
    return rows

def allocate_day(races,bank,policy):
    alloc=[]
    rc=float(policy["race_cap"]); hc=rc*float(policy["horse_cap_mult"])
    for rid,cands in races.items():
        rem=rc; hexp=defaultdict(float); used=0
        for c in cands:
            if c["edge"]<policy["edge_min"]: continue
            if used>=policy["max_tickets"] or rem<=1e-12: break
            frac=float(policy["kelly_mult"])*c["kelly"]
            if frac<=0: continue
            frac=min(frac,rem)
            for h in c["horses"]:
                frac=min(frac,hc-hexp[int(h)])
            if frac<=1e-12: continue
            x=dict(c); x["frac"]=frac; alloc.append(x)
            rem-=frac
            for h in c["horses"]: hexp[int(h)]+=frac
            used+=1
    if not alloc:return []
    type_cap=float(policy["day_cap"])*TYPE_DAY_CAP_SHARE
    totals=defaultdict(float)
    for a in alloc: totals[a["bet_type"]]+=a["frac"]
    scales={b:(min(1.0,type_cap/v) if v>0 else 1.0) for b,v in totals.items()}
    for a in alloc:a["frac"]*=scales[a["bet_type"]]
    total=sum(a["frac"] for a in alloc)
    dc=float(policy["day_cap"])
    if total>dc and total>0:
        s=dc/total
        for a in alloc:a["frac"]*=s
    final=[]
    for a in alloc:
        stake=math.floor((bank*a["frac"])/100.0)*100.0
        if stake<100: continue
        x=dict(a); x["stake"]=stake; final.append(x)
    return final

def simulate(days,policy,start_bank=INITIAL_BANKROLL,date_filter=None):
    bank=float(start_bank); peak=bank; min_bank=bank; maxdd=0.0
    stake=ret=0.0; tickets=hits=0; bet_days=0; races_bet=set()
    by_type={b:{"stake":0.0,"return":0.0,"tickets":0,"hits":0} for b in BET_TYPES}
    curve=[]
    all_dates=sorted(days)
    for date in all_dates:
        if date_filter is not None and date not in date_filter: continue
        day_start=bank
        al=allocate_day(days[date],day_start,policy)
        if al:
            bet_days+=1
            day_stake=sum(x["stake"] for x in al)
            day_ret=0.0
            for x in al:
                s=x["stake"]; r=(s/100.0)*x["return_per100"]
                day_ret+=r; stake+=s; ret+=r; tickets+=1; hits+=int(r>0)
                races_bet.add(x["race_id"])
                z=by_type[x["bet_type"]]; z["stake"]+=s; z["return"]+=r; z["tickets"]+=1; z["hits"]+=int(r>0)
            bank=bank-day_stake+day_ret
        peak=max(peak,bank); min_bank=min(min_bank,bank)
        if peak>0:maxdd=max(maxdd,100.0*(peak-bank)/peak)
        curve.append({"date":date,"bankroll":bank,"day_profit":bank-day_start})
        if bank<100: break
    total_races=sum(len(v) for d,v in days.items() if date_filter is None or d in date_filter)
    m={
        "start_bankroll":start_bank,"end_bankroll":bank,"bankroll_profit":bank-start_bank,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "tickets":tickets,"hits":hits,"bet_days":bet_days,
        "races_bet":len(races_bet),"source_races":total_races,
        "race_coverage_pct":100.0*len(races_bet)/total_races if total_races else 0.0,
        "max_drawdown_pct":maxdd,"min_bankroll":min_bank,
        "ruin":bank<0.20*start_bank,
    }
    return m,by_type,curve

def split_dates(days):
    ds=sorted(days)
    mid=len(ds)//2
    return set(ds[:mid]),set(ds[mid:])

def choose_policy(days):
    d1,d2=split_dates(days)
    rows=[]; qualified=[]
    for p in policy_grid():
        full,_,_=simulate(days,p)
        h1,_,_=simulate(days,p,date_filter=d1)
        h2,_,_=simulate(days,p,date_filter=d2)
        min_half=min(h1["roi_pct"] or 0.0,h2["roi_pct"] or 0.0)
        q=(
            full["race_coverage_pct"]>=100.0*MIN_CAL_RACE_COVERAGE and
            full["max_drawdown_pct"]<=MAX_CAL_DRAWDOWN_PCT and
            (h1["roi_pct"] or 0.0)>=MIN_HALF_ROI_FOR_QUALIFY and
            (h2["roi_pct"] or 0.0)>=MIN_HALF_ROI_FOR_QUALIFY and
            (full["roi_pct"] or 0.0)>=100.0
        )
        r={**p,
           "qualified":q,
           "cal_roi_pct":full["roi_pct"],"cal_profit_yen":full["profit_yen"],
           "cal_end_bankroll":full["end_bankroll"],"cal_max_drawdown_pct":full["max_drawdown_pct"],
           "cal_race_coverage_pct":full["race_coverage_pct"],
           "half1_roi_pct":h1["roi_pct"],"half2_roi_pct":h2["roi_pct"],
           "min_half_roi_pct":min_half}
        rows.append(r)
        if q:qualified.append(r)
    pool=qualified or rows
    pool.sort(key=lambda r:(
        r["min_half_roi_pct"],
        r["cal_roi_pct"] if r["cal_roi_pct"] is not None else -1e18,
        r["cal_end_bankroll"],
        -r["cal_max_drawdown_pct"],
        r["cal_race_coverage_pct"],
    ),reverse=True)
    return pool[0],rows,bool(qualified)

def flat_policy(days,edge_min=0.15,max_tickets=4,start_bank=INITIAL_BANKROLL):
    bank=float(start_bank); peak=bank; maxdd=0.0; stake=ret=0.0;tickets=hits=0;races=set();bet_days=0
    for date in sorted(days):
        chosen=[]
        for rid,cands in days[date].items():
            z=[c for c in cands if c["edge"]>=edge_min][:max_tickets]
            for c in z: chosen.append(c)
        if chosen: bet_days+=1
        for c in chosen:
            if bank<100:break
            s=100.0; r=c["return_per100"]
            bank=bank-s+r;stake+=s;ret+=r;tickets+=1;hits+=int(r>0);races.add(c["race_id"])
        peak=max(peak,bank)
        if peak>0:maxdd=max(maxdd,100.0*(peak-bank)/peak)
    total=sum(len(v) for v in days.values())
    return {
        "start_bankroll":start_bank,"end_bankroll":bank,"bankroll_profit":bank-start_bank,
        "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,"tickets":tickets,"hits":hits,
        "bet_days":bet_days,"races_bet":len(races),"source_races":total,
        "race_coverage_pct":100*len(races)/total if total else 0,
        "max_drawdown_pct":maxdd,"ruin":bank<0.2*start_bank,
    }

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8");return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    for y in (2021,2022,2023,2024,2025):
        p.add_argument(f"--l17-{y}",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--work-dir",required=True)
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    if hasattr(a,"l17_2026"):raise SystemExit("2026 sealed")
    work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
    compact={}
    prep=[]
    for y in (2021,2022,2023,2024,2025):
        path=work/f"y{y}.jsonl.gz";compact[y]=path
        prep.append(prepare_year(y,getattr(a,f"l17_{y}"),a.backfill_root,path))
        print("PREPARED",prep[-1],flush=True)

    folds=[]; beta_rows=[]; policy_rows=[]; bytype_rows=[]; continuous_curve=[]
    continuous_bank=INITIAL_BANKROLL
    fold_specs=[
        (2023,(2021,),2022),
        (2024,(2021,2022),2023),
        (2025,(2021,2022,2023),2024),
    ]
    for test,fit_years,cal_year in fold_specs:
        betas,br=select_betas(compact,fit_years)
        for r in br:
            r.update({"test_year":test,"fit_years":"|".join(map(str,fit_years))})
            beta_rows.append(r)
        cal_days,_=candidate_days(compact[cal_year],betas)
        selected,grid,qualified=choose_policy(cal_days)
        for r in grid:
            r.update({"test_year":test,"calibration_year":cal_year})
            policy_rows.append(r)
        test_days,_=candidate_days(compact[test],betas)
        annual,btype,curve=simulate(test_days,selected,INITIAL_BANKROLL)
        cont,btype_cont,curve_cont=simulate(test_days,selected,continuous_bank)
        continuous_bank=cont["end_bankroll"]
        flat=flat_policy(test_days,edge_min=selected["edge_min"],max_tickets=selected["max_tickets"],start_bank=INITIAL_BANKROLL)
        row={
            "test_year":test,"fit_years":"|".join(map(str,fit_years)),"calibration_year":cal_year,
            "selected_betas":json.dumps(betas,sort_keys=True,separators=(",",":")),
            "policy_qualified_on_calibration":qualified,
            **{f"policy_{k}":v for k,v in selected.items() if k in {"edge_min","kelly_mult","race_cap","day_cap","horse_cap_mult","max_tickets"}},
            **{f"test_{k}":v for k,v in annual.items()},
            "continuous_start_bankroll":cont["start_bankroll"],
            "continuous_end_bankroll":cont["end_bankroll"],
            "flat100_roi_pct":flat["roi_pct"],"flat100_profit_yen":flat["profit_yen"],
            "flat100_end_bankroll":flat["end_bankroll"],"flat100_max_drawdown_pct":flat["max_drawdown_pct"],
        }
        folds.append(row)
        for bet,z in btype.items():
            bytype_rows.append({
                "test_year":test,"bet_type":bet,**z,
                "roi_pct":100.0*z["return"]/z["stake"] if z["stake"] else None,
                "profit":z["return"]-z["stake"],
            })
        for x in curve_cont:
            continuous_curve.append({"test_year":test,**x})

    total_stake=sum(r["test_stake_yen"] for r in folds)
    total_ret=sum(r["test_return_yen"] for r in folds)
    pooled={
        "test_years":"2023|2024|2025",
        "stake_yen":total_stake,"return_yen":total_ret,"profit_yen":total_ret-total_stake,
        "roi_pct":100.0*total_ret/total_stake if total_stake else None,
        "continuous_start_bankroll":INITIAL_BANKROLL,
        "continuous_end_bankroll":continuous_bank,
        "continuous_profit":continuous_bank-INITIAL_BANKROLL,
        "winning_years":sum(1 for r in folds if (r["test_roi_pct"] or 0)>=100.0),
        "qualified_policy_years":sum(1 for r in folds if r["policy_qualified_on_calibration"]),
    }
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"folds.csv",folds)
    write_csv(out/"beta-grid.csv",beta_rows)
    write_csv(out/"policy-grid.csv",policy_rows)
    write_csv(out/"by-bet-type.csv",bytype_rows)
    write_csv(out/"continuous-bankroll.csv",continuous_curve)
    write_csv(out/"pooled.csv",[pooled])
    summary={
        "contract":"L3_MULTIBET_PORTFOLIO_V1_RESULT",
        "bet_types":list(BET_TYPES),
        "probability_model":"per-bet-type prior-only beta blend of normalized final WIN market probability and L1.7 mean win probability; exotic probabilities via Plackett-Luce/Harville",
        "allocation":"fractional independent Kelly with same-race, same-horse, daily and per-bet-type exposure caps; 100-yen units; bankroll updates at end of day",
        "policy_selection":"fit beta on years before calibration; choose active policy on immediately prior calibration year using two-half robustness; freeze into next test year",
        "folds":[{"test_year":x[0],"fit_years":list(x[1]),"calibration_year":x[2]} for x in fold_specs],
        "historical_market_timing":"FINAL_POSTHOC; not actionable same-time odds",
        "initial_bankroll_yen":INITIAL_BANKROLL,
        "pooled":pooled,
        "2026_locked":True,"promotion":False,"paid_compute":False,"artifact_cache":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== FOLDS =====");print((out/"folds.csv").read_text())
    print("===== POOLED =====");print((out/"pooled.csv").read_text())
    print("L3_MULTIBET_PORTFOLIO_V1_READY")

if __name__=="__main__":
    main()
