#!/usr/bin/env python3
"""L2 Expected Return V2: frozen EV horse, ticket probabilities × quoted price.

Uses race-day odds snapshots as retrospective FINAL prices, and exact official
payout. No 2026, no GPU, no unverified paid storage. Strict prior-year
calibration, 2024 policy choice, locked 2025 check. Neither year is a pristine
unseen test because prior studies inspected both. Results NOT live executable:
horse anchor itself also uses final odds.

No guessed missing remote files, no zero-filling missing quotes. PLACE and WIDE
may have quote price ranges: use conservative lower bound only, label separately.
Distinct race-wise PL probabilities treat final AI win strength and market win
strength as weights for sampled top3 finishing orders, explicitly hypothetical
rather than independently multiplying top3 probabilities.
"""
import argparse
import concurrent.futures
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

from run_all_race_flow_ticket_audit_v1 import YEARS, RACES_YEAR, TOTAL_RACES, get_inputs, assert_ok
from run_l2_value_anchor_bet_frontier_v1 import payouts
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map

BET_GROUP={"WIN":"1","PLACE":"2","QUINELLA":"4","WIDE":"5",
           "EXACTA":"6","TRIO":"7","TRIFECTA":"8"}
ODDS_TYPES=tuple(BET_GROUP)
UNORDERED={"QUINELLA","WIDE","TRIO"}
RANGE={"PLACE","WIDE"}
SUBGROUPS=ODDS_TYPES+("ALL_TYPES",)
METHODS=("LOWEST_PRICE","MARKET_PL","AI_PL","BLEND_PL","TICKET_CALIBRATED")
SKIP_PCT=(0,10,20)
BUDGETS=(1,2,3)
SCOPE="ONE_FROZEN_AI_VALUE_HORSE"
FEATURES=("z_market","z_ai","log_price","log_field","is_range")
EXPECTED={"WIN":1,"PLACE":1,"QUINELLA":2,"WIDE":2,"EXACTA":2,"TRIO":3,"TRIFECTA":3}

def priced_odds(race_odds,kind,numbers):
    group=(race_odds.get("odds") or {}).get(BET_GROUP[kind])
    if not isinstance(group,dict):return None,"MISSING_GROUP"
    key=(str(numbers[0]) if len(numbers)==1 else
         "".join(f"{int(n):02d}" for n in numbers))
    raw=group.get(key)
    if raw is None and len(numbers)==1:raw=group.get(f"{int(numbers[0]):02d}")
    if raw is None:return None,"NO_KEY"
    if not isinstance(raw,list) or len(raw)<3:return None,"BAD_RAW"
    # BACKFILL final triple is [3:6] for 6-value history, [:3] otherwise.
    tup=raw[3:6] if len(raw)>=6 else raw[:3]
    try:
        x=float(str(tup[0]).replace(",",""))
        if not math.isfinite(x) or x<=0:return None,"BAD_PRICE"
        if kind in RANGE:
            hi=float(str(tup[1]).replace(",",""))
            if not math.isfinite(hi) or hi<x:return None,"BAD_RANGE"
            # Conservative if public place/wide quote is a price interval.
            return x,"LOWER_BOUND_INTERVAL"
        return x,"FINAL_SINGLE_PRICE"
    except (TypeError,ValueError,IndexError):
        return None,"BAD_PRICE"

def pl_tables(prob):
    """Exact ordered P(1st i, 2nd j, 3rd k) in PL ranking model."""
    p=np.clip(np.asarray(prob,dtype=np.float64),1e-8,1.0)
    p=p/p.sum()
    n=len(p)
    check=np.arange(n)
    d1=np.maximum(1-p[:,None,None],1e-8)
    d2=np.maximum(1-p[:,None,None]-p[None,:,None],1e-8)
    tri=(p[:,None,None]*
         (p[None,:,None]/d1)*
         (p[None,None,:]/d2))
    valid=(check[:,None,None]!=check[None,:,None])&(
       check[:,None,None]!=check[None,None,:])&(
       check[None,:,None]!=check[None,None,:])
    tri*=valid
    # Normalize numerical truncation for finite valid cases (e.g. few runners).
    tri/=np.maximum(tri.sum(),1e-10)
    return p,tri

def event_prob(kind,idx,win,orders):
    if kind=="WIN":return float(win[idx[0]])
    if kind=="PLACE":
        i=idx[0]
        return float(orders[i,:,:].sum()+orders[:,i,:].sum()+orders[:,:,i].sum())
    if kind=="EXACTA":
        i,j=idx
        return float(orders[i,j,:].sum())
    if kind=="QUINELLA":
        i,j=idx
        return float(orders[i,j,:].sum()+orders[j,i,:].sum())
    if kind=="TRIFECTA":
        i,j,k=idx
        return float(orders[i,j,k])
    if kind=="TRIO":
        return float(sum(orders[i,j,k] for i,j,k in itertools.permutations(idx,3)))
    if kind=="WIDE":
        i,j=idx
        return float(
          orders[i,j,:].sum()+orders[j,i,:].sum()+
          orders[i,:,j].sum()+orders[j,:,i].sum()+
          orders[:,i,j].sum()+orders[:,j,i].sum()
        )
    raise KeyError(kind)

def family_tickets(axis,other_numbers):
    a=axis;others=other_numbers
    combos={kind:set() for kind in ODDS_TYPES}
    combos["WIN"].add((a,))
    combos["PLACE"].add((a,))
    for b in others:
        pair=tuple(sorted((a,b)))
        combos["QUINELLA"].add(pair);combos["WIDE"].add(pair)
        combos["EXACTA"].add((a,b));combos["EXACTA"].add((b,a))
    for b,c in itertools.combinations(others,2):
        tri=tuple(sorted((a,b,c)))
        combos["TRIO"].add(tri)
        for three in itertools.permutations((a,b,c),3):
            combos["TRIFECTA"].add(three)
    return combos

def one_year(year,raw,picks,root):
    subset=raw[raw.year.eq(year)].copy()
    subset["_race_date"]=subset["_race_date"].astype(str).str[:10]
    bydate=subset[["_race_id","_race_date"]].drop_duplicates().groupby("_race_date")
    groups=subset.groupby("_race_id",sort=False)
    rows=[];missing=defaultdict(int);sample_raw=defaultdict(dict)
    n_races=0
    for day,idsdf in bydate:
        ids=set(idsdf["_race_id"].astype(str))
        # One read of each exact existing remote daily file, reused all families.
        pack=load_day(root/"data"/"daily"/f"{day}.jsonl.gz",ids)
        odds=load_odds_day(root/"data"/"odds"/"daily"/f"{day}.jsonl.gz",ids)
        for rid in sorted(ids):
            n_races+=1
            z=pack.get(rid);o=odds.get(rid)
            assert_ok(z is not None and o is not None,f"missing race or odds record {rid}")
            group=groups.get_group(rid)
            horse_num=horse_number_map(z)
            axis=picks["ANCHOR_FULL"][rid]
            assert_ok(axis in horse_num,f"{rid}: frozen anchor horse absent")
            g=group.sort_values(["market_rank","_horse_id"]).copy()
            horse_ids=g["_horse_id"].astype(str).tolist()
            numbers=[horse_num[x] for x in horse_ids]
            assert_ok(len(numbers)==len(set(numbers)),f"{rid}: horse_number duplicates")
            index={n:i for i,n in enumerate(numbers)}
            market_rank=g.sort_values(["market_rank","final_win_odds","_horse_id"],
                             ascending=[True,True,True])["_horse_id"].astype(str).tolist()
            ai_rank=g.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                             ascending=[False,True,True])["_horse_id"].astype(str).tolist()
            other=set(market_rank[:5]+ai_rank[:5]);other.discard(axis)
            # Bound computation: 10 horse candidate union max, includes all predeclared.
            other_nums=sorted(horse_num[x] for x in other)
            axisno=horse_num[axis]
            tickets=family_tickets(axisno,other_nums)
            p0=g["market_prob_norm"].to_numpy(float)
            p1=g["P__ANCHOR_FULL"].to_numpy(float)
            assert_ok(np.isfinite(p0).all() and np.isfinite(p1).all() and
                  (p0>0).all() and (p1>0).all(),f"{rid}: invalid win probabilities")
            w0,t0=pl_tables(p0);w1,t1=pl_tables(p1)
            w2,t2=pl_tables(np.sqrt(w0*w1))
            official=payouts(z,rid)
            sample=(o.get("odds") or {})
            for kind,grpname in BET_GROUP.items():
                v=sample.get(grpname)
                if isinstance(v,dict) and v and not sample_raw[year].get(kind):
                    key=next(iter(v));sample_raw[year][kind]={
                        "sample_key":key,"sample_raw":str(v[key])[:110],
                        "total_odds_entries_example":len(v)}
            for kind,choices in tickets.items():
                for ticket in sorted(choices):
                    price,quality=priced_odds(o,kind,ticket)
                    if price is None:
                        missing[(year,kind,quality)]+=1
                        continue
                    if not (0<price<1e7):
                        missing[(year,kind,"UNPLAUSIBLE_PRICE")]+=1
                        continue
                    idx=tuple(index[x] for x in ticket)
                    pm=event_prob(kind,idx,w0,t0)
                    pa=event_prob(kind,idx,w1,t1)
                    pb=event_prob(kind,idx,w2,t2)
                    assert_ok(-1e-6<pm<=1.000001 and -1e-6<pa<=1.000001,
                         f"{rid}: invalid PL probability")
                    pay=official.get((kind,ticket),0)
                    # Final PLACE/WIDE quotes are conservative range bounds only.
                    rows.append((year,rid,day,kind,*ticket[:1],
                         ",".join(map(str,ticket)),len(numbers),price,quality,
                         max(0,pm),max(0,pa),max(0,pb),float(pay),int(pay>0)))
    assert_ok(n_races==RACES_YEAR,f"{year} races {n_races} != {RACES_YEAR}")
    print('QUOTE_AUDIT_YEAR',year,'covered_races',len(set(r[1] for r in rows)),
          'quoted_tickets',len(rows),
          'missing_by_kind',dict((str(k),v) for k,v in missing.items()),
          'raw_examples',sample_raw.get(year,{}),flush=True)
    return rows,missing,sample_raw

def to_dataframe(parts):
    columns=["year","race_id","race_date","bet_type","selected_axis_number",
             "ticket","field_size","quote_odds","quote_quality",
             "p_market","p_ai","p_blend","payout_yen","hit"]
    # selected_axis_number is the first number of TICKET for any ordered
    # combo, and MUST NOT be used as a feature. For reverse exacta it differs
    # from actual selected axis. Already only anchor-containing candidates.
    allr=pd.DataFrame([x for segment in parts for x in segment[0]],columns=columns)
    assert_ok(not allr.empty,"no odds ticket rows")
    assert_ok(allr.year.isin(YEARS).all(),"2026 leakage")
    assert_ok(allr.duplicated(["race_id","bet_type","ticket"]).sum()==0,
        "ticket duplicate")
    present=allr.groupby("year").race_id.nunique().to_dict()
    print("V2_QUOTES_YEAR_COVERAGE",json.dumps(present),flush=True)
    # Missing quotes are not imputed. We keep full 3456-year denominators
    # and expose unavailable races in every policy result.
    assert_ok(set(present)==set(YEARS) and min(present.values())>=RACES_YEAR*.70,
       f"insufficient odds coverage to evaluate fairly: {present}")
    return allr

def logit(x):
    a=np.clip(x,1e-7,1-1e-7)
    return np.log(a/(1-a))
def features(data):
    return np.column_stack([
       logit(data.p_market.to_numpy(float)),
       logit(data.p_ai.to_numpy(float)),
       np.log(data.quote_odds.to_numpy(float)),
       np.log(data.field_size.to_numpy(float)),
       (data.quote_quality=="LOWER_BOUND_INTERVAL").to_numpy(float),
    ]).astype(np.float64)

def chronological_calibrate(df):
    results=[]
    issues=[]
    for yr in (2024,2025):
        train=df[df.year.lt(yr)]
        test=df[df.year.eq(yr)]
        for kind in ODDS_TYPES:
            t=train[train.bet_type.eq(kind)]
            e=test[test.bet_type.eq(kind)]
            if len(t)<100 or len(e)<100 or t.hit.sum()<10:
                issues.append(f"{yr}:{kind}: insufficient samples")
                continue
            mod=LogisticRegression(C=.15,max_iter=160,solver="lbfgs")
            mod.fit(features(t),t.hit.astype(int).to_numpy())
            q=mod.predict_proba(features(e))[:,1]
            r=e.copy()
            r["p_calibrated"]=np.clip(q,1e-7,1-1e-7)
            results.append(r)
    assert_ok(not issues,"missing calibration "+str(issues))
    return pd.concat(results,ignore_index=True)

def model_qualities(x):
    out=[]
    for (yr,kind),r in x.groupby(["year","bet_type"]):
        for name,col in [("MARKET_PL","p_market"),("AI_PL","p_ai"),
                         ("BLEND_PL","p_blend"),("TICKET_CALIBRATED","p_calibrated")]:
            p=np.clip(r[col].to_numpy(float),1e-7,1-1e-7)
            y=r.hit.to_numpy(int)
            # Ticket events within a race are dependent; scores are
            # descriptive, not iid confidence intervals.
            out.append(dict(year=int(yr),kind=kind,model=name,
             quotes=len(r),hits=int(y.sum()),brier=round(float(np.mean((y-p)**2)),9),
             binary_logloss=round(float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),9),
             sum_predicted_hits=round(float(p.sum()),3),
             realized_hits=int(y.sum()),
             predicted_to_real_hit_ratio=round(float(p.sum()/max(1,y.sum())),4)))
    return pd.DataFrame(out)

def select_tickets(group,scope,method,budget):
    if scope!="ALL_TYPES":group=group[group.bet_type.eq(scope)]
    if len(group)==0:return None,0
    g=group.copy()
    if method=="LOWEST_PRICE":
        g["ev"]=-g.quote_odds
    elif method=="MARKET_PL":g["ev"]=g.p_market*g.quote_odds
    elif method=="AI_PL":g["ev"]=g.p_ai*g.quote_odds
    elif method=="BLEND_PL":g["ev"]=g.p_blend*g.quote_odds
    else:g["ev"]=g.p_calibrated*g.quote_odds
    ranked=g.sort_values(["ev","bet_type","ticket"],ascending=[False,True,True])
    chosen=ranked.head(budget)
    return chosen,float(ranked.ev.iloc[0])

def strategies(test):
    """Per-race NumPy score sort ONCE per family/method, reuse all budgets."""
    rows=[]
    for rid,g in test.groupby("race_id",sort=False):
        date=g.race_date.iloc[0];yr=int(g.year.iloc[0])
        kinds=g.bet_type.to_numpy(str)
        ticket_ids=g.ticket.to_numpy(str)
        payout=g.payout_yen.to_numpy(float)
        price=g.quote_odds.to_numpy(float)
        pm=g.p_market.to_numpy(float)
        pa=g.p_ai.to_numpy(float)
        pb=g.p_blend.to_numpy(float)
        pc=g.p_calibrated.to_numpy(float)
        scores={"LOWEST_PRICE":-price,
                "MARKET_PL":pm*price,
                "AI_PL":pa*price,
                "BLEND_PL":pb*price,
                "TICKET_CALIBRATED":pc*price}
        groups={kind:np.where(kinds==kind)[0] for kind in ODDS_TYPES}
        groups["ALL_TYPES"]=np.arange(len(g))
        for scope,arr in groups.items():
            if len(arr)==0:continue
            for method,score in scores.items():
                # stable deterministic tie break by physical bet group + ticket
                order=np.lexsort((ticket_ids[arr],kinds[arr],-score[arr]))
                idx=arr[order]
                available=min(len(idx),max(BUDGETS))
                paid=np.cumsum(payout[idx[:available]])
                for b in BUDGETS:
                    take=min(b,len(idx))
                    ids=idx[:take]
                    rows.append((yr,rid,date,scope,method,b,take,
                         float(paid[take-1]),float(100*take),float(score[idx[0]]),
                         int(paid[take-1]>0),
                         "|".join(kinds[ids]+":"+ticket_ids[ids])))
    return pd.DataFrame(rows,columns=["year","race_id","race_date","scope","method",
               "budget","tickets","payout_yen","stake_yen","top_ticket_score",
               "race_hit","selected_ticket_ids"])

def make_results(train,tgt,yr):
    # Thresholds trained on historical period scores; no test-year labels used.
    # train is 2024/2025 prediction dataframe only, used for cutoff at 2025.
    # In 2024, default no cutoffs when no model OOS predictions from 2023,
    # use target-score threshold within prior 2023 uncalibrated PL simulation.
    rows=[]
    for (scope,method,b),te in tgt.groupby(["scope","method","budget"],sort=False):
        te=te.sort_values("race_id")
        train_grp=train[(train.scope.eq(scope))&(train.method.eq(method))&
                  (train.budget.eq(b))]
        # Crucial: training-score distribution comes from strictly earlier
        # race ids. If no earlier score, no selection, never use test data
        # to set thresholds.
        if len(train_grp)<200:continue
        for skip in SKIP_PCT:
            if skip:
                threshold=float(train_grp.top_ticket_score.quantile(skip/100))
                selected=te[te.top_ticket_score.gt(threshold)]
            else:selected=te
            paid=float(selected.payout_yen.sum());cost=float(selected.stake_yen.sum())
            rows.append(dict(
                year=yr,scope=scope,method=method,budget=int(b),
                target_skip_pct=skip,
                all_races=RACES_YEAR,
                available_races=len(te),
                bet_races=len(selected),
                bet_coverage_pct=100*len(selected)/RACES_YEAR,
                avg_tickets_bought=float(selected.tickets.mean()) if len(selected) else 0,
                stake_yen=cost,payout_yen=paid,
                profit_yen=paid-cost,roi_pct=100*paid/cost if cost else np.nan,
                hit_races=int(selected.race_hit.sum()),
                foregone_payout_yen=float(te.payout_yen.sum()-paid),
                missed_races_without_quote=RACES_YEAR-len(te),
                quote_warning="PLACE/WIDE interval lower quote is not executable payout price"
                   if scope in ("PLACE","WIDE","ALL_TYPES") else "",
            ))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--picks",required=True)
    ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    pred,picks=get_inputs(args.predictions,args.picks)
    extra=pd.read_csv(args.predictions,usecols=["year","_race_id","_horse_id",
                           "_race_date","market_prob_norm"],
        compression="infer",dtype={"_race_id":str,"_horse_id":str},low_memory=False)
    pred=pred.merge(extra,on=["year","_race_id","_horse_id"],validate="one_to_one")
    assert_ok(pred._race_id.nunique()==TOTAL_RACES,"pred count drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        parts=list(pool.map(lambda y:one_year(y,pred,picks,Path(args.backfill_root)),YEARS))
    ledger=to_dataframe(parts)
    # One concrete stored record per source, not huge redundant full ticket panel.
    missing=defaultdict(int)
    examples={}
    for _,m,s in parts:
        for k,n in m.items():missing[k]+=n
        examples.update(s)
    # Full quote sample coverage audit. Do not interpret availability from outcomes.
    qa=[]
    for (yr,kind),rows in ledger.groupby(["year","bet_type"]):
        qa.append(dict(year=yr,bet_type=kind,quoted_candidates=len(rows),
          races_with_quotes=rows.race_id.nunique(),race_coverage_pct=round(
                100*rows.race_id.nunique()/RACES_YEAR,4),
          min_quote=float(rows.quote_odds.min()),max_quote=float(rows.quote_odds.max()),
          quote_quality="LOWER_BOUND_INTERVAL" if kind in RANGE else "FINAL_SINGLE_PRICE",
          winning_tickets_observed=int(rows.hit.sum())))
    qa=pd.DataFrame(qa)
    qa.to_csv(out/"quote-availability-and-quality.csv",index=False)
    with (out/"quote-missing-audit.json").open("w") as f:
        json.dump({"year_kind_status_counts":[dict(year=y,bet_type=b,status=q,count=v)
                    for (y,b,q),v in sorted(missing.items())],
                  "raw_odds_group_examples":examples},f,indent=2,ensure_ascii=False)
    evalrows=chronological_calibrate(ledger)
    quality=model_qualities(evalrows)
    quality.to_csv(out/"joint-ticket-probability-calibration.csv",index=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decision_parts=list(pool.map(
            lambda y:strategies(evalrows[evalrows.year.eq(y)]),(2024,2025)))
    decisions=pd.concat(decision_parts,ignore_index=True)
    assert_ok(not decisions.empty and decisions.year.nunique()==2,"decisions empty")
    # In lieu of in-sample model calibration for 2023, reconstruct uncalibrated
    # 2023 ticket panels with a fixed no-outcome ticket score; this is only
    # for determining 2024 purchase threshold, NOT ticket model training.
    base23=ledger[ledger.year.eq(2023)].copy()
    base23["p_calibrated"]=base23["p_blend"]
    history23=strategies(base23)
    history24=decisions[decisions.year.eq(2024)]
    results=pd.concat([make_results(history23,history24,2024),
                       make_results(history24,decisions[decisions.year.eq(2025)],2025)],
                      ignore_index=True)
    assert_ok(not results.empty,"no policies")
    results.to_csv(out/"ticket-price-value-frontier.csv",index=False)
    results["quoted_full_field"]=results.missed_races_without_quote.eq(0)
    # Lock any model and policy choice using 2024, without observing 2025.
    # 2024 in previous experiments has been examined; exploratory policy choice.
    selections=[]
    for mandate,coverage in [("BUY_90",90),("BUY_80",80),("BUY_70",70)]:
        pool=results[(results.year.eq(2024))&
            (results.bet_coverage_pct.ge(coverage))&
            results.quote_warning.eq("")].copy()
        assert_ok(len(pool)>0,f"no strategies in 2024 coverage {mandate}")
        s=pool.sort_values(["profit_yen","roi_pct","bet_races"],
                            ascending=[False,False,False]).iloc[0]
        next_=results[(results.year.eq(2025)) &
              results.scope.eq(s.scope) & results.method.eq(s.method)&
              results.budget.eq(s.budget)&
              results.target_skip_pct.eq(s.target_skip_pct)]
        assert_ok(len(next_)==1,f"missing next-year policy {mandate}")
        z=next_.iloc[0]
        selections.append({
          "mandate":mandate,
          "selected":{name:s[name] for name in
             ("scope","method","budget","target_skip_pct")},
          "2024":{name:s[name] for name in
             ("bet_races","bet_coverage_pct","stake_yen","payout_yen","profit_yen","roi_pct")},
          "locked_2025":{name:z[name] for name in
             ("bet_races","bet_coverage_pct","stake_yen","payout_yen","profit_yen","roi_pct")},
        })
    # Price-odds consistency for physically settled winning tickets only.
    ppaid=ledger[ledger.hit.eq(1)].copy()
    ppaid["price_payout_ratio"]=ppaid.payout_yen/(100*ppaid.quote_odds)
    price_audit=ppaid.groupby(["year","bet_type"]).agg(
        winning_tickets=("hit","size"),
        median_actual_to_quote=("price_payout_ratio","median"),
        pct_far_from_quote=("price_payout_ratio",lambda x:
                       float(100*np.mean((x<.75)|(x>1.30))))
    ).reset_index()
    price_audit.to_csv(out/"official-payout-versus-final-quote.csv",index=False)
    # Avoid pretending external quote ranges are guaranteed; they are lower bound.
    summary={
      "contract":"L2_EXPECTED_RETURN_V2_ALL_TICKET_PRICES",
      "cohort_years":list(YEARS),
      "model_eval_years":[2024,2025],
      "total_races":TOTAL_RACES,
      "selection_axis":"Frozen ANCHOR_FULL, based on final win odds",
      "bet_types":list(ODDS_TYPES),
      "models":list(METHODS),
      "ticket_budgets":list(BUDGETS),
      "skip_percent_targets":list(SKIP_PCT),
      "joint_probs":"Full ordered top-3 Plackett-Luce from market/AI win probabilities, plus prior-year binary calibration.",
      "price_time":"FINAL ODDS: research retrospective only, not pre-off available.",
      "place_wide_warning":"PLACE and WIDE use conservative LOWER bounds of odds intervals; not validated as executable prices, and excluded from promotion.",
      "price_missing_rule":"Missing/unreadable exact quote => no evaluated ticket, never imputed.",
      "outcome_source":"Actual official payouts, never final quote as a payout substitute.",
      "previous_year_calibration":"2024 train 2023, 2025 train 2023+2024. Thresholds from prior-year score distribution.",
      "selection_rule":"Select with 2024 evidence only, lock then report 2025. Both previously inspected => exploratory.",
      "best_2024_policies_locked_to_2025":selections,
      "price_availability":qa.to_dict("records"),
      "models_deployable":False,
      "2026_sealed":True,
      "free_standard_cpu_only":True,
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str)+"\n")
    print("L2_EXPECTED_RETURN_V2_DONE")
    print("AUDIT")
    print(qa.to_csv(index=False))
    print("CHOSEN_AND_LOCKED")
    print(json.dumps(selections,ensure_ascii=False,default=str))
    print("MARKET_V_AI")
    baseline=results[(results.scope.eq("TRIO"))&(results.budget.eq(1))&
           results.target_skip_pct.eq(0)]
    print(baseline.to_csv(index=False))
if __name__=="__main__":
    main()
