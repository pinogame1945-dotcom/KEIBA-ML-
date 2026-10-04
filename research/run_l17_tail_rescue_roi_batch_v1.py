#!/usr/bin/env python3
import argparse,itertools,json,math,time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_l2_trio_probability_v2 import (
    load_horse_dataset,attach_win_market,attach_outsider,prepare_l175,trio_pl_probs
)
from run_l17_tail_rescue_audit_v1 import prep,fit_tail_model,score_tail,race_frames
from build_l2_bet_kings_dataset_v1 import load_day,payout_map

SEED=20261004
STAKE=100.0
FOLDS=((2023,(2022,)),(2024,(2022,2023)),(2025,(2022,2023,2024)))
METHODS=("KING_ORDER","MARKET","OUTSIDER","L175_P3","ML_COMBINED")
RESCUE_K=(1,2,3)
TOPKS=(12,20,30)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--contract",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--outsider-predictions",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def ticket_scores(g):
    # All unordered TRIO tickets for the original field, scored without TRIO odds.
    g=g.sort_values(["consensus_rank","horse_number","horse_id"]).reset_index(drop=True)
    n=len(g)
    comb=np.asarray(list(itertools.combinations(range(n),3)),dtype=np.int16)
    probs=np.clip(g["king_probability_mean"].to_numpy(dtype=float),0.0,None)
    probs=probs/probs.sum() if probs.sum()>0 else np.full(n,1.0/n)
    king=trio_pl_probs(probs,comb)
    p3=np.clip(g["p3_calibrated"].to_numpy(dtype=float),1e-9,1.0)
    p3score=np.log(p3[comb]).sum(axis=1)
    nos=g["horse_number"].astype(int).to_numpy()
    out={}
    for row,ks,ps in zip(comb,king,p3score):
        nums=tuple(sorted(int(nos[i]) for i in row))
        out[nums]=(float(ks),float(ps))
    return out

def metrics(rows):
    if not rows:
        return {}
    z=pd.DataFrame(rows)
    races=int(z["race_id"].nunique())
    tickets=int(z["tickets"].sum())
    stake=float(z["stake"].sum())
    ret=float(z["return_yen"].sum())
    hits=int(z["hit"].sum())
    avg_payout=float(z.loc[z["hit"]==1,"return_yen"].mean()) if hits else None
    avg_tickets=float(z["tickets"].mean())
    be=(100.0*(avg_tickets*STAKE)/avg_payout) if avg_payout and avg_payout>0 else None
    roi100=(100.0*avg_payout/(avg_tickets*STAKE)) if avg_payout and avg_tickets>0 else None
    return {
        "races":races,
        "tickets":tickets,
        "avg_tickets_per_race":avg_tickets,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100.0*ret/stake if stake else None,
        "hit_races":hits,
        "hit_rate_pct":100.0*hits/races if races else None,
        "avg_payout_when_hit_yen":avg_payout,
        "break_even_hit_rate_pct_at_observed_hit_payout":be,
        "roi_if_100pct_hit_at_observed_avg_payout":roi100,
    }

def add_strategy(acc,key,rid,tickets,payouts):
    tickets=list(dict.fromkeys(tuple(sorted(x)) for x in tickets))
    ret=sum(float(payouts.get(("TRIO",t),0.0)) for t in tickets)
    hit=int(ret>0)
    acc[key].append({
        "race_id":rid,
        "tickets":len(tickets),
        "stake":STAKE*len(tickets),
        "return_yen":ret,
        "hit":hit,
    })

def choose_topk(pool_tickets,scoremap,index,k):
    q=sorted(pool_tickets,key=lambda t:(-scoremap[t][index],t))
    return q[:min(k,len(q))]

def process_year(df,year,model,root):
    root=Path(root)
    races,excluded=race_frames(df,year)
    bydate=defaultdict(list)
    for g in races:
        bydate[str(g["race_date"].iloc[0])[:10]].append(g)

    acc=defaultdict(list)
    horse_capture=defaultdict(lambda:{"races":0,"capture":0,"selected_horses":0})
    agreement_rows=[]
    signal_rows=[]
    missing=defaultdict(int)

    for date,gl in sorted(bydate.items()):
        wanted={str(g["race_id"].iloc[0]) for g in gl}
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for g in gl:
            rid=str(g["race_id"].iloc[0])
            pack=day.get(rid)
            if pack is None:
                missing["missing_pack"]+=1; continue
            payouts,present=payout_map(pack)
            if "TRIO" not in present:
                missing["no_trio_payout"]+=1; continue

            g=g.sort_values(["consensus_rank","horse_number","horse_id"]).reset_index(drop=True)
            winners=set(g.loc[g["target_top3"]==1,"horse_number"].astype(int))
            base=set(g.loc[g["consensus_rank"]<=12,"horse_number"].astype(int))
            axis_no=int(g.sort_values(["consensus_rank","horse_number"]).iloc[0]["horse_number"])
            scoremap=ticket_scores(g)

            tail=g[g["consensus_rank"]>=13].copy()
            signal_rows.append({
                "year":year,"race_id":rid,
                "tail_horses":len(tail),
                "outsider_available_horses":int(tail["outsider_available"].sum()),
                "p3_positive_horses":int((tail["p3_calibrated"]>0).sum()),
                "outsider_nonzero_horses":int((tail["outsider_score_scaled"]!=0).sum()),
                "market_unique_values":int(tail["market_rank"].nunique()),
                "p3_unique_values":int(tail["p3_calibrated"].nunique()),
                "outsider_unique_values":int(tail["outsider_score_scaled"].nunique()),
            })

            ordered={}
            for method in METHODS:
                ordered[method]=score_tail(g,method,model)["horse_number"].astype(int).tolist()

            # Pairwise selector agreement on top1/top2/top3 sets.
            for i,m1 in enumerate(METHODS):
                for m2 in METHODS[i+1:]:
                    for k in RESCUE_K:
                        s1=set(ordered[m1][:k]); s2=set(ordered[m2][:k])
                        agreement_rows.append({
                            "year":year,"race_id":rid,"method_a":m1,"method_b":m2,"k":k,
                            "same_set":int(s1==s2)
                        })

            scenarios=[("BASE",0,base)]
            for method in METHODS:
                for k in RESCUE_K:
                    selected=base|set(ordered[method][:k])
                    scenarios.append((method,k,selected))

            for method,k,selected in scenarios:
                keybase=(year,method,k)
                h=horse_capture[keybase]
                h["races"]+=1; h["capture"]+=int(winners.issubset(selected)); h["selected_horses"]+=len(selected)

                pool_tickets=[t for t in scoremap if set(t).issubset(selected)]
                add_strategy(acc,(year,method,k,"BOX_ALL",0),rid,pool_tickets,payouts)

                axis_tickets=[t for t in pool_tickets if axis_no in t]
                add_strategy(acc,(year,method,k,"KING1_AXIS_ALL",0),rid,axis_tickets,payouts)

                for topk in TOPKS:
                    add_strategy(acc,(year,method,k,"KING_PL_TOPK",topk),rid,
                                 choose_topk(pool_tickets,scoremap,0,topk),payouts)
                    add_strategy(acc,(year,method,k,"L175_P3_TOPK",topk),rid,
                                 choose_topk(pool_tickets,scoremap,1,topk),payouts)

    result_rows=[]
    for (y,method,k,fam,topk),rows in acc.items():
        m=metrics(rows)
        h=horse_capture[(y,method,k)]
        result_rows.append({
            "year":y,"selector":method,"rescue_k":k,
            "bet_family":fam,"ticket_topk":topk if topk else "",
            "horse_pool_capture_pct":100*h["capture"]/h["races"] if h["races"] else None,
            "avg_selected_horses":h["selected_horses"]/h["races"] if h["races"] else None,
            **m
        })
    return result_rows,agreement_rows,signal_rows,excluded,dict(missing)

def pooled_metrics(result_rows):
    z=pd.DataFrame(result_rows)
    pooled=[]
    groupcols=["selector","rescue_k","bet_family","ticket_topk"]
    for keys,g in z.groupby(groupcols,dropna=False):
        selector,k,fam,topk=keys
        races=int(g["races"].sum()); tickets=int(g["tickets"].sum())
        stake=float(g["stake_yen"].sum()); ret=float(g["return_yen"].sum())
        hits=int(g["hit_races"].sum())
        # Reconstruct weighted observed payout among hit races.
        total_hit_return=sum(float(r["avg_payout_when_hit_yen"])*int(r["hit_races"]) for _,r in g.iterrows() if pd.notna(r["avg_payout_when_hit_yen"]))
        avg_payout=total_hit_return/hits if hits else None
        avg_tickets=tickets/races if races else None
        be=100*(avg_tickets*STAKE)/avg_payout if avg_payout else None
        roi100=100*avg_payout/(avg_tickets*STAKE) if avg_payout and avg_tickets else None
        capture=sum(float(r["horse_pool_capture_pct"])*int(r["races"]) for _,r in g.iterrows())/races if races else None
        selected=sum(float(r["avg_selected_horses"])*int(r["races"]) for _,r in g.iterrows())/races if races else None
        pooled.append({
            "year":"POOLED","selector":selector,"rescue_k":int(k),"bet_family":fam,
            "ticket_topk":topk,
            "horse_pool_capture_pct":capture,
            "avg_selected_horses":selected,
            "races":races,"tickets":tickets,"avg_tickets_per_race":avg_tickets,
            "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
            "roi_pct":100*ret/stake if stake else None,
            "hit_races":hits,"hit_rate_pct":100*hits/races if races else None,
            "avg_payout_when_hit_yen":avg_payout,
            "break_even_hit_rate_pct_at_observed_hit_payout":be,
            "roi_if_100pct_hit_at_observed_avg_payout":roi100,
        })
    return pooled

def summarize_agreement(rows):
    z=pd.DataFrame(rows)
    out=[]
    for keys,g in z.groupby(["method_a","method_b","k"]):
        a,b,k=keys
        out.append({"method_a":a,"method_b":b,"k":int(k),"races":len(g),"same_set_pct":100*float(g["same_set"].mean())})
    return out

def summarize_signal(rows):
    z=pd.DataFrame(rows)
    return {
        "tail_races":int(len(z)),
        "tail_horses":int(z["tail_horses"].sum()),
        "outsider_available_horses":int(z["outsider_available_horses"].sum()),
        "outsider_available_pct":100*float(z["outsider_available_horses"].sum())/float(z["tail_horses"].sum()),
        "p3_positive_horses":int(z["p3_positive_horses"].sum()),
        "p3_positive_pct":100*float(z["p3_positive_horses"].sum())/float(z["tail_horses"].sum()),
        "outsider_nonzero_horses":int(z["outsider_nonzero_horses"].sum()),
        "outsider_nonzero_pct":100*float(z["outsider_nonzero_horses"].sum())/float(z["tail_horses"].sum()),
        "mean_market_unique_values":float(z["market_unique_values"].mean()),
        "mean_p3_unique_values":float(z["p3_unique_values"].mean()),
        "mean_outsider_unique_values":float(z["outsider_unique_values"].mean()),
    }

def main():
    t0=time.time(); a=parse_args()
    c=json.load(open(a.contract,encoding="utf-8"))
    assert c["contract"]=="L17_TAIL_RESCUE_ROI_BATCH_V1"
    assert c["cost_policy"]["github_standard_cpu_only"] is True
    assert c["cost_policy"]["gpu"] is False
    assert c["data_policy"]["2026_locked"] is True

    _,df=load_horse_dataset(a.dataset_dir)
    df,skipped_market=attach_win_market(df,a.backfill_root)
    df=attach_outsider(df,a.outsider_predictions)
    df=prep(prepare_l175(df))
    if 2026 in set(df["year"]): raise SystemExit("2026 sealed")

    all_results=[]; all_agreement=[]; all_signal=[]; exclusions=[]; missing=[]
    for test_year,train_years in FOLDS:
        model,nrows,pos,spw=fit_tail_model(df,train_years)
        rr,aa,ss,ex,mi=process_year(df,test_year,model,a.backfill_root)
        all_results.extend(rr); all_agreement.extend(aa); all_signal.extend(ss)
        exclusions.append({"year":test_year,**{f"exclude_{k}":v for k,v in ex.items()}})
        missing.append({"year":test_year,**{f"missing_{k}":v for k,v in mi.items()}})
        print("TAIL_ROI_YEAR_DONE "+json.dumps({"year":test_year,"rows":len(rr)},separators=(",",":")),flush=True)

    pooled=pooled_metrics(all_results)
    agreement=summarize_agreement(all_agreement)
    signal=summarize_signal(all_signal)

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_results).to_csv(out/"year-strategy-metrics.csv",index=False)
    pd.DataFrame(pooled).to_csv(out/"pooled-strategy-metrics.csv",index=False)
    pd.DataFrame(agreement).to_csv(out/"selector-agreement.csv",index=False)
    pd.DataFrame(all_signal).to_csv(out/"tail-signal-race-audit.csv",index=False)
    pd.DataFrame(exclusions).to_csv(out/"exclusions.csv",index=False)
    pd.DataFrame(missing).to_csv(out/"missing-payouts.csv",index=False)

    # Compact comparison: BASE and each selector's k=1..3 for BOX, axis, and fixed ticket budgets.
    summary={
        "contract":"L17_TAIL_RESCUE_ROI_BATCH_V1_RESULT",
        "pooled_strategy_metrics":pooled,
        "selector_agreement":agreement,
        "tail_signal_coverage":signal,
        "skipped_market_races":len(skipped_market),
        "stake_per_ticket_yen":STAKE,
        "probability_times_odds":False,
        "trio_odds_used_for_ranking":False,
        "2026_locked":True,
        "elapsed_seconds":time.time()-t0
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L1.7 Tail Rescue ROI Batch V1\n\n"
        "Economic audit of horse-level Top12 plus 1-3 rescue horses from ranks 13-18. "
        "Includes full BOX, King1-axis full flow, and fixed ticket budgets ranked by Seven-King PL or L1.75 p3. "
        "No TRIO odds are used to rank or promote tickets. Flat 100-yen stake per ticket.\n",
        encoding="utf-8"
    )
    print("===== TAIL SIGNAL COVERAGE ====="); print(json.dumps(signal,ensure_ascii=False,indent=2))
    print("===== SELECTOR AGREEMENT ====="); print(pd.DataFrame(agreement).to_string(index=False))
    print("===== POOLED STRATEGIES ====="); print(pd.DataFrame(pooled).to_string(index=False))
    print("L17_TAIL_RESCUE_ROI_BATCH_V1_READY")

if __name__=="__main__":
    main()
