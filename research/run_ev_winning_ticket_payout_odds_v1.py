#!/usr/bin/env python3
"""Winning-ticket payout-multiplier spectrum of frozen EV selected anchor.

This is NOT ROI conditioned on the odds of losing tickets: only winning tickets
have actual payout multiplier in the archived exact-dividend packs. Separate
pre-race selected WIN-odds bands measure true ROI for all placed tickets.
"""
import argparse,concurrent.futures,json
from pathlib import Path
from collections import Counter,defaultdict
import pandas as pd
import run_l2_value_anchor_bet_frontier_v1 as prior
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,get_inputs,assert_ok
from build_l2_market_helpers_v1 import load_day,horse_number_map

PAYOUT_BANDS=[(1,2),(2,3),(3,5),(5,10),(10,20),(20,50),(50,100),
              (100,200),(200,500),(500,1000),(1000,3000),(3000,float("inf"))]
HORSE_BANDS=[(1,1.5),(1.5,2),(2,3),(3,5),(5,10),(10,20),(20,50),
             (50,100),(100,float("inf"))]

def band(value,bins):
    for low,high in bins:
        if low<=value<high:
            return str(low)+"-"+(str(high) if high!=float("inf") else "+")
    raise ValueError("not in bins: "+str(value))

def aggregate_one(year,pred,picks,races,root):
    group=pred.loc[pred.year.eq(year)].copy()
    group["_race_date"]=group["_race_date"].astype(str).str[:10]
    indexed=group.groupby("_race_id",sort=False)
    relevant=races[races.year.eq(year)].set_index("race_id")
    dates=group[["_race_id","_race_date"]].drop_duplicates().groupby("_race_date")["_race_id"].apply(list)
    assert_ok(len(relevant)==RACES_YEAR,"missing year races")
    winning=defaultdict(Counter)
    horses=defaultdict(Counter)
    totals=defaultdict(Counter)
    for day,ids in dates.items():
        packs=load_day(root/"data"/"daily"/(str(day)+".jsonl.gz"),set(ids))
        for rid in ids:
            pack=packs.get(rid)
            assert_ok(pack is not None,"missing race pack "+str(rid))
            g=indexed.get_group(rid)
            row=relevant.loc[rid]
            anchor=picks["ANCHOR_FULL"][rid]
            assert_ok(str(row["VALUE_EV_horse"])==anchor,"selected identity drift "+rid)
            no=horse_number_map(pack)
            assert_ok(anchor in no, "selected horse unmapped "+rid)
            payout=prior.payouts(pack,rid)
            anchor_no=no[anchor]
            ai=(g.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
                ascending=[False,True,True],kind="mergesort")["_horse_id"].astype(str).tolist())
            market=(g.sort_values(["market_rank","final_win_odds","_horse_id"],
                ascending=[True,True,True],kind="mergesort")["_horse_id"].astype(str).tolist())
            selection=g.set_index("_horse_id").loc[anchor]
            odds=float(selection["final_win_odds"])
            scopes=["ALL"]
            if anchor==str(row["AI_PROB_TOP_horse"]):
                scopes.append("EV_EQ_PROB")
                if anchor==str(row["MARKET_TOP_horse"]):
                    scopes.append("EV_EQ_PROB_EQ_MARKET")
            tickets={("WIN","SINGLE",0):{("WIN",(anchor_no,))},
                     ("PLACE","SINGLE",0):{("PLACE",(anchor_no,))}}
            for label,seq in (("AI_PROB",ai),("MARKET",market)):
                others=[no[h] for h in seq if h!=anchor]
                assert_ok(len(others)==len(seq)-1,"missing partner map "+rid)
                for k in prior.PARTNER_COUNTS:
                    bytype=prior.pair_templates(anchor_no,others[:k])
                    for family,t in bytype.items():
                        tickets[(family,label,k)]=t
            assert_ok(len(tickets)==82,"unexpected templates")
            ob=band(odds,HORSE_BANDS)
            for tpl,ticket_set in tickets.items():
                receipt=[payout.get(z,0.0) for z in ticket_set]
                hits=[v for v in receipt if v>0]
                for scope in scopes:
                    key=(scope,*tpl)
                    t=totals[key]
                    t["races"]+=1
                    t["tickets"]+=len(ticket_set)
                    t["stake_yen"]+=100*len(ticket_set)
                    t["return_yen"]+=sum(receipt)
                    t["hit_races"]+=int(bool(hits))
                    t["hit_tickets"]+=len(hits)
                    w=horses[(*key,ob)]
                    w["races"]+=1
                    w["stake_yen"]+=100*len(ticket_set)
                    w["return_yen"]+=sum(receipt)
                    w["hit_races"]+=int(bool(hits))
                    for v in hits:
                        b=band(v/100,PAYOUT_BANDS)
                        kstats=winning[(*key,b)]
                        kstats["hit_tickets"]+=1
                        kstats["return_yen"]+=v
                        kstats["hit_races_approx"]+=1
    assert_ok(totals[("ALL","WIN","SINGLE",0)]["races"]==RACES_YEAR,
              "incomplete year")
    return winning,horses,totals

def roll(chunks):
    outputs=[defaultdict(Counter) for _ in range(3)]
    for yr,stats in zip(YEARS,chunks):
        for i,part in enumerate(stats):
            for key,value in part.items():
                outputs[i][(str(yr),*key)].update(value)
                outputs[i][("2023-2025",*key)].update(value)
    return outputs

def to_frame(data,kind,totals):
    out=[]
    for key,ct in data.items():
        year,scope,fam,src,n,b=key
        total=totals[(year,scope,fam,src,n)]
        if kind=="winning":
            out.append(dict(year=year,scope=scope,family=fam,partner_order=src,k=n,
              payout_odds_band=b,hit_tickets=ct["hit_tickets"],return_yen=ct["return_yen"],
              share_hit_tickets_pct=round(100*ct["hit_tickets"]/total["hit_tickets"],5),
              share_total_return_pct=round(100*ct["return_yen"]/total["return_yen"],5)))
        else:
            out.append(dict(year=year,scope=scope,family=fam,partner_order=src,k=n,
              selected_horse_win_odds_band=b,races=ct["races"],
              stake_yen=ct["stake_yen"],return_yen=ct["return_yen"],
              hit_races=ct["hit_races"],roi_pct=round(100*ct["return_yen"]/ct["stake_yen"],5),
              share_races_pct=round(100*ct["races"]/total["races"],5)))
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True);ap.add_argument("--picks",required=True)
    ap.add_argument("--racewise",required=True);ap.add_argument("--backfill-root",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    d,picks=get_inputs(a.predictions,a.picks)
    # Critical: horse IDs are strings, not floats/ints inferred by pandas.
    id_cols={s+"_horse":str for s in ("AI_RAW_TOP","AI_PROB_TOP","MARKET_TOP","VALUE_EV")}
    id_cols["race_id"]=str
    races=pd.read_csv(a.racewise,compression="infer",dtype=id_cols)
    races["year"]=pd.to_numeric(races["year"],errors="raise").astype(int)
    assert_ok(len(races)==TOTAL_RACES and races.race_id.is_unique
              and set(races.year)==set(YEARS),"OOS race identity drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        chunks=list(pool.map(lambda year:aggregate_one(year,d,picks,races,
                         Path(a.backfill_root)),YEARS))
    wins,horse,totals=roll(chunks)
    win_df=to_frame(wins,"winning",totals).sort_values(
        ["year","scope","family","partner_order","k","payout_odds_band"])
    horse_df=to_frame(horse,"horses",totals).sort_values(
        ["year","scope","family","partner_order","k","selected_horse_win_odds_band"])
    win_df.to_csv(out/"winning-ticket-actual-payout-odds-bands.csv",index=False)
    horse_df.to_csv(out/"selected-horse-win-odds-bands-real-roi.csv",index=False)
    checks={
      ("WIN","SINGLE",0):(900960,1917,10368),
      ("PLACE","SINGLE",0):(865630,4467,10368),
      ("TRIO","AI_PROB",2):(921600,866,10368),
      ("EXACTA_FIRST","MARKET",2):(1839560,1000,20736),
      ("WIDE","MARKET",2):(1782070,3561,20736),
    }
    for template,(ret,hits,tickets) in checks.items():
        c=totals[("2023-2025","ALL",*template)]
        assert_ok(round(c["return_yen"])==ret
             and c["hit_races"]==hits and c["tickets"]==tickets,
             "baseline drift "+str(template)+" "+str(c))
    assert_ok(len(set((year,scope,fam,src,k)
          for year,scope,fam,src,k in totals if year=="2023-2025" and scope=="ALL"))==82,
          "82 original strategies not covered")
    summaries=[]
    for (year,scope,fam,src,k),c in totals.items():
        summaries.append({"year":year,"scope":scope,"family":fam,"partner_order":src,
         "k":k,"races":c["races"],"tickets":c["tickets"],"stake_yen":c["stake_yen"],
         "return_yen":c["return_yen"],"hit_races":c["hit_races"],
         "hit_tickets":c["hit_tickets"],"roi_pct":round(100*c["return_yen"]/c["stake_yen"],6)})
    pd.DataFrame(summaries).to_csv(out/"all-template-verified-totals.csv",index=False)
    focus=(("WIN","SINGLE",0),("PLACE","SINGLE",0),
           ("TRIO","AI_PROB",2),("EXACTA_FIRST","MARKET",2),
           ("WIDE","MARKET",2),("TRIFECTA_FIRST","AI_PROB",7))
    for scope in ("ALL","EV_EQ_PROB","EV_EQ_PROB_EQ_MARKET"):
        for fam,src,k in focus:
            q=win_df[(win_df.year=="2023-2025")&(win_df.scope==scope)&
                     (win_df.family==fam)&(win_df.partner_order==src)&(win_df.k==k)]
            t=totals.get(("2023-2025",scope,fam,src,k),{})
            print("WINNING_ODDS_HISTOGRAM",json.dumps({"scope":scope,"template":[fam,src,k],
                "races":t.get("races"),"all_ticket_hits":t.get("hit_tickets"),
                "overall_roi":round(100*t.get("return_yen",0)/t.get("stake_yen",1),5),
                "bands":q[["payout_odds_band","hit_tickets","return_yen",
                         "share_hit_tickets_pct","share_total_return_pct"]].to_dict("records")},
                 ensure_ascii=False),flush=True)
    print("HISTOGRAM_COMPLETE")

if __name__=="__main__":
    main()
