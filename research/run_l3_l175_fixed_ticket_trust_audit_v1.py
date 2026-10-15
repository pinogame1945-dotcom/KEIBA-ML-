#!/usr/bin/env python3
"""Audit L1.75/King+Outsider signals on FROZEN L2 stable-medium physical tickets.
No new picks, skips, rankings, stake weights or target-leaking ticket gates.
L1.75 OOS p3 and p3_delta are estimated podium probabilities, NOT exact bet probabilities.
L1.75 source has King ranks <= 10 only; uncovered tickets are retained and reported.
2026 excluded. Historical FINAL odds are descriptive, not live-execution proof.
"""
from __future__ import annotations
import argparse, csv, gzip, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_l2_all_fixed_odds_band_lenses_v1 as lens
import run_l2_fixed_ticket_pruning_3040_v1 as fixed
from build_l2_market_helpers_v1 import load_day, load_odds_day, horse_number_map

YEARS=(2023,2024,2025)
ACCEPTED_TAGS={"STABLE_MEDIUM":"STANDARD","STABLE_STRICT":"STRICT"}

def cli():
    p=argparse.ArgumentParser()
    for name in ("predictions","picks","racewise","accepted-lines","stable-portfolio",
                 "l175-predictions","backfill-root","out-dir"):
        p.add_argument("--"+name,required=True)
    return p.parse_args()

def load_accepted(path):
    allowed={t:set() for t in ACCEPTED_TAGS}
    with open(path,encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            if r["policy"] not in allowed:continue
            key=tuple(r["line"].split("|"))
            if len(key)!=4:raise ValueError("bad line "+r["line"])
            allowed[r["policy"]].add((key,r["gate"]))
    assert len(allowed["STABLE_MEDIUM"])==50,len(allowed["STABLE_MEDIUM"])
    assert len(allowed["STABLE_STRICT"])==26,len(allowed["STABLE_STRICT"])
    assert allowed["STABLE_STRICT"].issubset(allowed["STABLE_MEDIUM"])
    index=defaultdict(list)
    for pol,pairs in allowed.items():
        for line,gate in pairs:index[line].append((pol,gate))
    return index

def load_l175(path):
    # Explicit allowlist excludes historical finish/y3/outcome columns.
    cols=["year","race_id","horse_id","rank","score","p3","p3_rank_baseline","p3_delta"]
    d=pd.read_csv(path,compression="gzip",usecols=cols,
                  dtype={"race_id":str,"horse_id":str},low_memory=False)
    d["year"]=pd.to_numeric(d.year,errors="raise").astype(int)
    d=d[d.year.isin(YEARS)]
    assert not d.duplicated(["year","race_id","horse_id"]).any()
    assert d["rank"].between(1,10).all()
    assert d["score"].between(0,39).all()
    assert d["p3"].between(0,1).all()
    assert d["p3_rank_baseline"].between(0,1).all()
    assert np.allclose(d["p3"]-d["p3_rank_baseline"],d["p3_delta"],atol=1e-8)
    by_race=defaultdict(dict)
    for z in d.itertuples(index=False):
        by_race[(int(z.year),str(z.race_id))][str(z.horse_id)]=(
            int(z.rank),int(z.score),float(z.p3),float(z.p3_delta))
    print("L175_SAFE_OOS_HORSE_SIGNALS",len(d),"RACES",len(by_race),flush=True)
    return by_race

def tickets_by_year(year, pred, picks, races, root, line_index, source):
    grp=pred[pred.year.eq(year)].groupby("_race_id",sort=False)
    yrs=races[races.year.eq(year)]
    assert len(yrs)==3456
    idx=yrs.set_index("race_id")
    dates=yrs.groupby("race_date")["race_id"].apply(list)
    rows=[]
    for di,(date,ids) in enumerate(dates.items(),1):
        pack_by_id=load_day(root/"data"/"daily"/(date+".jsonl.gz"),set(ids))
        odds_by_id=load_odds_day(root/"data"/"odds"/"daily"/(date+".jsonl.gz"),set(ids))
        for rid in sorted(ids):
            pack=pack_by_id.get(rid);market=odds_by_id.get(rid)
            assert pack is not None and market is not None,("source drift",date,rid)
            assert market.get("odds_kind")=="FINAL" and market.get("source_status")=="result"
            group=grp.get_group(rid)
            info=dict(idx.loc[rid]);info["race_id"]=rid
            axis=picks["ANCHOR_FULL"][rid]
            no=horse_number_map(pack)
            inv={v:k for k,v in no.items()}
            anchor_no=no[axis]
            l175=source.get((year,rid),{})
            union={}
            seen_return={}
            gate_cache={}
            horse_signals=None
            for tid,ticket,card in fixed.create_cards(group,pack,market,rid,axis,info):
                line=tuple(map(str,tid))
                requirements=line_index.get(line)
                if not requirements:continue
                returned=int(card[1])
                if ticket in seen_return:assert seen_return[ticket]==returned
                seen_return[ticket]=returned
                if ticket not in gate_cache:
                    if horse_signals is None:
                        horse_signals,_=lens_horses(group,pack)
                    gate_cache[ticket]=lens.gate_flags(ticket,horse_signals,anchor_no)
                gate=gate_cache[ticket]
                matched=[pol for pol,name in requirements if gate[name]]
                if not matched:continue
                # Strict tickets are a subset of standard tickets; never double-buy.
                if ticket in union:
                    union[ticket]["strict"] |= int("STABLE_STRICT" in matched)
                    continue
                bet_type,nums=ticket
                hids=[inv[n] for n in nums]
                signals=[l175.get(h) for h in hids]
                full=all(x is not None for x in signals)
                if full:
                    p3=[x[2] for x in signals]
                    delt=[x[3] for x in signals]
                    supports=[x[1] for x in signals]
                    kin=[x[0] for x in signals]
                else:p3=delt=supports=kin=[]
                union[ticket]=dict(
                    year=year,race_id=rid,bet_type=bet_type,
                    selection="-".join(map(str,nums)),ticket_odds=float(card[0]),
                    odds_band=line[-1],hit=int(returned>0),
                    payout_yen=returned,strict=int("STABLE_STRICT" in matched),
                    l175_complete=int(full),l175_matched_horses=sum(z is not None for z in signals),
                    l175_ticket_horses=len(nums),
                    min_p3=min(p3) if full else np.nan,
                    mean_p3=float(np.mean(p3)) if full else np.nan,
                    min_p3_delta=min(delt) if full else np.nan,
                    mean_p3_delta=float(np.mean(delt)) if full else np.nan,
                    support_min=min(supports) if full else np.nan,
                    support_sum=sum(supports) if full else np.nan,
                    king_worst_rank=max(kin) if full else np.nan,
                    all_delta_positive=int(all(v>0 for v in delt)) if full else -1,
                    all_delta_negative=int(all(v<0 for v in delt)) if full else -1)
            rows.extend(union.values())
        if di%60==0:
            print("TRUST_REPLAY_PROGRESS",year,"days",di,"tickets",len(rows),flush=True)
    return rows

def lens_horses(group,pack):
    no=horse_number_map(pack)
    rank=group.sort_values(["P__ANCHOR_FULL","market_rank","_horse_id"],
        ascending=[False,True,True],kind="mergesort")
    pr=dict(zip(rank["_horse_id"].astype(str),range(1,len(rank)+1)))
    horse={}
    for _,row in group.iterrows():
        hid=str(row["_horse_id"])
        horse[no[hid]]={
            "P__ANCHOR_FULL":float(row["P__ANCHOR_FULL"]),
            "market_prob_norm":float(row["market_prob_norm"]),
            "risk":float(row["risk"]),
            "distill_score_z":float(row["distill_score_z"]),
            "distill_ai_rank":int(row["distill_ai_rank"]),
            "market_rank":int(row["market_rank"]),
            "P_rank":pr[hid]}
    assert len(horse)==len(group)
    return horse,no

def group_stat(g):
    n=len(g);hits=int(g.hit.sum());ret=int(g.payout_yen.sum())
    biggest=sorted((int(x) for x in g.loc[g.hit.eq(1),"payout_yen"]),reverse=True)[:3]
    sub=ret-sum(biggest)
    return dict(tickets=n,hits=hits,
        hit_rate_pct=round(100*hits/n,5) if n else None,
        races=int(g[["year","race_id"]].drop_duplicates().shape[0]),
        payout_yen=ret,stake_yen=n*100,profit_yen=ret-100*n,
        roi_pct=round(100*ret/(100*n),5) if n else None,
        roi_minus_three_biggest_pct=round(100*sub/(100*n),5) if n else None)

def report(df,dimension,out):
    rows=[]
    for ((year,label),g) in df.groupby(["report_year",dimension],dropna=False,sort=True):
        row={"dimension":dimension,"bucket":str(label),"year":str(year)}
        row.update(group_stat(g));rows.append(row)
    rows.sort(key=lambda x:(x["dimension"],x["bucket"],x["year"]))
    return rows

def main():
    a=cli()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    accepted=load_accepted(a.accepted_lines)
    source=load_l175(a.l175_predictions)
    pred,picks=lens.frame(a.predictions,a.picks)
    races=pd.read_csv(a.racewise,compression="infer",dtype={"race_id":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert set(races.year)==set(YEARS)
    assert len(races)==3*3456 and races.race_id.is_unique
    records=[]
    for year in YEARS:
        chunk=tickets_by_year(year,pred,picks,races,Path(a.backfill_root),accepted,source)
        records.extend(chunk)
        print("YEAR_PHYSICAL_TICKETS_REPLAYED",year,len(chunk),flush=True)
    df=pd.DataFrame.from_records(records)
    assert not df.duplicated(["year","race_id","bet_type","selection"]).any()
    with open(a.stable_portfolio,encoding="utf-8",newline="") as f:
        baselines={(r["year"],r["portfolio"]):r for r in csv.DictReader(f)}
    for year in YEARS:
        year_data=df[df.year.eq(year)]
        for policy,label in [("STABLE_MEDIUM","STANDARD"),("STABLE_STRICT","STRICT")]:
            part=year_data if label=="STANDARD" else year_data[year_data.strict.eq(1)]
            stats=group_stat(part);ref=baselines[(str(year),policy)]
            for fld,col in [("tickets","tickets"),("hits","winning_tickets"),
                            ("payout_yen","return_yen"),("races","races_bought")]:
                assert int(stats[fld])==int(ref[col]),("BASELINE_MISMATCH",year,policy,fld,stats[fld],ref[col])
    summary={"contract":"L3_L175_FROZEN_TICKET_TRUST_AUDIT_V1",
             "L2_purchase_policy":"STABLE_MEDIUM / 50 unchanged line-gates / exact physical dedup",
             "strict_comparison":"STRICT member ticket subset (not an additional purchase)",
             "years":[2023,2024,2025],"horse_source":"frozen rank<=10 OOS L1.7 King plus Outsider podium calibrator run-36747215581",
             "data_source_features":["king_rank","outsider_score","p3","p3_delta"],
             "not_a_hard_l175_confidence_tier":True,
             "p3_is_podium_probability_not_exact_ticket_win_probability":True,
             "signals_use_only_past_year_calibration":True,
             "confounding":"odds band and bet type; report stratified associations and do not infer bet EV from p3 alone",
             "historical_final_odds_only":True,"2026_sealed":True,
             "never_change_axis_tickets_or_stakes":True}
    summary["standard_baseline"]=group_stat(df)
    summary["strict_baseline"]=group_stat(df[df.strict.eq(1)])
    summary["l175_full_coverage"]=dict(
      tickets=int(df.l175_complete.sum()),total=len(df),
      coverage_pct=100*float(df.l175_complete.mean()),
      missing_tickets=int(df.l175_complete.eq(0).sum()),
      reasons="source only has historical King ranks <=10; unmatched never imputed")
    for kind,sub in df.groupby("bet_type"):
        summary.setdefault("coverage_by_bet_type",{})[str(kind)] =dict(
            total=len(sub),full=int(sub.l175_complete.sum()),
            full_pct=round(100*sub.l175_complete.mean(),3))
    df["report_year"]=df.year.astype(str)
    known=df[df.l175_complete.eq(1)].copy()
    unknown=df[df.l175_complete.eq(0)].copy()
    assert len(known)+len(unknown)==len(df)
    if len(known)<1000:raise ValueError("L175 FULL COVERAGE TOO LOW; cannot validate trust tiers")
    # Descriptive percentiles per bet type based on PRE-RACE signals only; never labels.
    known["p3_tier"]=pd.Series(index=known.index,dtype="object")
    boundaries={}
    for bet,g in known.groupby("bet_type"):
        vals=g.min_p3
        rank=vals.rank(pct=True,method="average")
        tier=np.minimum(5,np.maximum(1,np.ceil(rank*5))).astype(int)
        known.loc[g.index,"p3_tier"]=["Q"+str(v) for v in tier]
        boundaries[bet]={f"Q{i}":dict(
            min_p3=float(vals[tier==i].min()) if (tier==i).any() else None,
            max_p3=float(vals[tier==i].max()) if (tier==i).any() else None,
            count=int((tier==i).sum())) for i in range(1,6)}
    summary["p3_min_quintile_by_bet_type"]=boundaries
    # Outsider raw 3-2-1 support and rank-relative uplift are distinct from p3.
    known["out_score_bucket"]=pd.cut(known.support_min,[-1,0,3,9,39],
                                  labels=["0","1-3","4-9","10+"],include_lowest=True)
    known["delta_agreement"]=np.select(
        [known.all_delta_positive.eq(1),known.all_delta_negative.eq(1)],
        ["ALL_POS","ALL_NEG"],default="MIXED")
    known["type_x_tier"]=known.bet_type.astype(str)+"|"+known.p3_tier.astype(str)
    known["odds_x_tier"]=known.odds_band.astype(str)+"|"+known.p3_tier.astype(str)
    known["type_odds_tier"]=known.bet_type.astype(str)+"|"+known.odds_band.astype(str)+"|"+known.p3_tier.astype(str)
    known["year_x_tier"]=known.report_year+"|"+known.p3_tier
    known["strict_x_tier"]=np.where(known.strict.eq(1),"STRICT","STANDARD_ONLY")+"|"+known.p3_tier
    known["coverage"]="FULL_L175"
    unknown["coverage"]="MISSING_KING_TOP10"
    combined=pd.concat([known,unknown],ignore_index=True,sort=False)
    report_rows=[]
    dimensions=["p3_tier","out_score_bucket","delta_agreement","type_x_tier",
                "odds_x_tier","type_odds_tier","strict_x_tier","coverage"]
    for d in dimensions:
        x=combined if d=="coverage" else known
        report_rows.extend(report(x,d,out))
        z=x.copy();z["report_year"]="2023-2025"
        report_rows.extend(report(z,d,out))
    pd.DataFrame(report_rows).to_csv(out/"trust_hit_roi_by_tier_year_and_context.csv",index=False)
    summary["yearly"]={str(y):group_stat(df[df.year.eq(y)]) for y in YEARS}
    summary["signal_rank_association"]={}
    for col in ("min_p3","mean_p3","mean_p3_delta","support_sum","support_min"):
        summary["signal_rank_association"][col]=dict(
            spearman_hit=float(known[col].corr(known.hit,method="spearman")),
            spearman_per100_return=float(known[col].corr(known.payout_yen,method="spearman")))
    summary["ticket_group_counts"]={str(k):int(v) for k,v in known.bet_type.value_counts().items()}
    # Keep a compact exact joined ledger for reproducibility, not hidden cherry-picked rows.
    cols=["year","race_id","bet_type","selection","ticket_odds","odds_band","hit",
          "payout_yen","strict","l175_complete","l175_matched_horses","l175_ticket_horses",
          "min_p3","mean_p3","min_p3_delta","mean_p3_delta","support_min","support_sum",
          "king_worst_rank","p3_tier","out_score_bucket","delta_agreement"]
    known[cols].to_csv(out/"joined_ticket_trust_complete.csv.gz",index=False,compression="gzip")
    df.loc[df.l175_complete.eq(0),["year","race_id","bet_type","selection","odds_band",
          "hit","payout_yen","l175_matched_horses","l175_ticket_horses"]].to_csv(
          out/"tickets_missing_l175_signal.csv.gz",index=False,compression="gzip")
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    with open(out/"trust_hit_roi_by_tier_year_and_context.csv",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        highlights=[r for r in reader if r["dimension"]=="p3_tier" and r["year"]=="2023-2025"]
    print("L3_L175_FROZEN_TICKET_TRUST_AUDIT_DONE",json.dumps(summary,ensure_ascii=False),flush=True)
    print("P3_TIER_HEADLINE",json.dumps(highlights,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
