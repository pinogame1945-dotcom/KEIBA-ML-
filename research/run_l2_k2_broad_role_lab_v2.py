#!/usr/bin/env python3
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import (
    load_day,load_odds_day,decode_odds,payout_map
)
from run_l2_bet_kings_arena_v1 import (
    load_template,feature_columns,encode_fit_other,chronological_split,make_model,
    safe_auc,safe_ap
)

YEARS=(2022,2023,2024,2025)
TEST_YEARS=(2023,2024,2025)
COVERAGE_TARGETS=(1.00,0.75,0.50,0.25)
TEMPLATE="WIN_ALL_CANDIDATES"
STRUCTURAL_DROP={"selected_outsider_1","selected_outsider_2"}


def parse_args():
    p=argparse.ArgumentParser(description="K2 broad role learner: separate 1st-route and A1/A2->K2 3rd-route without over-narrowing.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def canonical(bet,nums):
    nums=tuple(int(x) for x in nums)
    if bet in {"QUINELLA","TRIO"}:
        return tuple(sorted(nums))
    return nums


def build_k2_table(df,backfill_root):
    by_date=defaultdict(set)
    by_race={}
    for rid,grp in df.groupby(df["race_id"].astype(str),sort=False):
        rows=grp.to_dict("records")
        date=str(rows[0]["race_date"])[:10]
        by_date[date].add(rid)
        by_race[rid]=rows

    out=[]
    root=Path(backfill_root)
    processed=0
    for di,date in enumerate(sorted(by_date),1):
        wanted=by_date[date]
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        odds_day=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in sorted(wanted):
            rows=by_race[rid]
            year=int(float(rows[0]["year"]))
            if year not in YEARS:
                continue
            pack=day.get(rid); oddsrec=odds_day.get(rid)
            if pack is None or oddsrec is None:
                raise SystemExit(f"missing market pack race={rid}")
            odds_map=decode_odds(oddsrec)
            payouts,present=payout_map(pack)
            if "EXACTA" not in present or "TRIFECTA" not in present:
                continue

            win_market=[
                (int(nums[0]),float(odd))
                for (bt,nums),odd in odds_map.items()
                if bt=="WIN" and odd is not None and float(odd)>0
            ]
            win_market.sort(key=lambda x:(x[1],x[0]))
            rank_map={no:i+1 for i,(no,_) in enumerate(win_market)}
            win_odds={no:odd for no,odd in win_market}

            cand=[int(float(r["selection_numbers"])) for r in rows]
            a1s=[r for r in rows if int(float(r.get("s1_is_anchor1") or 0))==1]
            a2s=[r for r in rows if int(float(r.get("s1_is_anchor2") or 0))==1]
            if len(a1s)!=1 or len(a2s)!=1:
                raise SystemExit(f"anchor cardinality race={rid}")
            a1=int(float(a1s[0]["selection_numbers"]))
            a2=int(float(a2s[0]["selection_numbers"]))

            for r in rows:
                if int(float(r.get("s1_is_novel") or 0))!=1:
                    continue
                h=int(float(r["selection_numbers"]))
                if h in {a1,a2}:
                    raise SystemExit(f"K2 novel overlaps anchor race={rid} h={h}")
                mr=rank_map.get(h)
                wo=win_odds.get(h)
                if mr is None or wo is None:
                    continue

                ex_tickets=[]
                for x in cand:
                    if x==h: continue
                    nums=(h,x)
                    if ("EXACTA",nums) in odds_map:
                        ex_tickets.append(nums)
                tri_tickets=[]
                for nums in ((a1,a2,h),(a2,a1,h)):
                    if ("TRIFECTA",nums) in odds_map:
                        tri_tickets.append(nums)
                if not ex_tickets or len(tri_tickets)!=2:
                    continue

                ex_ret=sum(float(payouts.get(("EXACTA",nums),0.0)) for nums in ex_tickets)
                tri_ret=sum(float(payouts.get(("TRIFECTA",nums),0.0)) for nums in tri_tickets)

                z=dict(r)
                z.update({
                    "k2_market_rank":int(mr),
                    "k2_win_odds":float(wo),
                    "k2_win_implied":1.0/float(wo),
                    "target_first":int(bool(r["hit"])),
                    "target_a12_third":int(tri_ret>0),
                    "exacta_first_tickets":len(ex_tickets),
                    "exacta_first_stake":100.0*len(ex_tickets),
                    "exacta_first_return":ex_ret,
                    "trifecta_a12_third_tickets":len(tri_tickets),
                    "trifecta_a12_third_stake":100.0*len(tri_tickets),
                    "trifecta_a12_third_return":tri_ret,
                })
                out.append(z)
            processed+=1
        if di%50==0:
            print(f"K2_TABLE_PROGRESS dates={di}/{len(by_date)} races={processed}",flush=True)
    if not out:
        raise SystemExit("no K2 rows")
    return pd.DataFrame(out)


def thresholds_from_cal(scores):
    s=np.asarray(scores,dtype=float)
    return {
        1.00:-math.inf,
        0.75:float(np.quantile(s,0.25)),
        0.50:float(np.quantile(s,0.50)),
        0.25:float(np.quantile(s,0.75)),
    }


def concentration(selected,return_col,stake_col):
    if selected.empty:
        return {"top1_return_share_pct":None,"top5_return_share_pct":None,"roi_without_top1_pct":None}
    race=(selected.groupby("race_id",as_index=False)
          .agg(stake=(stake_col,"sum"),ret=(return_col,"sum")))
    total_ret=float(race["ret"].sum()); total_stake=float(race["stake"].sum())
    vals=sorted([float(x) for x in race["ret"]],reverse=True)
    top1=vals[0] if vals else 0.0
    top5=sum(vals[:5])
    return {
        "top1_return_share_pct":100*top1/total_ret if total_ret>0 else None,
        "top5_return_share_pct":100*top5/total_ret if total_ret>0 else None,
        "roi_without_top1_pct":100*max(0,total_ret-top1)/total_stake if total_stake>0 else None,
    }


def eval_portfolio(test,route,target_coverage,threshold,score_col):
    if route=="FIRST":
        stake_col="exacta_first_stake"; ret_col="exacta_first_return"; target_col="target_first"
    else:
        stake_col="trifecta_a12_third_stake"; ret_col="trifecta_a12_third_return"; target_col="target_a12_third"
    sel=test[test[score_col]>=threshold].copy()
    stake=float(sel[stake_col].sum()) if len(sel) else 0.0
    ret=float(sel[ret_col].sum()) if len(sel) else 0.0
    source_races=int(test["race_id"].nunique())
    selected_races=int(sel["race_id"].nunique()) if len(sel) else 0
    tickets=int(sel["exacta_first_tickets"].sum()) if route=="FIRST" and len(sel) else (
        int(sel["trifecta_a12_third_tickets"].sum()) if len(sel) else 0
    )
    c=concentration(sel,ret_col,stake_col)
    return {
        "route":route,
        "target_coverage":target_coverage,
        "score_threshold_from_prior_calibration":threshold,
        "source_instances":int(len(test)),
        "selected_instances":int(len(sel)),
        "instance_coverage_pct":100*len(sel)/len(test) if len(test) else 0.0,
        "source_races":source_races,
        "selected_races":selected_races,
        "race_coverage_pct":100*selected_races/source_races if source_races else 0.0,
        "tickets":tickets,
        "stake_yen":stake,
        "return_yen":ret,
        "profit_yen":ret-stake,
        "roi_pct":100*ret/stake if stake else None,
        "portfolio_hit_instances":int((sel[ret_col]>0).sum()) if len(sel) else 0,
        "portfolio_hit_rate_pct":100*float((sel[ret_col]>0).mean()) if len(sel) else 0.0,
        "target_rate_pct":100*float(sel[target_col].mean()) if len(sel) else 0.0,
        "avg_market_rank":float(sel["k2_market_rank"].mean()) if len(sel) else None,
        "avg_win_odds":float(sel["k2_win_odds"].mean()) if len(sel) else None,
        **c,
    }


def stability(metrics):
    idx=defaultdict(dict)
    for r in metrics:
        idx[(r["route"],float(r["target_coverage"]))][int(r["test_year"])]=r
    out=[]
    for (route,cov),yr in sorted(idx.items()):
        if not all(y in yr for y in TEST_YEARS):
            continue
        stake=sum(yr[y]["stake_yen"] for y in TEST_YEARS)
        ret=sum(yr[y]["return_yen"] for y in TEST_YEARS)
        row={
            "route":route,
            "target_coverage":cov,
            "roi_2023":yr[2023]["roi_pct"],
            "roi_2024":yr[2024]["roi_pct"],
            "roi_2025":yr[2025]["roi_pct"],
            "race_coverage_2023":yr[2023]["race_coverage_pct"],
            "race_coverage_2024":yr[2024]["race_coverage_pct"],
            "race_coverage_2025":yr[2025]["race_coverage_pct"],
            "combined_roi_pct":100*ret/stake if stake else None,
            "combined_profit_yen":ret-stake,
            "min_year_roi_pct":min(yr[y]["roi_pct"] for y in TEST_YEARS),
            "min_year_race_coverage_pct":min(yr[y]["race_coverage_pct"] for y in TEST_YEARS),
            "all_years_100plus":int(all(yr[y]["roi_pct"]>=100 for y in TEST_YEARS)),
            "broad_coverage_50plus":int(all(yr[y]["race_coverage_pct"]>=50 for y in TEST_YEARS)),
        }
        row["broad_stable_candidate"]=int(row["all_years_100plus"] and row["broad_coverage_50plus"])
        out.append(row)
    out.sort(key=lambda r:(-r["broad_stable_candidate"],-r["min_year_race_coverage_pct"],-r["min_year_roi_pct"],-r["combined_roi_pct"],r["route"]))
    return out


def main():
    a=parse_args()
    manifest=json.loads((Path(a.dataset_dir)/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("wrong L1.5 source")
    if 2026 not in manifest.get("locked_years",[]):
        raise SystemExit("2026 lock missing")

    df=load_template(Path(a.dataset_dir)/manifest["templates"][TEMPLATE]["file"])
    df["race_month"]=pd.to_numeric(df["race_date"].astype(str).str.slice(5,7),errors="coerce").fillna(0.0)
    base_cols=[c for c in feature_columns(df) if c not in STRUCTURAL_DROP]
    if any(x in c.lower() for c in base_cols for x in ("odds","popularity","payout","return","profit","roi","hit","finish","winner")):
        raise SystemExit(f"market/outcome leakage in structural cols {base_cols}")

    k2=build_k2_table(df,a.backfill_root)
    metrics=[]; quality=[]

    route_defs=(
        ("FIRST","target_first","exacta_first_stake","exacta_first_return"),
        ("A12_THIRD","target_a12_third","trifecta_a12_third_stake","trifecta_a12_third_return"),
    )

    for route_index,(route,target_col,_,_) in enumerate(route_defs):
        for year_index,test_year in enumerate(TEST_YEARS):
            train=k2[k2["year"]<test_year].copy()
            test=k2[k2["year"]==test_year].copy()
            split=chronological_split(train)
            if split is None:
                raise SystemExit(f"split unavailable route={route} year={test_year}")
            fit,cal,fit_races,cal_races=split
            yfit=fit[target_col].astype(int).to_numpy()
            ycal=cal[target_col].astype(int).to_numpy()
            ytest=test[target_col].astype(int).to_numpy()
            if len(set(yfit.tolist()))<2:
                raise SystemExit(f"single class route={route} year={test_year}")
            xfit,(xcal,xtest)=encode_fit_other(fit,[cal,test],base_cols)
            model=make_model(120000+route_index*1000+year_index)
            model.fit(xfit,yfit)
            pcal=model.predict_proba(xcal)[:,1]
            ptest=model.predict_proba(xtest)[:,1]
            score_col=f"score_{route.lower()}"
            test=test.copy(); test[score_col]=ptest
            th=thresholds_from_cal(pcal)
            quality.append({
                "route":route,"test_year":test_year,
                "train_years":"|".join(map(str,sorted(map(int,train["year"].unique())))),
                "fit_rows":len(fit),"cal_rows":len(cal),"test_rows":len(test),
                "fit_races":fit_races,"cal_races":cal_races,"test_races":int(test["race_id"].nunique()),
                "positives_test":int(ytest.sum()),"positive_rate_test":float(ytest.mean()),
                "roc_auc":safe_auc(ytest,ptest),"pr_auc":safe_ap(ytest,ptest),
                "feature_count":int(xfit.shape[1]),
            })
            for cov in COVERAGE_TARGETS:
                r=eval_portfolio(test,route,cov,th[cov],score_col)
                r["test_year"]=test_year
                r["train_years"]="|".join(map(str,sorted(map(int,train["year"].unique()))))
                metrics.append(r)

    stable=stability(metrics)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"model-quality.csv",quality)
    write_csv(out/"coverage-metrics.csv",metrics)
    write_csv(out/"broad-stability.csv",stable)

    summary={
        "contract":"L2_K2_BROAD_ROLE_LAB_V2",
        "source_role_lab_run":36521825849,
        "routes":{
            "FIRST":"K2 novel fixed 1st in exacta, all other L1/L1.5 candidates as 2nd",
            "A12_THIRD":"A1/A2 occupy top2 in either order, K2 novel fixed 3rd (2 trifecta tickets)",
        },
        "walk_forward":{
            "2023":"train 2022",
            "2024":"train 2022-2023",
            "2025":"train 2022-2024",
        },
        "selection":"score thresholds come only from the prior-period chronological calibration split",
        "coverage_targets":[100,75,50,25],
        "anti_overfilter_guard":"No target below 25%; broad candidates require >=50% race coverage in every OOS year.",
        "probability_features_use_market":False,
        "market_used_for_diagnostics_only":True,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"All 2023-2025 metrics are walk-forward OOS research evidence. No production rule is promoted here.",
        "broad_stable_candidates":[r for r in stable if r["broad_stable_candidate"]==1],
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 K2 Broad Role Lab V2\n\n"
        "Learns two independent K2 role scores using only pre-race structural L1/L1.5 features: "
        "(1) K2 as exacta 1st, and (2) A1/A2 top-two with K2 3rd. "
        "Evaluation is walk-forward for 2023, 2024 and 2025. "
        "To prevent decorative over-filtering, only 100/75/50/25 coverage bands are evaluated, "
        "and any broad candidate must keep at least 50% race coverage in every OOS year. "
        "2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_K2_BROAD_ROLE_LAB_V2_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
