#!/usr/bin/env python3
import argparse,csv,gzip,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from run_l2_bet_kings_arena_v1 import (
    TEST_YEARS,load_template,feature_columns,chronological_split,encode_fit_other,
    make_model,calibrate
)

WIN_TEMPLATES=("WIN_ALL_CANDIDATES","WIN_ANCHOR1","WIN_ANCHORS2")
EVAL_YEARS=(2024,2025)
CF_THRESHOLDS=(-0.25,0.0,0.05,0.10,0.15,0.20)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--selections",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def read_csv(p):
    with open(p,newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def write_csv(p,rows):
    if not rows:
        Path(p).write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def bucket(odds):
    x=float(odds)
    if x<3:return "LT3"
    if x<5:return "3_5"
    if x<10:return "5_10"
    if x<20:return "10_20"
    if x<50:return "20_50"
    if x<100:return "50_100"
    return "100_PLUS"

def selected_map(path):
    out={}
    for r in read_csv(path):
        if r.get("router")!="DIRECT_BET_TYPE" or r.get("bet_type")!="WIN": continue
        y=int(float(r["test_year"]))
        if y not in EVAL_YEARS: continue
        out[(y,r["race_id"])]=r
    return out

def predict_template(df,template_index):
    preds={}
    df["hit"]=df["hit"].astype(bool)
    df["odds"]=pd.to_numeric(df["odds"],errors="coerce")
    df["return_yen_per100"]=pd.to_numeric(df["return_yen_per100"],errors="coerce").fillna(0.0)
    df["race_month"]=pd.to_numeric(df["race_date"].astype(str).str.slice(5,7),errors="coerce").fillna(0.0)
    cols=feature_columns(df)
    for year_index,test_year in enumerate(TEST_YEARS):
        train=df[df["year"]<test_year].copy(); test=df[df["year"]==test_year].copy()
        if train.empty or test.empty: continue
        split=chronological_split(train)
        if split is None: continue
        fit,cal,_,_=split
        yfit=fit["hit"].astype(int).to_numpy(); ycal=cal["hit"].astype(int).to_numpy()
        if len(set(yfit.tolist()))<2: continue
        xfit,(xcal,xtest)=encode_fit_other(fit,[cal,test],cols)
        # exact seed parity with original WIN arena: bet_index=0
        model=make_model(71000+template_index*100+year_index)
        model.fit(xfit,yfit)
        pcal=model.predict_proba(xcal)[:,1]
        ptest0=model.predict_proba(xtest)[:,1]
        ptest,_=calibrate(ycal,pcal,ptest0)
        pred=test.copy()
        pred["predicted_probability"]=ptest
        pred["edge"]=pred["predicted_probability"]*pred["odds"]-1.0
        preds[test_year]=pred
    return preds

def summarize_counter(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r["year"],r["method"])].append(r)
    out=[]
    for (year,method),items in sorted(groups.items()):
        stake=sum(r["stake"] for r in items); ret=sum(r["ret"] for r in items)
        out.append({
            "year":year,"method":method,"races":len(items),
            "bought_races":sum(r["stake"]>0 for r in items),
            "tickets":sum(r["tickets"] for r in items),
            "race_hits":sum(r["hit"] for r in items),
            "hit_rate_pct":100*sum(r["hit"] for r in items)/len(items) if items else None,
            "stake_yen":stake,"return_yen":ret,"profit_yen":ret-stake,
            "roi_pct":100*ret/stake if stake else None,
        })
    return out

def main():
    a=args()
    smap=selected_map(a.selections)
    manifest=json.loads((Path(a.dataset_dir)/"manifest.json").read_text(encoding="utf-8"))
    details=[]; counter=[]
    # keep original template_index ordering exactly as arena: sorted WIN templates
    for ti,template in enumerate(sorted(WIN_TEMPLATES)):
        info=manifest["templates"][template]
        df=load_template(Path(a.dataset_dir)/info["file"])
        pred_by_year=predict_template(df,ti)
        for y in EVAL_YEARS:
            pred=pred_by_year.get(y)
            if pred is None: continue
            relevant=[rid for (yy,rid),s in smap.items() if yy==y and s["template"]==template]
            if not relevant: continue
            psel=pred[pred["race_id"].astype(str).isin(set(relevant))].copy()
            for rid,grp in psel.groupby(psel["race_id"].astype(str),sort=False):
                sel=smap[(y,rid)]
                threshold=float(sel["edge_threshold"])
                g=grp.sort_values("selection_key").copy()
                purchased=g[g["edge"]>=threshold].copy()
                # exact reconstruction guard
                got_tickets=len(purchased)
                got_ret=float(purchased["return_yen_per100"].sum())
                exp_tickets=int(float(sel["ticket_count"]))
                exp_ret=float(sel["actual_return_yen"])
                if got_tickets!=exp_tickets or abs(got_ret-exp_ret)>1e-6:
                    raise SystemExit(
                        f"WIN reconstruction drift y={y} race={rid} template={template} "
                        f"tickets {got_tickets}!={exp_tickets} ret {got_ret}!={exp_ret}"
                    )
                winners=g[g["hit"]==True].copy()
                if winners.empty:
                    reason="NO_RAW_WIN"
                    wp=wo=we=wi=None; wrank=None
                else:
                    w=winners.sort_values("predicted_probability",ascending=False).iloc[0]
                    wp=float(w["predicted_probability"]); wo=float(w["odds"]); we=float(w["edge"]); wi=1.0/wo
                    wrank=int((g["predicted_probability"]>wp).sum()+1)
                    if we<0:
                        reason="WINNER_NEGATIVE_EDGE"
                    elif we<threshold:
                        reason="WINNER_POSITIVE_BELOW_SELECTED_THRESHOLD"
                    else:
                        reason="WINNER_PASSED_EDGE"
                details.append({
                    "year":y,"race_id":rid,"template":template,"selected_threshold":threshold,
                    "selected_ticket_count":got_tickets,"selected_return_yen":got_ret,
                    "selected_hit":int(got_ret>0),"drop_reason":reason,
                    "winner_predicted_probability":wp,"winner_odds":wo,"winner_edge":we,
                    "winner_implied_probability":wi,
                    "winner_p_minus_implied_pp":(100*(wp-wi) if wp is not None else None),
                    "winner_predicted_rank_in_template":wrank,
                    "winner_odds_bucket":(bucket(wo) if wo is not None else None),
                    "gate_alert":int(float(g.iloc[0]["gate_alert"])),
                    "winner_is_novel":(int(float(w["s1_is_novel"])) if not winners.empty else None),
                })
                # counterfactual fixed edge thresholds
                for th in CF_THRESHOLDS:
                    sub=g[g["edge"]>=th]
                    counter.append({
                        "year":y,"race_id":rid,"method":f"EDGE_GE_{th:.2f}",
                        "tickets":len(sub),"stake":100.0*len(sub),
                        "ret":float(sub["return_yen_per100"].sum()),
                        "hit":int((sub["return_yen_per100"]>0).any()),
                    })
                # raw template all
                counter.append({
                    "year":y,"race_id":rid,"method":"RAW_TEMPLATE_ALL",
                    "tickets":len(g),"stake":100.0*len(g),"ret":float(g["return_yen_per100"].sum()),
                    "hit":int((g["return_yen_per100"]>0).any()),
                })
                # top predicted probability 1
                gp=g.sort_values(["predicted_probability","selection_key"],ascending=[False,True]).head(1)
                counter.append({
                    "year":y,"race_id":rid,"method":"TOP1_P",
                    "tickets":1,"stake":100.0,"ret":float(gp["return_yen_per100"].sum()),
                    "hit":int((gp["return_yen_per100"]>0).any()),
                })
                ge=g.sort_values(["edge","selection_key"],ascending=[False,True]).head(1)
                counter.append({
                    "year":y,"race_id":rid,"method":"TOP1_EDGE",
                    "tickets":1,"stake":100.0,"ret":float(ge["return_yen_per100"].sum()),
                    "hit":int((ge["return_yen_per100"]>0).any()),
                })

    # aggregate reasons
    reason_rows=[]
    for y in EVAL_YEARS:
        yy=[r for r in details if r["year"]==y]
        for reason in sorted(set(r["drop_reason"] for r in yy)):
            x=[r for r in yy if r["drop_reason"]==reason]
            vals=[r for r in x if r["winner_edge"] is not None]
            reason_rows.append({
                "year":y,"reason":reason,"races":len(x),"share_pct":100*len(x)/len(yy) if yy else None,
                "avg_winner_edge":(sum(r["winner_edge"] for r in vals)/len(vals) if vals else None),
                "avg_winner_odds":(sum(r["winner_odds"] for r in vals)/len(vals) if vals else None),
                "avg_winner_p_minus_implied_pp":(sum(r["winner_p_minus_implied_pp"] for r in vals)/len(vals) if vals else None),
            })

    bucket_rows=[]
    for y in EVAL_YEARS:
        yy=[r for r in details if r["year"]==y and r["winner_odds_bucket"]]
        for b in ("LT3","3_5","5_10","10_20","20_50","50_100","100_PLUS"):
            x=[r for r in yy if r["winner_odds_bucket"]==b]
            if not x: continue
            bucket_rows.append({
                "year":y,"winner_odds_bucket":b,"raw_winner_races":len(x),
                "winner_passed_edge":sum(r["drop_reason"]=="WINNER_PASSED_EDGE" for r in x),
                "negative_edge_winners":sum(r["drop_reason"]=="WINNER_NEGATIVE_EDGE" for r in x),
                "positive_below_threshold_winners":sum(r["drop_reason"]=="WINNER_POSITIVE_BELOW_SELECTED_THRESHOLD" for r in x),
                "avg_winner_edge":sum(r["winner_edge"] for r in x)/len(x),
                "avg_winner_predicted_rank":sum(r["winner_predicted_rank_in_template"] for r in x)/len(x),
            })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"winner-drop-reasons.csv",reason_rows)
    write_csv(out/"winner-odds-buckets.csv",bucket_rows)
    write_csv(out/"counterfactuals.csv",summarize_counter(counter))
    with gzip.open(out/"winner-details.csv.gz","wt",newline="",encoding="utf-8") as f:
        if details:
            w=csv.DictWriter(f,fieldnames=list(details[0])); w.writeheader(); w.writerows(details)
    summary={
        "contract":"L2_WIN_EDGE_AUDIT_RESULT_V1",
        "source_router_run":36511322402,
        "analysis_years":[2024,2025],
        "selected_win_races":len(details),
        "reconstructed_selected_actions_exactly":True,
        "phit_retrained_for_reconstruction":True,
        "phit_market_features":False,
        "2026_locked":True,
        "production_promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text("# L2 WIN Edge Audit V1\n\nReconstructs the original odds-free WIN P(hit) models and the exact DIRECT-selected WIN actions, then diagnoses which winning horses were discarded by negative edge versus the selected positive edge threshold. 2026 remains sealed.\n",encoding="utf-8")
    print("L2_WIN_EDGE_AUDIT_V1_READY",flush=True)

if __name__=="__main__": main()
