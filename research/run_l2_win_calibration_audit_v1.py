#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_l2_bet_kings_dataset_v1 import load_odds_day, decode_odds
from run_l2_bet_kings_arena_v1 import load_template
from run_l2_win_edge_audit_v1 import (
    WIN_TEMPLATES,
    EVAL_YEARS,
    selected_map,
    predict_template,
)

ALPHAS=(0.0,0.25,0.50,0.75,1.0)
EDGE_THRESHOLDS=(0.0,0.05,0.10,0.15,0.20)


def parse_args():
    p=argparse.ArgumentParser(description="Audit WIN P(hit) calibration versus normalized market probability.")
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--selections",required=True)
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
    with open(path,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def clip(p):
    return np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)


def brier(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    return float(np.mean((p-y)**2)) if len(y) else None


def logloss(y,p):
    y=np.asarray(y,dtype=float); p=clip(p)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))) if len(y) else None


def edge_bucket(x):
    x=float(x)
    if x < -0.50: return "LT_-50"
    if x < -0.25: return "-50_-25"
    if x < 0.0: return "-25_0"
    if x < 0.05: return "0_5"
    if x < 0.10: return "5_10"
    if x < 0.20: return "10_20"
    if x < 0.50: return "20_50"
    if x < 1.0: return "50_100"
    return "100_PLUS"


def odds_bucket(x):
    x=float(x)
    if x < 3: return "LT3"
    if x < 5: return "3_5"
    if x < 10: return "5_10"
    if x < 20: return "10_20"
    if x < 50: return "20_50"
    if x < 100: return "50_100"
    return "100_PLUS"


def prob_bucket(x):
    x=float(x)
    if x < 0.02: return "LT2"
    if x < 0.05: return "2_5"
    if x < 0.10: return "5_10"
    if x < 0.15: return "10_15"
    if x < 0.20: return "15_20"
    if x < 0.30: return "20_30"
    if x < 0.50: return "30_50"
    return "50_PLUS"


def calibration_row(items,year,dimension,bucket_name):
    n=len(items)
    wins=sum(int(r["hit"]) for r in items)
    model=np.array([r["model_p"] for r in items],dtype=float)
    market=np.array([r["market_p_norm"] for r in items],dtype=float)
    y=np.array([int(r["hit"]) for r in items],dtype=float)
    actual=float(y.mean()) if n else None
    return {
        "year":year,
        "dimension":dimension,
        "bucket":bucket_name,
        "rows":n,
        "wins":wins,
        "actual_win_rate_pct":100*actual if actual is not None else None,
        "mean_model_p_pct":100*float(model.mean()) if n else None,
        "model_calibration_gap_pp":100*(float(model.mean())-actual) if n else None,
        "model_brier":brier(y,model),
        "model_logloss":logloss(y,model),
        "mean_market_norm_p_pct":100*float(market.mean()) if n else None,
        "market_calibration_gap_pp":100*(float(market.mean())-actual) if n else None,
        "market_brier":brier(y,market),
        "market_logloss":logloss(y,market),
        "mean_raw_implied_pct":100*sum(r["market_p_raw"] for r in items)/n if n else None,
        "mean_market_overround":sum(r["market_overround"] for r in items)/n if n else None,
        "mean_edge_pct":100*sum(r["edge"] for r in items)/n if n else None,
        "mean_odds":sum(r["odds"] for r in items)/n if n else None,
    }


def load_market_context(backfill_root,wanted):
    by_date=defaultdict(set)
    for rid,date in wanted.items():
        by_date[date].add(rid)
    out={}
    odds_root=Path(backfill_root)/"data"/"odds"/"daily"
    for date,rids in sorted(by_date.items()):
        day=load_odds_day(odds_root/f"{date}.jsonl.gz",rids)
        missing=rids-set(day)
        if missing:
            raise SystemExit(f"missing odds records date={date}: {sorted(missing)[:5]}")
        for rid in rids:
            om=decode_odds(day[rid])
            win=[float(v) for (bt,_),v in om.items() if bt=="WIN" and v is not None and float(v)>0]
            if not win:
                raise SystemExit(f"no WIN odds race={rid}")
            over=sum(1.0/x for x in win)
            if not math.isfinite(over) or over<=0:
                raise SystemExit(f"invalid WIN overround race={rid}: {over}")
            out[rid]={"overround":over}
    return out


def main():
    a=parse_args()
    smap=selected_map(a.selections)
    wanted={
        rid:str(sel["race_date"])[:10]
        for (year,rid),sel in smap.items()
        if year in EVAL_YEARS
    }
    market_ctx=load_market_context(a.backfill_root,wanted)

    rows=[]
    manifest=json.loads((Path(a.dataset_dir)/"manifest.json").read_text(encoding="utf-8"))
    for template in sorted(WIN_TEMPLATES):
        info=manifest["templates"][template]
        df=load_template(Path(a.dataset_dir)/info["file"])
        preds=predict_template(df,template)
        for year in EVAL_YEARS:
            pred=preds.get(year)
            if pred is None:
                continue
            relevant={
                rid for (yy,rid),sel in smap.items()
                if yy==year and sel["template"]==template
            }
            if not relevant:
                continue
            psel=pred[pred["race_id"].astype(str).isin(relevant)].copy()
            for _,r in psel.iterrows():
                rid=str(r["race_id"])
                sel=smap[(year,rid)]
                odds=float(r["odds"])
                raw=1.0/odds
                over=float(market_ctx[rid]["overround"])
                norm=raw/over
                model=float(r["predicted_probability"])
                edge=float(r["edge"])
                rows.append({
                    "year":year,
                    "race_id":rid,
                    "race_date":str(r["race_date"]),
                    "template":template,
                    "selection_key":str(r["selection_key"]),
                    "hit":int(bool(r["hit"])),
                    "return_yen_per100":float(r["return_yen_per100"]),
                    "odds":odds,
                    "model_p":model,
                    "market_p_raw":raw,
                    "market_overround":over,
                    "market_p_norm":norm,
                    "edge":edge,
                    "selected_threshold":float(sel["edge_threshold"]),
                    "edge_bucket":edge_bucket(edge),
                    "odds_bucket":odds_bucket(odds),
                    "model_p_bucket":prob_bucket(model),
                    "market_p_bucket":prob_bucket(norm),
                    "gate_alert":int(float(r["gate_alert"])),
                    "is_novel":int(float(r["s1_is_novel"])),
                })

    if not rows:
        raise SystemExit("no calibration rows")

    calibration=[]
    for year in EVAL_YEARS:
        yy=[r for r in rows if r["year"]==year]
        calibration.append(calibration_row(yy,year,"ALL","ALL"))
        for dim,key in (
            ("EDGE","edge_bucket"),
            ("ODDS","odds_bucket"),
            ("MODEL_P","model_p_bucket"),
            ("MARKET_P","market_p_bucket"),
        ):
            vals=sorted(set(r[key] for r in yy))
            for val in vals:
                items=[r for r in yy if r[key]==val]
                calibration.append(calibration_row(items,year,dim,val))
        for gate in (0,1):
            items=[r for r in yy if r["gate_alert"]==gate]
            if items:
                calibration.append(calibration_row(items,year,"GATE",str(gate)))

    blend=[]
    for year in EVAL_YEARS:
        yy=[r for r in rows if r["year"]==year]
        for alpha in ALPHAS:
            probs=np.array([
                alpha*r["model_p"]+(1-alpha)*r["market_p_norm"]
                for r in yy
            ],dtype=float)
            y=np.array([r["hit"] for r in yy],dtype=float)
            blend.append({
                "year":year,
                "alpha_model":alpha,
                "alpha_market":1-alpha,
                "scope":"ALL_ROWS",
                "threshold":None,
                "rows":len(yy),
                "brier":brier(y,probs),
                "logloss":logloss(y,probs),
                "mean_probability_pct":100*float(probs.mean()),
                "actual_win_rate_pct":100*float(y.mean()),
            })
            for th in EDGE_THRESHOLDS:
                chosen=[]
                for r,p in zip(yy,probs):
                    e=float(p)*float(r["odds"])-1.0
                    if e>=th:
                        chosen.append((r,e))
                stake=100.0*len(chosen)
                ret=sum(r["return_yen_per100"] for r,_ in chosen)
                bought_races=len(set(r["race_id"] for r,_ in chosen))
                hit_races=len(set(r["race_id"] for r,_ in chosen if r["hit"]))
                blend.append({
                    "year":year,
                    "alpha_model":alpha,
                    "alpha_market":1-alpha,
                    "scope":"BET_EDGE",
                    "threshold":th,
                    "rows":len(chosen),
                    "bought_races":bought_races,
                    "race_hits":hit_races,
                    "stake_yen":stake,
                    "return_yen":ret,
                    "profit_yen":ret-stake,
                    "roi_pct":100*ret/stake if stake else None,
                })

    # Direct diagnostic on winner rows: is model systematically below market on actual winners?
    winner_diag=[]
    for year in EVAL_YEARS:
        wins=[r for r in rows if r["year"]==year and r["hit"]]
        for ob in sorted(set(r["odds_bucket"] for r in wins)):
            x=[r for r in wins if r["odds_bucket"]==ob]
            winner_diag.append({
                "year":year,
                "odds_bucket":ob,
                "winner_rows":len(x),
                "avg_model_p_pct":100*sum(r["model_p"] for r in x)/len(x),
                "avg_market_norm_p_pct":100*sum(r["market_p_norm"] for r in x)/len(x),
                "avg_model_minus_market_pp":100*sum(r["model_p"]-r["market_p_norm"] for r in x)/len(x),
                "avg_edge_pct":100*sum(r["edge"] for r in x)/len(x),
            })

    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"calibration.csv",calibration)
    write_csv(out/"blend-diagnostics.csv",blend)
    write_csv(out/"winner-market-comparison.csv",winner_diag)

    # compact year-level verdict ingredients
    yearly=[]
    for year in EVAL_YEARS:
        allrow=next(r for r in calibration if r["year"]==year and r["dimension"]=="ALL")
        yearly.append({
            "year":year,
            "rows":allrow["rows"],
            "wins":allrow["wins"],
            "actual_win_rate_pct":allrow["actual_win_rate_pct"],
            "mean_model_p_pct":allrow["mean_model_p_pct"],
            "mean_market_norm_p_pct":allrow["mean_market_norm_p_pct"],
            "model_brier":allrow["model_brier"],
            "market_brier":allrow["market_brier"],
            "model_logloss":allrow["model_logloss"],
            "market_logloss":allrow["market_logloss"],
        })

    summary={
        "contract":"L2_WIN_CALIBRATION_AUDIT_RESULT_V1",
        "source_router_run":36511322402,
        "source_win_edge_run":36519899646,
        "analysis_years":[2024,2025],
        "selected_win_races":len(wanted),
        "candidate_ticket_rows":len(rows),
        "phit_market_features":False,
        "market_probability_normalization":"(1/odds) / sum_all_field(1/odds)",
        "edge_formula":"predicted_probability * decimal_odds - 1",
        "blend_grid":[0.0,0.25,0.5,0.75,1.0],
        "yearly":yearly,
        "2025_is_holdout":False,
        "2026_locked":True,
        "production_promotion":False,
        "interpretation_guard":"Calibration/blend results are development diagnostics on already-observed 2024-2025; they do not authorize a production rule."
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 WIN Calibration Audit V1\n\n"
        "Compares odds-free model P(hit), observed win rate, raw break-even probability 1/odds, "
        "and normalized full-field market probability. Also tests diagnostic model/market blends. "
        "2025 is development evidence, not a fresh holdout. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_WIN_CALIBRATION_AUDIT_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
