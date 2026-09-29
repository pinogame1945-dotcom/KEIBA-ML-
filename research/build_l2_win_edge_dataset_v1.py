#!/usr/bin/env python3
import argparse,gzip,json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    YEARS,EXPECTED_PER_YEAR,EXPECTED_TOTAL,EXPECTED_ALERTS,
    load_fixed_ledgers,load_router,seven_stats,load_day,load_odds_day,
    horse_number_map,decode_odds,payout_map,horse_feature,
    finite,mean
)

WIN_TEMPLATES=("WIN_ANCHOR1","WIN_ANCHORS2","WIN_ALL_CANDIDATES")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()

def parse_paths(items):
    out={}
    for x in items:
        y,p=x.split(":",1); out[int(y)]=p
    return out

def main():
    a=parse_args(); ps=parse_paths(a.router_year)
    if set(ps)!=set(YEARS): raise SystemExit(f"router years mismatch {sorted(ps)}")
    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={}; race_owner={}; dates=defaultdict(list)
    total=0
    for y in YEARS:
        routers[y]=load_router(ps[y],y)
        for rid,row in routers[y].items():
            d=str(row.get("race_date") or "")[:10]
            race_owner[rid]=y; dates[d].append(rid); total+=1
    if total!=EXPECTED_TOTAL: raise SystemExit(f"race total {total}")

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=defaultdict(int); hits=defaultdict(int); races=defaultdict(set)
    with ExitStack() as stack:
        handles={t:stack.enter_context(gzip.open(out/f"{t}.jsonl.gz","wt",encoding="utf-8")) for t in WIN_TEMPLATES}
        backfill=Path(a.backfill_root)
        for di,date in enumerate(sorted(dates),1):
            wanted=set(dates[date])
            day=load_day(backfill/"data"/"daily"/f"{date}.jsonl.gz",wanted)
            odds_day=load_odds_day(backfill/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
            for rid in sorted(wanted):
                y=race_owner[rid]; router=routers[y][rid]; pack=day.get(rid); oddsrec=odds_day.get(rid)
                if pack is None or oddsrec is None: raise SystemExit(f"missing market pack {rid}")
                seven,seven_map=seven_stats(router); fr=fixed[y].get(rid)
                if fr:
                    candidates=[str(x) for x in fr["candidate_horse_ids"]]
                    novel=[str(x) for x in fr["novel_horse_ids"]]
                    anchors=[str(x) for x in fr["seven_anchor_horse_ids"]]
                    gate=True; gate_score=finite(fr.get("gate_score")) or 0.0
                    selected_outsiders=list(fr.get("selected_outsiders") or [])
                else:
                    candidates=list(seven); novel=[]; anchors=seven[:2]; gate=False; gate_score=0.0; selected_outsiders=[]
                if len(anchors)<2: continue
                a1,a2=anchors; horse_no=horse_number_map(pack)
                if any(h not in horse_no for h in candidates): raise SystemExit(f"horse number missing {rid}")
                odds_map=decode_odds(oddsrec); payouts,present=payout_map(pack)
                if "WIN" not in present: continue
                # Fast path: this audit needs only the three WIN templates.
                # Do not build QUINELLA/EXACTA/TRIO/TRIFECTA combinations.
                templates={
                    "WIN_ANCHOR1":[(a1,)],
                    "WIN_ANCHORS2":[(a1,),(a2,)],
                    "WIN_ALL_CANDIDATES":[(x,) for x in candidates],
                }
                seven_pos={h:i+1 for i,h in enumerate(seven)}
                cand_pos={h:i+1 for i,h in enumerate(candidates)}
                novel_pos={h:i+1 for i,h in enumerate(novel)}
                hfeat={h:horse_feature(h,seven_map,seven_pos,cand_pos,novel_pos,a1,a2) for h in candidates}
                meta=router.get("race") or {}; cons=router.get("consensus") or {}
                outs1=selected_outsiders[0] if len(selected_outsiders)>0 else "__NONE__"
                outs2=selected_outsiders[1] if len(selected_outsiders)>1 else "__NONE__"
                for template in WIN_TEMPLATES:
                    for hids in templates[template]:
                        hid=hids[0]; no=horse_no[hid]; key=("WIN",(no,))
                        odd=odds_map.get(key)
                        if odd is None: continue
                        ret=float(payouts.get(key,0.0)); hit=ret>0; f=hfeat[hid]
                        row={
                            "contract":"L2_WIN_EDGE_TICKET_V1","year":y,"race_id":rid,"race_date":date,
                            "bet_type":"WIN","template":template,"selection_key":str(no),
                            "selection_numbers":str(no),"selection_horse_ids":hid,
                            "hit":hit,"return_yen_per100":ret,"odds":odd,
                            "gate_alert":1 if gate else 0,"gate_score_alert_only":gate_score,
                            "selected_outsider_1":outs1,"selected_outsider_2":outs2,
                            "candidate_pool_size":len(candidates),"seven_union_count":len(seven),
                            "novel_pool_count":len(novel),"ticket_novel_count":int(f["is_novel"]),
                            "ticket_anchor_count":int(f["is_anchor1"] or f["is_anchor2"]),
                            "king_support_sum":f["king_support"],"king_support_min":f["king_support"],"king_support_max":f["king_support"],
                            "king_borda_sum":f["king_borda"],"king_borda_mean":f["king_borda"],
                            "king_top1_votes_sum":f["king_top1_votes"],
                            "venue_code":meta.get("venue_code"),"surface":meta.get("surface"),
                            "race_class":meta.get("race_class"),"discipline":meta.get("discipline"),
                            "direction":meta.get("direction"),"weather":meta.get("weather"),
                            "track_condition":meta.get("track_condition"),"distance_m":finite(meta.get("distance_m")) or 0.0,
                            "field_size":finite(meta.get("field_size")) or 0.0,
                            "cw_top1_max_vote_share":finite(cons.get("top1_max_vote_share")) or 0.0,
                            "cw_top3_jaccard":finite(cons.get("top3_pairwise_jaccard_mean")) or 0.0,
                            "cw_top6_jaccard":finite(cons.get("top6_pairwise_jaccard_mean")) or 0.0,
                            "cw_rank_diff_mean":finite(cons.get("pairwise_rank_abs_diff_mean")) or 0.0,
                            "cw_rank_std_mean":finite(cons.get("horse_rank_std_mean")) or 0.0,
                            "cw_prob_std_mean":finite(cons.get("horse_probability_std_mean")) or 0.0,
                            "cw_prob_std_max":finite(cons.get("horse_probability_std_max")) or 0.0,
                        }
                        for name in ("king_support","king_borda","king_best_rank","king_mean_rank","king_top1_votes","consensus_position","candidate_position","novel_position","is_novel","is_anchor1","is_anchor2"):
                            row["s1_"+name]=float(f.get(name,0.0))
                            row["s2_"+name]=0.0; row["s3_"+name]=0.0
                        handles[template].write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
                        rows[template]+=1; hits[template]+=int(hit); races[template].add(rid)
            if di%50==0: print(f"WIN_DATASET_PROGRESS {di}/{len(dates)}",flush=True)

    manifest={
      "contract":"L2_WIN_EDGE_DATASET_V1","source_l15":"L15_FIXED_V1","years":list(YEARS),
      "races":EXPECTED_TOTAL,"gate_alerts":EXPECTED_ALERTS,"probability_model_uses_odds":False,
      "market_price_stage":"FINAL_ODDS_AFTER_PREDICTION","locked_years":[2026],"templates":{}
    }
    for t in WIN_TEMPLATES:
        manifest["templates"][t]={"bet_type":"WIN","priced_rows":rows[t],"hits":hits[t],"races":len(races[t]),"file":f"{t}.jsonl.gz"}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L2_WIN_EDGE_DATASET_V1_READY",flush=True)

if __name__=="__main__": main()
