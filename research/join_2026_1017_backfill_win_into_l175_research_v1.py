#!/usr/bin/env python3
"""Read-only 2026-10-17 frozen L1.75 research market join from exact BACKFILL handoff.

No labels, FINAL odds, interpolated prices, 2026 learning/evaluation, original King
rank change, ticket pricing, or automated wagering. Source declares as-of independent
verification MISSING; this adapter never invents that approval.
"""
from __future__ import annotations
import argparse,datetime,gzip,hashlib,json,math
from collections import defaultdict
from pathlib import Path

INPUT_CONTRACT="KEIBA_2026_1017_FROZEN_L175_RESEARCH_PRECURSOR_SAVE_V1"
FEATURE_INDEX="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"
ODDS_INDEX="KEIBA_BACKFILL_TO_ML_ODDS_INDEX_V1"
OUTPUT="KEIBA_2026_1017_L175_MARKET_JOIN_RESEARCH_V1"
DAY="2026-10-17"
TYPES={"WIN","PLACE","BRACKET_QUINELLA","QUINELLA","WIDE","EXACTA","TRIO","TRIFECTA"}
BLOCK="PRE_RACE_OBSERVED_ASOF_UNVERIFIED"

def need(x,msg):
    if not x:raise ValueError(msg)

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def read_rows(p):
    op=gzip.open if str(p).endswith(".gz") else open
    with op(p,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip():yield json.loads(line)

def exact_path(root,rel):
    need(isinstance(rel,str) and rel.startswith("ml_features/delivery/") and
         ".." not in rel.split("/") and "\\" not in rel and
         "//" not in rel,"BACKFILL_HANDOFF_PATH_INVALID")
    return root/rel

def stamp(raw):
    need(isinstance(raw,str) and raw,"CAPTURE_TIME_MISSING")
    try:v=datetime.datetime.fromisoformat(raw.replace("Z","+00:00"))
    except ValueError as e:raise ValueError("INVALID_CAPTURE_TIMESTAMP") from e
    need(v.tzinfo is not None and v.utcoffset() is not None,
         "CAPTURE_TIMESTAMP_NOT_TIMEZONE_AWARE")
    return v.astimezone(datetime.timezone.utc)

def competition_rank(price_by_horse):
    # Fixed L175 MARKET CONTEXT V1: equal WIN odds share rank (1, 2, 2, 4).
    sorted_prices=sorted(price_by_horse.values())
    return {no:1+sum(v<p for v in sorted_prices)
            for no,p in price_by_horse.items()}

def compute(input_index,input_file,handoff,out):
    need(not out.exists(),"NO_RESEARCH_MARKET_OUTPUT_OVERWRITE")
    idx=load(input_index)
    need(idx.get("contract")==INPUT_CONTRACT and idx.get("races")==36 and
         idx.get("horses")==471 and idx.get("model_training_years")==[2021,2022,2023,2024,2025] and
         idx.get("ready_for_betting") is False and idx.get("asof_independently_verified") is False and
         idx.get("market_readiness")=="BLOCKED_NO_AUTHENTIC_PRE_RACE_WIN_ODDS" and
         idx.get("files",{}).get(input_file.name)==digest(input_file),
         "L175_FROZEN_P3_SOURCE_PROVENANCE_MISMATCH")
    p3=list(read_rows(input_file))
    by_race={str(x.get("race_id") or ""):x for x in p3}
    need(len(p3)==len(by_race)==36 and
         sum(len(x.get("horses") or []) for x in p3)==471 and
         all(x.get("prediction_year")==2026 and
             x.get("ranking_policy")=="SEVEN_KING_UNCHANGED" and
             x.get("ready_for_betting") is False and
             x.get("2026_target_results_used") is False and
             x.get("2026_asof_independently_verified") is False for x in p3),
         "FROZEN_PRECURSOR_36R_471H_RANKS_OR_SEAL_DRIFT")
    fi=load(handoff/"ml_features/delivery/690/index.json")
    oi=load(handoff/"ml_features/delivery/odds/index.json")
    need(fi.get("contract")==FEATURE_INDEX and oi.get("contract")==ODDS_INDEX and
         oi.get("total_races")==60 and
         oi.get("feature_index_path")=="ml_features/delivery/690/index.json",
         "BACKFILL_FEATURE_ODDS_INDEX_DRIFT")
    all_odds=oi["races"]; all_features=fi["races"]
    selected={rid for rid,v in all_odds.items() if v.get("date")==DAY}
    need(selected==set(by_race) and len(selected)==36 and
         all(rid in all_features and all_features[rid]["date"]==DAY for rid in selected),
         "BACKFILL_OCT17_36R_COVERAGE_MISMATCH")
    result=[]; metrics=defaultdict(int)
    for rid in sorted(selected):
        fr=all_features[rid];odef=all_odds[rid]
        need(odef.get("snapshot_type")=="PRE_RACE_LIVE_ODDS" and
             odef.get("source")=="JRA_OFFICIAL_POST" and
             odef.get("readiness")==BLOCK and
             odef.get("feature_sha256")==fr["sha256"],"NO_AUTHENTIC_LIVE_ODDS_LINK:"+rid)
        feature_file=exact_path(handoff,fr["path"])
        odd_file=exact_path(handoff,odef["path"])
        need(feature_file.is_file() and odd_file.is_file(),"MISSING_ORIGINAL_BACKFILL_DELIVERY_FILE:"+rid)
        need(digest(feature_file)==fr["sha256"] and
             digest(odd_file)==odef["sha256"],"BACKFILL_DELIVERY_SHA256_DRIFT:"+rid)
        f=load(feature_file);o=load(odd_file)
        need(f.get("contract")=="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1" and
             f.get("race_id")==rid and f.get("race_date")==DAY and
             f.get("feature_count")==690 and
             f.get("used_target_results_or_final_odds") is False and
             f.get("source_capture_asof_independently_verified") is False and
             o.get("contract")=="KEIBA_BACKFILL_TO_ML_ODDS_V1" and
             o.get("race_id")==rid and o.get("race_date")==DAY and
             o.get("feature_reference")==fr["path"] and
             o.get("feature_sha256")==fr["sha256"] and
             o.get("original_is_unmodified_source") is True,
             "BACKFILL_ODDS_FEATURE_PROVENANCE_DRIFT:"+rid)
        feats=f.get("rows") or []; source=by_race[rid];horses=source["horses"]
        need(len(feats)==len(horses)==f["starters"]==source["horse_count"] and
             [h["king_rank"] for h in horses]==list(range(1,len(horses)+1)),
             "L175_FROZEN_KING_POPULATION_DRIFT:"+rid)
        identities={}
        for row in feats:
            hid=str(row["_horse_id"]);num=row["horse_number"]
            need(row["_race_id"]==rid and row["_race_date"]==DAY and
                 hid not in identities and type(num) is int and 1<=num<=24,
                 "BACKFILL_690_HORSE_ID_OR_NUMBER_DRIFT:"+rid)
            identities[hid]=num
        need({h["horse_id"] for h in horses}==set(identities) and
             len(set(identities.values()))==len(identities),"REAL_L175_HORSE_POPULATION_MISMATCH:"+rid)
        odds=o["odds"]
        need(odds.get("source")=="JRA_OFFICIAL_POST" and
             odds.get("snapshot_type")=="PRE_RACE_LIVE_ODDS" and
             odds.get("readiness")==BLOCK and
             odds.get("publisher_update_display_consistent") is True and
             odds.get("publisher_update_day_not_inferred") is True and
             set(odds.get("by_bet_type") or {})==TYPES and
             odds.get("expected_combination_count")==sum(odds.get(key,-100)
                   for key in ("quoted_combination_count","no_votes_combination_count","scratched_combination_count")),
             "ORIGINAL_JRA_OFFICIAL_PRE_RACE_MARKET_PROVENANCE_DRIFT:"+rid)
        captured=stamp(odds["source_captured_at_utc"])
        need(captured.date()<=datetime.date(2026,10,17) and
             captured>=datetime.datetime(2026,10,16,tzinfo=datetime.timezone.utc),
             "OUTSIDE_ALLOWED_RACE_DAY_CAPTURE_WINDOW:"+rid)
        summary=odds["by_bet_type"]["WIN"]
        # JRA includes scratched horses in the full WIN table, whereas
        # frozen 690/7King correctly exclude them. Preserve that distinction.
        need(summary.get("status")=="COLLECTED" and
             type(summary.get("expected")) is int and
             summary.get("expected")==summary.get("listed") and
             summary.get("expected")>=len(horses),"OFFICIAL_JRA_WIN_FIELD_INCOMPLETE:"+rid)
        win={}
        for quote in odds.get("quotes") or []:
            if quote.get("bet_type")!="WIN":continue
            nums=quote.get("combination")
            v=quote.get("odds")
            need(isinstance(nums,list) and len(nums)==1 and
                 type(nums[0]) is int and nums[0] in identities.values() and
                 type(v) in (int,float) and math.isfinite(v) and v>0 and nums[0] not in win,
                 "INVALID_WIN_MARKET_QUOTE:"+rid)
            win[nums[0]]=float(v)
        other_win=[x for key in ("no_votes","scratched_combinations")
                   for x in (odds.get(key) or []) if x.get("bet_type")=="WIN"]
        no_vote_set={int(x["combination"][0]) for x in (odds.get("no_votes") or []) if x.get("bet_type")=="WIN"}
        scratched_set={int(x["combination"][0]) for x in (odds.get("scratched_combinations") or []) if x.get("bet_type")=="WIN"}
        active=set(identities.values())
        need(len(win)==summary.get("quoted") and
             len(other_win)==summary.get("no_votes",0)+summary.get("scratched",0) and
             len(win)+len(other_win)==summary["expected"] and
             not (set(win)&(no_vote_set|scratched_set)) and
             not (no_vote_set&scratched_set) and
             not (active&scratched_set) and
             active.issubset(set(win)|no_vote_set) and
             len(set(win)|no_vote_set|scratched_set)==summary["expected"],
             "WIN_QUOTE_OR_NO_VOTE_POPULATION_MISMATCH:"+rid)
        full=set(win)==active
        ranks=competition_rank(win) if full else {}
        updated=[]
        for h in horses:
            no=identities[h["horse_id"]]
            n=dict(h)
            if no in ranks:
                market_rank=ranks[no]; gap=market_rank-n["king_rank"]
                delta=n.get("p3_delta")
                align=0 if gap==0 or delta is None else (1 if gap>0 else -1)*delta
                n.update({"horse_number":no,"market_win_odds":win[no],
                  "market_rank":market_rank,"signed_rank_gap":gap,"abs_rank_gap":abs(gap),
                  "dissent_direction":"UP" if gap>=2 else "DOWN" if gap<=-2 else "NEAR",
                  "outsider_alignment_score":align if delta is not None else None,
                  "outsider_alignment":("AGREE" if align>0 else
                     "OPPOSE" if align<0 else "NEUTRAL") if delta is not None else "NA"})
                metrics["market_horses"]+=1
            else:
                # If WIN field has NO_VOTES, never fabricate odds / partial competition ranks.
                n.update({"horse_number":no,"market_win_odds":win.get(no),
                  "market_rank":None,"signed_rank_gap":None,"abs_rank_gap":None,
                  "dissent_direction":None,"outsider_alignment_score":None,
                  "outsider_alignment":"NA"})
            need(n["king_rank"]==h["king_rank"] and
                 not any(k in n for k in ("reranked_rank","blended_rank","buy_flag","skip_flag","stake")),
                 "FROZEN_L175_RANK_OR_OUTPUT_DRIFT")
            updated.append(n)
        result.append({**{k:v for k,v in source.items() if k!="horses"},
          "contract":OUTPUT,"horse_count":len(updated),"horses":updated,
          "market_readiness":"RESEARCH_ONLY_INDEPENDENT_ASOF_NOT_VERIFIED",
          "win_market_source":"JRA_OFFICIAL_POST","win_market_snapshot_type":"PRE_RACE_LIVE_ODDS",
          "win_market_readiness":BLOCK,"win_odds_observed_utc":odds["source_captured_at_utc"],
          "win_quote_full_field":full,"market_field_complete":full,
          "win_quote_count":len(win),"no_vote_or_scratched_win_count":len(other_win),
          "market_source_sha256":odef["sha256"],
          "feature_source_sha256":fr["sha256"],
          "market_source_backfill_branch":"handoff/690-feature-data-v1",
          "market_asof_independently_verified":False,"ready_for_betting":False,
          "l21_l3_invoked":False})
        metrics["races"]+=1;metrics["horses"]+=len(horses)
        metrics["complete_win_races"]+=int(full)
        metrics["missing_win_quote_horses"]+=len(horses)-len(win)
        metrics["scratched_jra_win_horses_not_in_frozen_active_field"]+=len(scratched_set)
    need(metrics["races"]==36 and metrics["horses"]==471 and
         all(x["ranking_policy"]=="SEVEN_KING_UNCHANGED" and
             x["ready_for_betting"] is False and
             x["market_asof_independently_verified"] is False for x in result),
         "FINAL_REAL_MARKET_RESEARCH_LINEAGE_FAIL")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as stream:
        for row in result:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
    print("REAL_2026_1017_FROZEN_L175_BACKFILL_WIN_RESEARCH_MARKET_JOIN_PASS "+
          json.dumps(dict(metrics),sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    for n in ("input-index","input-file","backfill-handoff","output"):
        p.add_argument("--"+n,type=Path,required=True)
    z=p.parse_args();compute(z.input_index,z.input_file,z.backfill_handoff,z.output)
if __name__=="__main__":main()
