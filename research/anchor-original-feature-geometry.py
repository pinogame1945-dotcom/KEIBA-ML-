#!/usr/bin/env python3
"""Pure original ANCHOR_FULL feature geometry; no race targets or outcome data.

Input requirements: distinct numbered starters with ORIGINAL prior calculated
AI distill rank/z/gap/risk and one same-time WIN price for each starter.
Market ties retain competition/min-rank matching original market rank.
This module does not assert input publication provenance; the caller must
verify original companion/model and JRA quote timestamps independently.
"""
from __future__ import annotations
import math

FIELDS=("field_size","distill_ai_rank01","distill_score_z","distill_score_gap_to_best",
    "distill_is_ai_top1","distill_is_ai_top3","distill_is_ai_top6","risk",
    "rank_gap_norm","abs_rank_gap_norm","top3_overlap_norm","ai_unique_top3",
    "market_unique_top3","top1_match","risk_x_abs_gap","ability_x_market_gap",
    "ability_x_disagree_level","risk_x_disagree_level","rankgap_x_disagree_level")

def build_geometry(companions,win_price):
    if not isinstance(companions,dict) or not isinstance(win_price,dict):
        raise ValueError("ORIGINAL_GEOMETRY_MAPS_REQUIRED")
    field=set(companions)
    if len(field)<3 or len(field)>24 or field!=set(win_price):
        raise ValueError("ORIGINAL_GEOMETRY_FULL_STARTER_FIELD_REQUIRED")
    n=len(field)
    seen_ids=set()
    ai_ranks=set()
    for no,item in companions.items():
        if type(no)is not int or not 1<=no<=24:
            raise ValueError("ORIGINAL_GEOMETRY_HORSE_NUMBER")
        hid=item.get("horse_id")
        if not isinstance(hid,str) or not hid or hid in seen_ids:
            raise ValueError("ORIGINAL_GEOMETRY_HORSE_ID")
        seen_ids.add(hid)
        ai=item.get("distill_ai_rank")
        if type(ai)is not int or not 1<=ai<=n or ai in ai_ranks:
            raise ValueError("ORIGINAL_GEOMETRY_AI_RANK_NONUNIQUE")
        ai_ranks.add(ai)
        for key in ("distill_score_z","distill_score_gap_to_best","risk"):
            value=item.get(key)
            if isinstance(value,bool):
                raise ValueError("ORIGINAL_GEOMETRY_NONFINITE_"+key)
            try:x=float(value)
            except (TypeError,ValueError) as ex:
                raise ValueError("ORIGINAL_GEOMETRY_NONFINITE_"+key) from ex
            if not math.isfinite(x):raise ValueError("ORIGINAL_GEOMETRY_NONFINITE_"+key)
            if key=="risk" and not 0<=x<=1:raise ValueError("ORIGINAL_GEOMETRY_RISK_OUT_OF_RANGE")
            if key=="distill_score_gap_to_best" and x<0:raise ValueError("ORIGINAL_GEOMETRY_NEGATIVE_SCORE_GAP")
        price=win_price[no]
        if isinstance(price,bool):
            raise ValueError("ORIGINAL_GEOMETRY_INVALID_WIN_PRICE")
        try:v=float(price)
        except (TypeError,ValueError) as ex:raise ValueError("ORIGINAL_GEOMETRY_INVALID_WIN_PRICE") from ex
        if not math.isfinite(v) or v<1 or v>9999.9:
            raise ValueError("ORIGINAL_GEOMETRY_INVALID_WIN_PRICE")
    if ai_ranks!=set(range(1,n+1)):
        raise ValueError("ORIGINAL_GEOMETRY_AI_RANK_GAP")
    order=sorted(field,key=lambda no:(float(win_price[no]),companions[no]["horse_id"]))
    market_rank={}
    for index,no in enumerate(order):
        price=float(win_price[no])
        market_rank[no]=1+next(i for i,k in enumerate(order) if float(win_price[k])==price)
    inverse={no:1/float(win_price[no]) for no in field}
    overround=math.fsum(inverse.values())
    market_p={no:inverse[no]/overround for no in field}
    ai1=next(no for no in field if companions[no]["distill_ai_rank"]==1)
    mk1=[no for no in field if market_rank[no]==1]
    if len(mk1)!=1:
        raise ValueError("ORIGINAL_GEOMETRY_MARKET_FAVORITE_TIE_NO_POPULARITY")
    overlap=sum(companions[no]["distill_ai_rank"]<=3 and market_rank[no]<=3 for no in field)
    top1_match=int(ai1==mk1[0])
    dis=(3-max(0,min(3,overlap)))/3
    result={}
    for no in sorted(field):
        h=companions[no]
        ai=h["distill_ai_rank"]
        mr=market_rank[no]
        diff=mr-ai
        gap=diff/n
        absgap=abs(diff)/n
        risk=float(h["risk"])
        ability=float(h["distill_score_z"])
        f={
            "field_size":n,
            "distill_ai_rank01":1-(ai-1)/max(1,n-1),
            "distill_score_z":ability,
            "distill_score_gap_to_best":float(h["distill_score_gap_to_best"]),
            "distill_is_ai_top1":float(ai==1),
            "distill_is_ai_top3":float(ai<=3),
            "distill_is_ai_top6":float(ai<=6),
            "risk":risk,
            "rank_gap_norm":gap,
            "abs_rank_gap_norm":absgap,
            "top3_overlap_norm":overlap/3,
            "ai_unique_top3":int(ai<=3 and mr>3),
            "market_unique_top3":int(ai>3 and mr<=3),
            "top1_match":top1_match,
            "risk_x_abs_gap":risk*absgap,
            "ability_x_market_gap":ability*gap,
            "ability_x_disagree_level":ability*dis,
            "risk_x_disagree_level":risk*dis,
            "rankgap_x_disagree_level":gap*dis,
        }
        if set(f)!=set(FIELDS):raise ValueError("ORIGINAL_GEOMETRY_FEATURE_SET_DRIFT")
        result[no]={
            "horse_id":h["horse_id"],"horse_number":no,
            "features":f,"market_rank":mr,"market_prob_norm":market_p[no],
            "win_odds":float(win_price[no]),
        }
    return result
