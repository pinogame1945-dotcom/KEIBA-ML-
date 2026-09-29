#!/usr/bin/env python3
import math
from collections import Counter

RANK_TIE_POLICY = "score_desc_then_horse_id_asc_v1"

def _normalize(scores, horse_ids):
    if len(scores) != len(horse_ids):
        raise ValueError("scores/horse_ids length mismatch")
    ids=[str(x or "") for x in horse_ids]
    if any(not x for x in ids):
        raise ValueError("empty horse_id in ranking input")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate horse_id in ranking input")
    vals=[]
    for x in scores:
        v=float(x)
        if not math.isfinite(v):
            raise ValueError("non-finite ranking score")
        vals.append(v)
    return vals,ids

def deterministic_order(scores, horse_ids):
    vals,ids=_normalize(scores,horse_ids)
    return sorted(range(len(vals)), key=lambda i:(-vals[i], ids[i]))

def deterministic_ranks(scores, horse_ids):
    order=deterministic_order(scores,horse_ids)
    ranks=[0]*len(order)
    for rank,pos in enumerate(order, start=1):
        ranks[pos]=rank
    return ranks

def tie_diagnostics(scores, horse_ids, boundaries=(1,3,6)):
    vals,ids=_normalize(scores,horse_ids)
    order=deterministic_order(vals,ids)
    counts=Counter(vals)
    tied_values={score:size for score,size in counts.items() if size>1}
    rank_by_pos=[0]*len(order)
    for rank,pos in enumerate(order,start=1):
        rank_by_pos[pos]=rank

    boundary={}
    for n in boundaries:
        if not isinstance(n,int) or n<1:
            raise ValueError("boundaries must be positive integers")
        if n>=len(order):
            boundary[str(n)]=False
            continue
        left=order[n-1]
        right=order[n]
        boundary[str(n)]=vals[left]==vals[right]

    tied_horses=sum(tied_values.values())
    return {
        "policy":RANK_TIE_POLICY,
        "field_size":len(vals),
        "tie_group_count":len(tied_values),
        "tied_horse_count":tied_horses,
        "max_tie_group_size":max(tied_values.values(),default=1),
        "any_tie":bool(tied_values),
        "boundary_tie":boundary,
    }
