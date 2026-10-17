#!/usr/bin/env python3
"""Audit a sealed 2026 unapproved seven-king consensus, with no results."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

EXPERTS={"core4","pedlegacy","condition","full","light","pedv1","condrc"}
BANNED=("odds","popularity","payout","return_yen","profit_yen",
        "roi","target","finish_position","actual_is_win","actual_")
EXPECTED_RACES=60
EXPECTED_HORSES=786

def fail(reason):
    raise ValueError(reason)

def audit(path):
    seen=set()
    horses=0
    days={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            race=json.loads(line)
            rid=str(race.get("race_id") or "")
            if race.get("contract")!="L16_FULL_RANK_CONSENSUS_V1" or \
               race.get("expert_count")!=7 or race.get("candidate_policy")!="ALL_HORSES":
                fail("2026_RESEARCH_COUNCIL_SCHEMA_DRIFT")
            if not rid.startswith("2026") or rid in seen:
                fail("2026_RESEARCH_COUNCIL_RACE_SCOPE_DRIFT")
            seen.add(rid)
            rows=race.get("horses") or []
            size=race.get("field_size")
            if not isinstance(size,int) or size!=len(rows) or size<1:
                fail("2026_RESEARCH_COUNCIL_FIELD_SIZE_MISMATCH")
            ids={str(x.get("horse_id")) for x in rows}
            if len(ids)!=size:
                fail("2026_RESEARCH_COUNCIL_DUPLICATE_HORSE")
            if sorted(x.get("consensus_rank") for x in rows)!=list(range(1,size+1)):
                fail("2026_RESEARCH_COUNCIL_RANKS_NOT_CONTIGUOUS")
            for h in rows:
                er=h.get("expert_ranks") or {}
                if set(er)!=EXPERTS or any(not isinstance(x,int) or x<1 or x>size
                                            for x in er.values()):
                    fail("2026_RESEARCH_COUNCIL_EXPERT_RANKS_MISSING")
            def assert_no_banned(v):
                if isinstance(v,dict):
                    for k,val in v.items():
                        low=str(k).lower()
                        if any(x in low for x in BANNED):
                            fail("2026_RESEARCH_COUNCIL_RESULTS_MARKET_LEAK:"+low)
                        assert_no_banned(val)
                elif isinstance(v,list):
                    for x in v: assert_no_banned(x)
                elif isinstance(v,float) and not math.isfinite(v):
                    fail("2026_RESEARCH_COUNCIL_NONFINITE")
            assert_no_banned(race)
            horses+=size
            day=rid[:4]
            days[day]=days.get(day,0)+1
    if len(seen)!=EXPECTED_RACES or horses!=EXPECTED_HORSES:
        fail("2026_RESEARCH_COUNCIL_60_786_COVERAGE_DRIFT")
    result={"contract":"KEIBA_2026_UNAPPROVED_RESEARCH_COUNCIL_AUDIT_V1",
            "races":len(seen),"horse_rows":horses,"experts":7,
            "2026_results_used":False,"not_approved_for_betting":True,
            "l175_l21_l3_invoked":False,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    print("2026_UNRELEASED_SEVEN_KING_L16_COUNCIL_PASS "+
          json.dumps(result,sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--input",type=Path,required=True)
    args=a.parse_args()
    audit(args.input)
