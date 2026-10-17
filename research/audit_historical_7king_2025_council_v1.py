#!/usr/bin/env python3
"""Audit the real 2025 historical seven-king consensus; 2026 stays sealed."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

EXPERTS = {"core4", "pedlegacy", "condition", "full", "light", "pedv1", "condrc"}
FORBIDDEN = ("odds", "popularity", "payout", "return_yen", "profit_yen",
             "roi", "target", "finish_position", "actual_is_win")

def verify(path: Path):
    races=set()
    count=0
    first=None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            race=json.loads(line)
            rid=str(race.get("race_id") or "")
            if (race.get("contract") != "L16_FULL_RANK_CONSENSUS_V1" or
                race.get("expert_count") != 7 or
                race.get("candidate_policy") != "ALL_HORSES" or
                not rid.startswith("2025") or rid in races):
                raise ValueError("REAL_2025_COUNCIL_RACE_CONTRACT_DRIFT:"+rid)
            horses=race.get("horses") or []
            n=race.get("field_size")
            if n != len(horses) or not n:
                raise ValueError("REAL_2025_COUNCIL_FIELD_COUNT_DRIFT:"+rid)
            if sorted(int(row["consensus_rank"]) for row in horses) != list(range(1,n+1)):
                raise ValueError("REAL_2025_COUNCIL_RANKS_NOT_CONTIGUOUS:"+rid)
            if len({str(row["horse_id"]) for row in horses}) != n:
                raise ValueError("REAL_2025_COUNCIL_DUPLICATED_HORSE:"+rid)
            for row in horses:
                er=row.get("expert_ranks") or {}
                if set(er)!=EXPERTS or not all(type(x)==int and 1<=x<=n for x in er.values()):
                    raise ValueError("REAL_2025_COUNCIL_EXPERT_RANK_DRIFT:"+rid)
                for key,val in row.items():
                    if any(x in key.lower() for x in FORBIDDEN):
                        raise ValueError("REAL_2025_COUNCIL_OUTCOME_OR_ODDS_LEAK:"+key)
                    if isinstance(val,float) and not math.isfinite(val):
                        raise ValueError("REAL_2025_COUNCIL_NONFINITE:"+key)
            if first is None:
                first=rid
            races.add(rid)
            count+=n
    if not races or not count:
        raise ValueError("REAL_2025_COUNCIL_EMPTY")
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            digest.update(chunk)
    result={
      "contract":"KEIBA_2025_ORIGINAL_7KING_L16_COUNCIL_VALIDATION_V1",
      "year":2025,"race_count":len(races),"horse_rows":count,
      "experts":7,"first_race_id":first,
      "consensus_gzip_sha256":digest.hexdigest(),
      "source_2025_out_of_sample_real_scores":True,
      "original_l16_logic_reused_without_modifications":True,
      "2026_races_or_outcomes_read":False,
      "2026_models_trained_or_scored":False,
      "l175_l21_l3_invoked":False,
      "approved_for_2026_betting":False,
    }
    print("REAL_2025_SEVEN_KING_FROZEN_L16_COUNCIL_PASS "+
          json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    a=p.parse_args()
    verify(a.input)
