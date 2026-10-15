#!/usr/bin/env python3
"""KEIBA L2.1 frozen STANDARD 50 line-gates: exact pricing, zero strategy drift.

Inputs are separately attested ORIGINAL ANCHOR_FULL axis/horse features,
not interchangeable 7-King or L1.7 proxy rankings. Odds are independently
observed official-JRA PRE_RACE mobile OddsRow snapshots with one timestamp.
No result, payout, historical FINAL or 2026 hindsight input. No purchases.
"""
from __future__ import annotations
import argparse,datetime,hashlib,itertools,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/"contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json"
FROZEN_ID="KEIBA_L2_1_FIXED_PURCHASE_V1"
FAMILIES={"QUINELLA","TRIO","TRIFECTA_FIRST","TRIFECTA_THIRD"}
BET_TYPES={"QUINELLA":2,"TRIO":3,"TRIFECTA":3}
UNORDERED={"QUINELLA","TRIO"}
ORDERS={"AI_PROB","AI_RAW","MARKET"}
GATE_NAMES={"ALL","RAW_TOP6","RAW_TOP10","CALIBRATED_TOP6","CALIBRATED_TOP10",
    "MARKET_TOP6","MARKET_TOP10","RISK_LT040","RISK_LT060","ABILITY_Z_GE_M1",
    "ABILITY_Z_GE_0","MODEL_WIN_P_GE_003","MARKET_UNDERVALUE_GE_2","AGREEMENT_TOP6"}
BAND={"15-20":(15.0,20.0),"30-50":(30.0,50.0),
      "20-30":(20.0,30.0),"200-300":(200.0,300.0)}

def timestamp(value,name):
    if not isinstance(value,str) or not value:
        raise ValueError("MISSING_TIME_"+name)
    try:d=datetime.datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as e:raise ValueError("INVALID_TIME_"+name) from e
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError("TIMEZONE_REQUIRED_"+name)
    return d.astimezone(datetime.timezone.utc)

def number(value,label):
    if isinstance(value,bool):raise ValueError("BOOLEAN_NOT_NUMBER_"+label)
    try:n=float(value)
    except (ValueError,TypeError) as e:raise ValueError("MISSING_NUMERIC_"+label) from e
    if not math.isfinite(n):raise ValueError("NONFINITE_"+label)
    return n

def no(value,label):
    if isinstance(value,bool) or type(value) is not int or not 1<=value<=24:
        raise ValueError("INVALID_HORSE_NUMBER_"+label)
    return value

def source_contract():
    data=json.loads(FROZEN.read_text(encoding="utf-8"))
    if (data.get("contract")!=FROZEN_ID or data.get("version")!="2.1.0"
        or data.get("primary_policy")!="STANDARD"
        or data.get("rule_counts",{}).get("STANDARD")!=50
        or data.get("rule_counts",{}).get("STRICT_SUBSET")!=26
        or data.get("ticket_policy",{}).get("stake_baseline_yen_per_ticket")!=100):
        raise ValueError("FROZEN_L21_CONTRACT_DRIFT")
    lines=data["frozen_rules"]["STANDARD"]
    strict=data["frozen_rules"]["STRICT_SUBSET"]
    keys={(x["line"],x["gate"]) for x in lines}
    strict_keys={(x["line"],x["gate"]) for x in strict}
    if len(keys)!=50 or not strict_keys.issubset(keys) or len(strict_keys)!=26:
        raise ValueError("FROZEN_L21_RULE_DRIFT")
    if set(x["line"] for x in lines)!=set(data["unique_standard_original_lines"]):
        raise ValueError("FROZEN_NINE_TEMPLATE_DRIFT")
    for line,gate in keys:
        parts=line.split("|")
        if len(parts)!=4 or parts[0] not in ORDERS or int(parts[1]) not in (2,3,5) or parts[2] not in FAMILIES or parts[3] not in BAND or gate not in GATE_NAMES:
            raise ValueError("UNRECOGNIZED_FROZEN_RULE_"+line+":"+gate)
    return data,keys,strict_keys

def canonical(kind,nums):
    if kind not in BET_TYPES or not isinstance(nums,(list,tuple)):
        raise ValueError("UNKNOWN_TICKET")
    if len(nums)!=BET_TYPES[kind]:raise ValueError("WRONG_SELECTION_ARITY")
    values=tuple(no(n,"ticket") for n in nums)
    if len(set(values))!=len(values):raise ValueError("DUPLICATE_HORSE_IN_SELECTION")
    return tuple(sorted(values)) if kind in UNORDERED else values

def generated(anchor,partners,family):
    if family=="QUINELLA":
        return {("QUINELLA",canonical("QUINELLA",(anchor,b))) for b in partners}
    pairs=list(itertools.combinations(partners,2))
    if family=="TRIO":
        return {("TRIO",canonical("TRIO",(anchor,b,c))) for b,c in pairs}
    if family=="TRIFECTA_FIRST":
        return {("TRIFECTA",(anchor,x,y)) for b,c in pairs for x,y in ((b,c),(c,b))}
    if family=="TRIFECTA_THIRD":
        return {("TRIFECTA",(x,y,anchor)) for b,c in pairs for x,y in ((b,c),(c,b))}
    raise ValueError("NOT_FROZEN_FAMILY")

def horse_features(source,rid,active):
    if (source.get("contract")!="KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1"
        or source.get("race_id")!=rid
        or source.get("anchor_source")!="EXACT_EXISTING_ANCHOR_FULL"
        or source.get("feature_source")!="ORIGINAL_L21_HORSE_FEATURES"
        or source.get("input_phase")!="PRE_RACE"
        or source.get("features_asof_asserted") is not True):
        raise ValueError("L21_ORIGINAL_HORSE_FEATURE_PROVENANCE_NOT_VERIFIED")
    people=source.get("horses")
    if not isinstance(people,list) or len(people)!=len(active):
        raise ValueError("L21_HORSE_FIELD_COVERAGE")
    by_no={}; ids=set()
    for h in people:
        n=no(h.get("horse_number"),"original_features")
        hid=h.get("horse_id")
        if n in by_no or not isinstance(hid,str) or not hid or hid in ids:
            raise ValueError("HORSE_NUMBER_OR_ID_DUPLICATED")
        # Copy and canonicalize numeric fields, never trust CSV-string
        # lexicographic ordering or mutate the caller's original model panel.
        h=dict(h)
        by_no[n]=h;ids.add(hid)
        for name in ("P__ANCHOR_FULL","risk","distill_score_z","market_prob_norm"):
            v=number(h.get(name),name)
            if name in ("P__ANCHOR_FULL","risk","market_prob_norm") and not 0<=v<=1:
                raise ValueError("FEATURE_PROBABILITY_OUT_OF_RANGE_"+name)
            h[name]=v
        for name in ("distill_ai_rank","market_rank"):
            x=h.get(name)
            if isinstance(x,bool) or type(x)is not int or not 1<=x<=len(active):
                raise ValueError("RANK_FIELD_MISSING_OR_INVALID_"+name)
        h["win_odds_at_cutoff"]=number(h.get("win_odds_at_cutoff"),"win_odds_at_cutoff")
        if h["win_odds_at_cutoff"]<1:
            raise ValueError("WIN_PRICE_INVALID")

    if set(by_no)!=active:raise ValueError("ORIGINAL_FEATURE_HORSES_NOT_ACTIVE_FIELD")
    ahid=source.get("anchor_horse_id")
    if not isinstance(ahid,str) or ahid not in ids:
        raise ValueError("FROZEN_ANCHOR_ID_NOT_FOUND")
    axis=next(n for n,h in by_no.items() if h["horse_id"]==ahid)
    return by_no,axis

def gate_values(ticket,horse,axis):
    kind,nums=ticket
    partners=[n for n in nums if n!=axis]
    if len(partners)!=len(nums)-1:
        raise ValueError("FROZEN_TICKET_MISSING_OR_DUPLICATE_AXIS")
    ps=[horse[n] for n in partners]
    risk=max(p["risk"] for p in ps)
    ability=min(p["distill_score_z"] for p in ps)
    pwin=min(p["P__ANCHOR_FULL"] for p in ps)
    raw=max(p["distill_ai_rank"] for p in ps)
    prob=max(p["P_rank"] for p in ps)
    market=max(p["market_rank"] for p in ps)
    undervalue=min(p["market_rank"]-p["distill_ai_rank"] for p in ps)
    return {
        "ALL":True,"RAW_TOP6":raw<=6,"RAW_TOP10":raw<=10,
        "CALIBRATED_TOP6":prob<=6,"CALIBRATED_TOP10":prob<=10,
        "MARKET_TOP6":market<=6,"MARKET_TOP10":market<=10,
        "RISK_LT040":risk<.40,"RISK_LT060":risk<.60,
        "ABILITY_Z_GE_M1":ability>=-1,"ABILITY_Z_GE_0":ability>=0,
        "MODEL_WIN_P_GE_003":pwin>=.03,
        "MARKET_UNDERVALUE_GE_2":undervalue>=2,
        "AGREEMENT_TOP6":raw<=6 and market<=6
    }

def combine_quote(source, rid, active, race_key, start):
    if (source.get("contract")!="KEIBA_JRA_COMBINATION_ODDS_ASOF_V1"
        or source.get("canonical_race_id")!=rid
        or source.get("raceKey")!=race_key
        or source.get("source_system")!="KEIBA_MOBILE_JRA"
        or source.get("source_phase")!="PRE_RACE"
        or source.get("is_final") is not False
        or source.get("source_status")!="LIVE"
        or source.get("source_confirmed_prestart") is not True):
        raise ValueError("COMBINATION_QUOTES_NOT_AUTHORIZED_PREBET")
    t=timestamp(source.get("observed_at"),"quote_capture")
    cutoff=timestamp(source.get("bet_cutoff_at"),"bet_cutoff")
    if not t<cutoff<start:
        raise ValueError("QUOTE_NOT_BEFORE_BET_CUTOFF_OR_START")
    if rid.startswith("2026") and t>datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("2026_FUTURE_OBSERVATION_NOT_YET_MADE")
    if timestamp(source.get("scheduled_start_at"),"market_start")!=start:
        raise ValueError("MARKET_RACE_START_TIME_DRIFT")
    rows=source.get("rows")
    # 18 starters: C(18,2)+C(18,3)+18*17*16 = 5,865
    # three-combination quotes (plus 18 WIN in a separate list).
    if not isinstance(rows,list) or not rows or len(rows)>5865:
        raise ValueError("UNSAFE_COMBINATION_QUOTE_ROWS")
    prices={}
    for r in rows:
        kind=r.get("betType")
        if kind not in BET_TYPES:raise ValueError("UNEXPECTED_NONL21_BET_TYPE")
        if r.get("raceKey")!=race_key:raise ValueError("ODDS_RACE_KEY_DRIFT")
        if timestamp(r.get("observedAt"),"row_observed")!=t:
            raise ValueError("MIXED_QUOTE_TIMESTAMPS")
        u=urlparse(str(r.get("sourceUrl") or ""))
        if u.scheme!="https" or u.hostname not in ("www.jra.go.jp","jra.go.jp","www.jra.jp","jra.jp"):
            raise ValueError("UNTRUSTED_JRA_SOURCE_URL")
        nums=[r.get("selection1"),r.get("selection2")]
        if kind in ("TRIO","TRIFECTA"):nums.append(r.get("selection3"))
        elif r.get("selection3") is not None:
            raise ValueError("UNEXPECTED_THIRD_HORSE_FOR_QUINELLA")
        ticket=(kind,canonical(kind,nums))
        if not set(ticket[1]).issubset(active):raise ValueError("QUOTE_NOT_IN_ACTIVE_RACE_FIELD")
        if ticket in prices:raise ValueError("DUPLICATE_PHYSICAL_QUOTE")
        if r.get("oddsMin") is not None or r.get("oddsMax") is not None:
            raise ValueError("RANGE_ODDS_UNUSABLE_FOR_L21")
        n=number(r.get("odds"),"combination_odds")
        if not 1<=n<=1000000:raise ValueError("INVALID_COMBINATION_PRICE")
        prices[ticket]=n
    win_rows=source.get("win_odds_rows")
    if not isinstance(win_rows,list) or len(win_rows)!=len(active):
        raise ValueError("WIN_ODDS_SAME_SNAPSHOT_COMPLETE_FIELD_REQUIRED")
    win_prices={}
    for row in win_rows:
        n=no(row.get("selection1"),"win_quote")
        if (row.get("betType")!="WIN" or row.get("raceKey")!=race_key or
            row.get("selection2") is not None or row.get("selection3") is not None or
            n not in active or n in win_prices):
            raise ValueError("ORIGINAL_WIN_QUOTE_RACE_OR_FIELD_DRIFT")
        if timestamp(row.get("observedAt"),"win_row_observed")!=t:
            raise ValueError("ORIGINAL_WIN_QUOTE_NOT_SAME_COMBO_SNAPSHOT")
        if row.get("oddsMin") is not None or row.get("oddsMax") is not None:
            raise ValueError("INVALID_WIN_QUOTE_RANGE")
        u=urlparse(str(row.get("sourceUrl") or ""))
        if u.scheme!="https" or u.hostname not in ("www.jra.go.jp","jra.go.jp","www.jra.jp","jra.jp"):
            raise ValueError("WIN_SOURCE_MUST_BE_OFFICIAL_JRA")
        price=number(row.get("odds"),"win_odds")
        if not 1<=price<=9999.9:raise ValueError("INVALID_WIN_ODDS_PRICE")
        win_prices[n]=price
    if set(win_prices)!=active:
        raise ValueError("ORIGINAL_WIN_QUOTE_HORSE_COVERAGE_DRIFT")
    return prices,win_prices,t,cutoff

def validate_live_card(payload):
    race=payload.get("race") or {}
    rid=str(race.get("canonicalRaceId") or "")
    if not re.fullmatch(r"202[2-6]\d{8}",rid):
        raise ValueError("L21_ONLY_2022_2026_RACE_IDS")
    if (race.get("raceStatus")!="SCHEDULED" or race.get("scheduleStatus")!="ACTIVE"
        or race.get("status")!="OFFICIAL"):
        raise ValueError("RESULT_OR_CANCELLED_RACE_NOT_LIVE")
    d=race.get("raceDate");clock=race.get("startTime")
    if not re.fullmatch(r"\d{4}-\d\d-\d\d",str(d)) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d",str(clock)):
        raise ValueError("LIVE_START_TIME_MISSING")
    start=timestamp(f"{d}T{clock}:00+09:00","live_start")
    # 2026 results remain sealed, but FUTURE 2026 live races are not
    # historical post-result backtests.
    if rid.startswith("2026") and start<=datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("2026_HISTORICAL_RACE_REPLAY_SEALED_ONLY_FUTURE_PREBET_ALLOWED")
    entries=payload.get("entries")
    if not isinstance(entries,list) or len(entries)<3:raise ValueError("LIVE_ENTRIES_MISSING")
    active=set()
    for r in entries:
        if r.get("raceKey")!=race.get("raceKey"):
            raise ValueError("ENTRY_RACE_KEY_MISMATCH")
        st=r.get("entryStatus")
        if st=="ACTIVE":
            n=no(r.get("horseNo"),"live_active")
            if n in active:raise ValueError("DUPLICATE_ACTIVE_HORSE_NUMBER")
            active.add(n)
        elif st not in ("SCRATCHED","EXCLUDED"):raise ValueError("UNKNOWN_ENTRY_STATUS")
    if len(active)<3:raise ValueError("NOT_ENOUGH_ACTIVE_HORSES")
    return race,rid,active,start

def execute(card,model,quotes,policy="STANDARD"):
    contract,standard,strict=source_contract()
    if policy not in ("STANDARD","STRICT_SUBSET"):
        raise ValueError("NO_SUCH_FROZEN_POLICY")
    race,rid,active,start=validate_live_card(card)
    horse,axis=horse_features(model,rid,active)
    prices,win_prices,observed,cutoff=combine_quote(quotes,rid,active,race["raceKey"],start)
    # Time alignment for ORIGINAL model data. Never admit models computed
    # after the quote/bet cutoff (even when the quote itself is PRE_RACE).
    feature_time=timestamp(model.get("features_observed_at"),"model_feature_observed_at")
    win_quote_time=timestamp(model.get("win_odds_observed_at"),"original_win_quote_observed_at")
    model_time=timestamp(model.get("anchor_computed_at"),"original_anchor_computed_at")
    if feature_time>observed:
        raise ValueError("L21_HORSE_FEATURES_AFTER_QUOTE_SNAPSHOT")
    if win_quote_time!=observed:
        raise ValueError("L21_ORIGINAL_ANCHOR_WIN_ODDS_NOT_SAME_MARKET_SNAPSHOT")
    if not observed<=model_time<cutoff:
        raise ValueError("L21_ORIGINAL_ANCHOR_COMPUTED_OUTSIDE_QUOTE_TO_CUTOFF")
    if rid.startswith("2026") and model_time>datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("2026_MODEL_OUTPUT_FROM_FUTURE_NOT_ALLOWED")
    if model.get("canonical_race_key")!=race["raceKey"]:
        raise ValueError("L21_ORIGINAL_AXIS_RACE_KEY_DRIFT")
    # Exact historic sort_horses: AI_PROB by P then market rank then horse ID;
    # MARKET by market rank and win odds then horse ID, AI_RAW by raw rank,
    # market rank, horse ID. Here field market_win_odds must be original
    # same-snapshot pre-cutoff quote, NOT archived FINAL prices.
    for n,h in horse.items():
        v=number(h.get("win_odds_at_cutoff"),"win_odds_at_cutoff")
        if v<1:raise ValueError("INVALID_WIN_ODDS_AT_CUTOFF")
        if abs(v-win_prices[n])>1e-8:
            raise ValueError("ORIGINAL_ANCHOR_WIN_PRICE_NOT_EQUAL_ACTUAL_JRA_SAME_SNAPSHOT")
    orders={
       "AI_PROB":sorted(active,key=lambda n:(-horse[n]["P__ANCHOR_FULL"],horse[n]["market_rank"],horse[n]["horse_id"])),
       "AI_RAW":sorted(active,key=lambda n:(horse[n]["distill_ai_rank"],horse[n]["market_rank"],horse[n]["horse_id"])),
       "MARKET":sorted(active,key=lambda n:(horse[n]["market_rank"],horse[n]["win_odds_at_cutoff"],horse[n]["horse_id"]))
    }
    prob_order=orders["AI_PROB"]
    for i,n in enumerate(prob_order,start=1):horse[n]["P_rank"]=i
    approved=standard if policy=="STANDARD" else strict
    by_line=defaultdict(set);physical=defaultdict(set);missing=set()
    counts=Counter()
    for line,gate in sorted(approved):
        order,k,family,band=line.split("|")
        others=[n for n in orders[order] if n!=axis][:int(k)]
        for ticket in generated(axis,others,family):
            quote=prices.get(ticket)
            if quote is None:
                missing.add(ticket)
                counts["MISSING_TICKET_QUOTE"]+=1
                continue
            low,high=BAND[band]
            if not low<=quote<high:continue
            if not gate_values(ticket,horse,axis)[gate]:continue
            by_line[(line,gate)].add(ticket)
            physical[ticket].add((line,gate))
    tickets=[]
    for ticket,causes in sorted(physical.items()):
        kind,nums=ticket
        tickets.append({
            "race_id":rid,"bet_type":kind,
            "canonical_selection":"-".join(map(str,nums)),
            "selection_numbers":list(nums),
            "observed_at":quotes["observed_at"],
            "decimal_odds":prices[ticket],
            "baseline_stake_yen":100,
            "matched_line_gates":[{"line":line,"gate":gate} for line,gate in sorted(causes)],
            "strict_subset":any((line,gate) in strict for line,gate in causes)
        })
    # Missing candidate quotes MUST NOT be silently interpreted as filtered
    # odds bands. Incomplete quote books block all purchase-ready outputs.
    blocked=len(missing)>0
    if blocked:tickets=[]
    return {
      "contract":"KEIBA_L21_PREBET_QUOTED_TICKETS_V1",
      "frozen_policy":contract["contract"],
      "frozen_version":contract["version"],
      "policy":policy,
      "race_id":rid,
      "axis_horse_id":horse[axis]["horse_id"],"axis_horse_number":axis,
      "status":"BLOCKED_INCOMPLETE_QUOTE_BOOK" if blocked else "PAPER_ONLY_PREBET_QUOTE_VALIDATED",
      "source_authenticity_independently_verified":False,
      "original_anchor_and_horse_model_provenance_asserted_not_independently_verified":True,
      "historical_final_odds_used":False,
      "2026_hindsight_sealed":True,
      "quote_observed_at":quotes["observed_at"],
      "bet_cutoff_at":quotes["bet_cutoff_at"],
      "base_stake_yen":100,
      "physical_ticket_count":len(tickets),
      "baseline_total_stake_yen":100*len(tickets),
      "matching_line_gate_count":sum(len(v) for v in physical.values()),
      "ticket_quote_missing_unique_count":len(missing),
      "ticket_quote_missing_rule_occurrences":counts["MISSING_TICKET_QUOTE"],
      "missing_quote_tickets":[{"bet_type":k,"canonical_selection":"-".join(map(str,nums))}
          for k,nums in sorted(missing)],
      "tickets":tickets,
      "strict_is_diagnostic_subset_only":True,
      "not_a_live_bet_execution_approval":True,
      "not_a_future_roi_guarantee":True,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--card",required=True,help="KEIBA-MOBILE exported PRE_RACE {race,entries} JSON")
    p.add_argument("--original-features",required=True,help="EXACT_ORIGINAL L2.1 ANCHOR_FULL, P__ANCHOR_FULL, raw ranks, risk, score-z, etc, nonproxy")
    p.add_argument("--combination-quotes",required=True,help="OFFICIAL pre-cutoff JRA OddsRows, one observedAt, quinella/trio/trifecta")
    p.add_argument("--policy",choices=("STANDARD","STRICT_SUBSET"),default="STANDARD")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    result=execute(
        json.loads(Path(a.card).read_text(encoding="utf-8")),
        json.loads(Path(a.original_features).read_text(encoding="utf-8")),
        json.loads(Path(a.combination_quotes).read_text(encoding="utf-8")),a.policy)
    dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L21_FROZEN_PREBET_PRICE",
      result["race_id"],result["status"],"tickets",result["physical_ticket_count"],
      "missing quotes",result["ticket_quote_missing_unique_count"])
if __name__=="__main__":main()
