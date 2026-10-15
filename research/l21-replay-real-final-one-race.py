#!/usr/bin/env python3
"""Compare FROZEN L2.1 STANDARD tickets for an actual historical day against
the original ticket-level audit ledger. Strict retrospective FINAL-only audit.

This is NOT a pre-bet inference or executable trading system. No outcome or
payout is used in input selection, gating, price or confidence. Audit compares
with baseline ONLY AFTER deriving the independent set of historical tickets.
"""
from __future__ import annotations
import argparse,csv,gzip,importlib.util,itertools,json,math,re
from collections import defaultdict
from pathlib import Path

SCRIPT=Path(__file__).resolve().with_name("l21-frozen-prebet-price.py")
spec=importlib.util.spec_from_file_location("keiba_frozen_l21",SCRIPT)
l21=importlib.util.module_from_spec(spec);spec.loader.exec_module(l21)
FLOORS=(0.0,1.5,2.0,3.0,5.0,7.0,10.0,15.0,20.0,30.0,50.0,
        75.0,100.0,150.0,200.0,300.0,500.0,1000.0,2000.0,5000.0)
GROUP={"QUINELLA":"4","TRIO":"7","TRIFECTA":"8"}

def csv_rows(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8",newline="") as file:
        yield from csv.DictReader(file)

def race_in_pack(path,rid,market=False):
    with gzip.open(path,"rt",encoding="utf-8") as f:
        found=[]
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            ident=str(row.get("race_id") or "") if market else str((row.get("race") or {}).get("race_id") or "")
            if ident==rid:found.append(row)
    if len(found)!=1:raise ValueError("BACKFILL_TARGET_RACE_NOT_UNIQUE_"+str(len(found)))
    return found[0]

def valid_rank(raw,label):
    x=l21.number(raw,label)
    if not x.is_integer() or not 1<=x<=24:
        raise ValueError("INVALID_INTEGER_RANK_"+label)
    return int(x)

def research_frame(rid,pred,ability,picks,race_pack):
    year=int(rid[:4])
    raw=[r for r in csv_rows(pred) if r.get("_race_id")==rid and int(r.get("year") or 0)==year]
    z=[r for r in csv_rows(ability) if r.get("_race_id")==rid and int(r.get("year") or 0)==year]
    axis=[r.get("_horse_id") for r in csv_rows(picks) if
          r.get("_race_id")==rid and r.get("strategy")=="ANCHOR_FULL" and int(r.get("year") or 0)==year]
    if len(axis)!=1:raise ValueError("EXACT_ORIGINAL_ANCHOR_FULL_NOT_UNIQUE")
    original={r["_horse_id"]:r for r in raw}
    ability_by_id={r["_horse_id"]:r for r in z}
    number={}
    for e in race_pack.get("entries") or []:
        hid=str(e.get("horse_id") or "")
        if hid and e.get("horse_number") is not None:
            n=int(e["horse_number"])
            if not 1<=n<=24 or n in number.values() or hid in number:
                raise ValueError("BACKFILL_HORSE_NUMBER_DRIFT")
            number[hid]=n
    if (not raw or len(raw)!=len(original) or len(z)!=len(ability_by_id) or
        set(original)!=set(ability_by_id) or set(number)!=set(original) or axis[0] not in number):
        raise ValueError("ORIGINAL_FEATURE_OR_PACK_HORSE_ID_COVERAGE_DRIFT "+
            str({"race_id":rid,"archive_predictions":len(raw),
                 "ability_cache":len(z),"numbered_entries":len(number),
                 "pred_without_pack":sorted(set(original)-set(number))[:8],
                 "pack_without_pred":sorted(set(number)-set(original))[:8],
                 "pack_unpredicted_entry_status":[{
                     "horse_id":str(e.get("horse_id")),
                     "horse_number":e.get("horse_number"),
                     **{k:e.get(k) for k in ("entry_status","result_status",
                       "status","finish_position","finish_raw","is_eligible","is_runner")}
                    } for e in (race_pack.get("entries") or [])
                    if str(e.get("horse_id")) in set(number)-set(original)][:8],
                 "pred_without_ability":sorted(set(original)-set(ability_by_id))[:8],
                 "ability_without_pred":sorted(set(ability_by_id)-set(original))[:8],
                 "axis_in_pack":axis[0] in number}))
    horse={}
    for hid,row in original.items():
        a=ability_by_id[hid]
        n=number[hid]
        item={
          "horse_id":hid,"horse_number":n,
          "P__ANCHOR_FULL":l21.number(row.get("P__ANCHOR_FULL"),"P__ANCHOR_FULL"),
          "market_prob_norm":l21.number(row.get("market_prob_norm"),"market_prob_norm"),
          "risk":l21.number(row.get("risk"),"risk"),
          "distill_score_z":l21.number(a.get("distill_score_z"),"distill_score_z"),
          "distill_ai_rank":valid_rank(row["distill_ai_rank"],"distill_ai_rank"),
          "market_rank":valid_rank(row["market_rank"],"market_rank"),
          "win_odds_at_cutoff":l21.number(row.get("final_win_odds"),"HISTORICAL_FINAL_WIN_PRICE"),
        }
        horse[n]=item
    # Exact source tie-safe ranks; these are original archival L2.1 inputs,
    # *not* reconstructed from the new 7-King committee.
    probs=sorted(horse, key=lambda n:(-horse[n]["P__ANCHOR_FULL"],horse[n]["market_rank"],horse[n]["horse_id"]))
    for i,n in enumerate(probs,1):horse[n]["P_rank"]=i
    return horse,number[axis[0]],axis[0]

def historical_price(kind,nums,oddspack):
    seq="".join(f"{n:02d}" for n in nums)
    raw=(oddspack.get("odds") or {}).get(GROUP[kind],{}).get(seq)
    if raw is None:return None
    values=raw[3:6] if len(raw)>=6 else raw[:3]
    if len(values)<1:return None
    try:v=float(str(values[0]).replace(",",""))
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) and v>0 else None

def band(value):
    for lo,hi in zip(FLOORS,FLOORS[1:]):
        if lo<=value<hi:
            name=lambda v: str(int(v)) if float(v).is_integer() else str(v)
            return name(lo)+"-"+name(hi)
    return "5000+" if value>=5000 else "UNKNOWN"

def replay_one(rid,pred,ability,picks,pack_path,odds_path):
    if not re.fullmatch(r"202[3-5]\d{8}",rid):
        raise ValueError("RESEARCH_REPLAY_ONLY_2023_2025_2026_SEALED")
    frozen,standard,strict=l21.source_contract()
    pack=race_in_pack(pack_path,rid)
    oddspack=race_in_pack(odds_path,rid,True)
    if oddspack.get("odds_kind")!="FINAL" or oddspack.get("source_status")!="result":
        raise ValueError("ONLY_EXPLICIT_HISTORICAL_FINAL_CAN_BE_REPLAYED")
    horse,axis,hid=research_frame(rid,pred,ability,picks,pack)
    active=set(horse)
    ranked={
      "AI_PROB":sorted(active,key=lambda n:(-horse[n]["P__ANCHOR_FULL"],horse[n]["market_rank"],horse[n]["horse_id"])),
      "MARKET":sorted(active,key=lambda n:(horse[n]["market_rank"],horse[n]["win_odds_at_cutoff"],horse[n]["horse_id"])),
      "AI_RAW":sorted(active,key=lambda n:(horse[n]["distill_ai_rank"],horse[n]["market_rank"],horse[n]["horse_id"]))
    }
    wanted=set(line for line,gate in standard)
    records=defaultdict(set)
    # Invoke exact frozen generations/gates. Check all eligible candidates
    # have FINAL prices: missing originals need not imply all types complete.
    missing=set()
    for line in sorted(wanted):
        source,k,fam,accepted_band=line.split("|")
        others=[n for n in ranked[source] if n!=axis][:int(k)]
        for ticket in l21.generated(axis,others,fam):
            price=historical_price(*ticket,oddspack)
            if price is None:
                missing.add(ticket)
                continue
            if band(price)!=accepted_band:continue
            flags=l21.gate_values(ticket,horse,axis)
            for chosen,gate in standard:
                if chosen==line and flags[gate]:
                    records[ticket].add((line,gate))
    if missing:raise ValueError(f"HISTORICAL_COMBINATION_ODDS_MISSING_{len(missing)}")
    tickets=[]
    for ticket,matched in sorted(records.items()):
        kind,nums=ticket
        tickets.append({
            "bet_type":kind,"selection":"-".join(map(str,nums)),
            "ticket_odds":historical_price(*ticket,oddspack),
            "strict":any(x in strict for x in matched),
            "line_gate_count":len(matched),
        })
    return {"contract":"KEIBA_L21_HISTORICAL_FINAL_REPRODUCTION_V1",
      "race_id":rid,"anchor_horse_id":hid,"anchor_horse_number":axis,
      "from_original_anchor_full":True,
      "all_model_outputs_historical_final_not_verified_bet_time":True,
      "source_odds_kind":"FINAL","official_live_execution":False,
      "ticket_count":len(tickets),"tickets":tickets}

def compare(rows,baseline_known,baseline_missing,rid):
    reference={}
    for name,path in (("known",baseline_known),("uncovered",baseline_missing)):
        for row in csv_rows(path):
            if row.get("race_id")!=rid:continue
            key=(row["bet_type"],row["selection"])
            if key in reference:raise ValueError("ORIGINAL_L21_BASELINE_DUPLICATE_TICKET")
            reference[key]=row
    tested={(t["bet_type"],t["selection"]):t for t in rows["tickets"]}
    if tested.keys()!=reference.keys():
        raise ValueError("HISTORICAL_L21_TICKET_SET_DRIFT missing="+
            str(sorted(reference.keys()-tested.keys())[:12])+
            " extra="+str(sorted(tested.keys()-reference.keys())[:12]))
    for key,t in tested.items():
        row=reference[key]
        if "ticket_odds" in row and row["ticket_odds"]:
            if abs(float(row["ticket_odds"])-t["ticket_odds"])>1e-8:
                raise ValueError("HISTORICAL_L21_TICKET_ODDS_DRIFT "+str(key))
        if "strict" in row and row["strict"]:
            if int(row["strict"])!=int(t["strict"]):
                raise ValueError("HISTORICAL_L21_STRICT_SUBSET_DRIFT "+str(key))
    rows["original_saved_ticket_count"]=len(reference)
    rows["original_saved_ticket_set_exact"]=True
    return rows

def main():
    p=argparse.ArgumentParser()
    for flag in ("race","predictions","ability-cache","picks","race-pack","odds-pack",
                 "baseline-covered","baseline-uncovered","output"):
        p.add_argument("--"+flag,required=True)
    a=p.parse_args()
    rows=replay_one(a.race,a.predictions,a.ability_cache,a.picks,a.race_pack,a.odds_pack)
    result=compare(rows,a.baseline_covered,a.baseline_uncovered,a.race)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HISTORICAL_REAL_L21_PREVIOUS_RESEARCH_TICKETS_EXACT_MATCH",
          result["race_id"],result["ticket_count"],"anchor",result["anchor_horse_number"],
          "SOURCE_FINAL_NOT_PREBET",flush=True)
if __name__=="__main__":main()
