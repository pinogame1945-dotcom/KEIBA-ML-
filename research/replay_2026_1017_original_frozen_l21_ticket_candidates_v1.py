#!/usr/bin/env python3
"""Execute original frozen L2.1 STANDARD 50 ticket generator on saved 2026 candidates.

Frozen create_cards, gate_flags, pair_templates, ticket_price and sort_horses
are executed as untouched function AST from pinned 2025 research SHA.
Only original payout lookup is replaced with an always-empty research response:
the 2026 synthetic pack includes absolutely no results/payouts.
No 2026 model evaluation; quoted odds are saved PRE_RACE_LIVE_ODDS, not
independently authenticated at purchase cutoff. NO LIVE BETTING.
"""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,itertools,json,math
from bisect import bisect_right
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
import pandas as pd

DAY="2026-10-17"
FROZEN="KEIBA_L2_1_FIXED_PURCHASE_V1"
SOURCE="KEIBA_2026_ORIGINAL_PURE_AI_RISK_ANCHOR_RESEARCH_CANDIDATE_V1"
MARKET="KEIBA_L175_BACKFILL_2026_1017_RESEARCH_PRIVATE_V1"
MARKET_RACE="KEIBA_2026_1017_L175_MARKET_JOIN_RESEARCH_V1"
EXPECTED_ML_SOURCES={
 "fixed":"9845ef5c11e74d4e668b463674e3eef15c344702",
 "gates":"2a36c0411835db7a29359894a9bc8eb2ab715e2f"}
TYPE_TO_GROUP={"QUINELLA":"4","TRIO":"7","TRIFECTA":"8","WIN":"1"}
REQUIRED_BETS={"QUINELLA","TRIO","TRIFECTA"}

def guard(ok,why):
    if not ok:raise ValueError(why)

def dig(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_blob(path):
    d=Path(path).read_bytes()
    return hashlib.sha1(b"blob "+str(len(d)).encode()+b"\x00"+d).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def iter_jsonl(path):
    with Path(path).open(encoding="utf8") as file:
        for line in file:
            if line.strip():yield json.loads(line)

def load_frozen_math(files):
    for label in ("fixed","gates"):
        guard(git_blob(files[label])==EXPECTED_ML_SOURCES[label],
              "HISTORIC_FROZEN_SOURCE_BLOB_CHANGED:"+label)
    # Original source functions only: no imports, source data, historical training,
    # payout settlement, class definitions or mutable root source are executed.
    def extract(path,names,space):
        tree=ast.parse(Path(path).read_text(encoding="utf8"),filename=str(path))
        nodes=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in names]
        guard({z.name for z in nodes}==set(names),
              "ORIGINAL_FROZEN_FUNCTION_NOT_FOUND:"+str(names))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),space)
    template_space={"itertools":itertools}
    extract(files["pair"],["pair_templates"],template_space)
    p=SimpleNamespace(pair_templates=template_space["pair_templates"],
                      payouts=lambda sealed,rid: {})
    # Frozen ticket_price reads the same original odds group codec.
    labspace={"math":math,"RANGE_FAMILIES":{"PLACE","WIDE"},
              "BET_TO_GROUP":{"WIN":"1","PLACE":"2","QUINELLA":"4","WIDE":"5",
               "EXACTA":"6","TRIO":"7","TRIFECTA":"8"}}
    extract(files["lab"],["parse_price","sort_horses","ticket_price"],labspace)
    lab=SimpleNamespace(**{z:labspace[z] for z in ("parse_price","sort_horses","ticket_price")})
    lab.FLOORS=(0.,1.5,2.,3.,5.,7.,10.,15.,20.,30.,50.,75.,
                100.,150.,200.,300.,500.,1000.,2000.,5000.)
    lab.ORDERS=("AI_PROB","MARKET","AI_RAW")
    lab.FAMILIES=("QUINELLA","WIDE","EXACTA_FIRST","EXACTA_SECOND",
                  "EXACTA_BOTH","TRIO","TRIFECTA_FIRST","TRIFECTA_SECOND",
                  "TRIFECTA_THIRD","TRIFECTA_ANY")
    # Restore original immutable literal constants from the source AST.
    for name,expected in (("FLOORS",lab.FLOORS),("ORDERS",lab.ORDERS),
                          ("FAMILIES",lab.FAMILIES)):
        tree=ast.parse(files["lab"].read_text(encoding="utf8"))
        vals=[ast.literal_eval(node.value) for node in tree.body
              if isinstance(node,ast.Assign) and any(
                  isinstance(q,ast.Name) and q.id==name for q in node.targets)]
        guard(vals==[expected],"ORIGINAL_L21_TICKET_TEMPLATE_CONSTANT_DRIFT:"+name)
    guard(lab.ticket_price(("TRIO",(1,2,3)),{"odds":{"7":{"010203":[15.,15.,15.]}}})==15.,
          "ORIGINAL_QUOTE_ADAPTER_COMPATIBILITY_FAIL")
    gate_space={}
    extract(files["gates"],["gate_flags"],gate_space)
    def no(horse_pack):
        guard(not(horse_pack.get("results") or horse_pack.get("payouts")),
              "NO_2026_RESULT_OR_PAYOUT_ALLOWED_IN_L21")
        return {str(e["horse_id"]):int(e["horse_number"])
                for e in horse_pack["entries"]}
    fixed_space={"math":math,"mean":mean,"bisect_right":bisect_right,
       "BINS":lab.FLOORS,"FAMILIES":lab.FAMILIES,"prior":p,"lab":lab,
       "horse_number_map":no,
       "assert_ok":lambda v,why: guard(v,why)}
    extract(files["fixed"],["bin_name","partner_p","create_cards"],fixed_space)
    # Exact historical 2025 ANCHOR market-geometry model, no local formula copy.
    spec=importlib.util.spec_from_file_location("frozen_l21_original_anchor",files["geometry"])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    guard(callable(getattr(mod,"build_geometry",None)),"ORIGINAL_ANCHOR_GEOMETRY_ABSENT")
    return fixed_space["create_cards"],gate_space["gate_flags"],mod.build_geometry

def quote_key(typ,seq):
    return str(seq[0]) if len(seq)==1 else "".join(f"{n:02d}" for n in seq)

def official_quote_market(rid,backfill):
    root=backfill/"ml_features/delivery"
    index=read(root/"odds/index.json")
    guard(index["contract"]=="KEIBA_BACKFILL_TO_ML_ODDS_INDEX_V1" and
          rid in index["races"],"BACKFILL_ODDS_INDEX_MISSING:"+rid)
    e=index["races"][rid]
    raw=backfill/e["path"]
    guard(dig(raw)==e["sha256"] and e["date"]==DAY and
          e["source"]=="JRA_OFFICIAL_POST" and
          e["snapshot_type"]=="PRE_RACE_LIVE_ODDS","OFFICIAL_SOURCE_CHANGED:"+rid)
    obj=read(raw)
    guard(obj["contract"]=="KEIBA_BACKFILL_TO_ML_ODDS_V1" and
          obj["race_id"]==rid and obj["race_date"]==DAY and
          obj["original_is_unmodified_source"] is True and
          obj["odds"]["readiness"]=="PRE_RACE_OBSERVED_ASOF_UNVERIFIED" and
          not obj.get("results") and not obj.get("payouts") and
          not obj.get("settlements"),
          "NO_OUTCOME_AND_PRESTART_ODDS_REQUIRED:"+rid)
    source=obj["odds"]
    groups=defaultdict(dict);keys=set();missing=Counter()
    for q in source["quotes"]:
        typ=q["bet_type"];seq=tuple(q["combination"])
        key=(typ,seq)
        guard(key not in keys,"DUPLICATE_JRA_OFFICIAL_TICKET:"+rid)
        keys.add(key)
        if typ not in TYPE_TO_GROUP:continue
        if typ in ("QUINELLA","TRIO"):
            guard(seq==tuple(sorted(seq)),"OFFICIAL_UNORDERED_ODDS_NOT_CANONICAL:"+rid)
        guard(type(q.get("odds")) in (int,float) and
              math.isfinite(q["odds"]) and q["odds"]>=1,
              "ODDS_INVALID_OR_RANGE_NOT_ALLOWED:"+rid)
        v=float(q["odds"]);k=quote_key(typ,seq)
        guard(k not in groups[TYPE_TO_GROUP[typ]],"DUPLICATE_OFFICIAL_TICKET_KEY:"+rid)
        groups[TYPE_TO_GROUP[typ]][k]=[v,v,v]
    for typ in REQUIRED_BETS:
        guard(groups[TYPE_TO_GROUP[typ]],"MISSING_FROZEN_L21_MARKET:"+rid+":"+typ)
    for e in source["no_votes"]+source["scratched_combinations"]:
        if e["bet_type"] in REQUIRED_BETS:
            missing[e["bet_type"]]+=1
    win={int(q["combination"][0]):float(q["odds"]) for q in source["quotes"]
         if q["bet_type"]=="WIN"}
    return {"odds":dict(groups)},win,dict(missing),source["source_captured_at_utc"]

def load_saved_indices(pindex,scores,mindex,markets,freeze,backfill):
    pi,mi,contract=read(pindex),read(mindex),read(freeze)
    guard(pi.get("contract")==SOURCE and pi.get("races")==36 and
          pi.get("horses")==471 and pi.get("original_model_release_approved") is False and
          pi.get("source_asof_independently_verified") is False and
          pi.get("ready_for_betting") is False and
          pi.get("2026_target_outcomes_used") is False and
          pi["files"][scores.name]["sha256"]==dig(scores),
          "ORIGINAL_SAVED_2026_AI_SOURCE_NOT_VERIFIED")
    guard(mi.get("contract")==MARKET and mi.get("races")==36 and
          mi.get("horses")==471 and mi.get("ready_for_betting") is False and
          mi.get("2026_target_results_used") is False and
          mi["files"][markets.name]==dig(markets),
          "FROZEN_L175_SOURCE_UNVERIFIED")
    guard(contract.get("contract")==FROZEN and
          contract.get("version")=="2.1.0" and
          contract.get("primary_policy")=="STANDARD" and
          contract.get("rule_counts",{}).get("STANDARD")==50 and
          contract.get("rule_counts",{}).get("STRICT_SUBSET")==26 and
          len(contract["frozen_rules"]["STANDARD"])==50,
          "FROZEN_STANDARD_50_RULES_CHANGED")
    fs=contract["frozen_rules"]["STANDARD"]
    strict=contract["frozen_rules"]["STRICT_SUBSET"]
    lines=defaultdict(set); strict_lines=defaultdict(set)
    for item in fs:
        lines[item["line"]].add(item["gate"])
    for item in strict:
        strict_lines[item["line"]].add(item["gate"])
        guard(item["gate"] in lines[item["line"]],"STRICT_NOT_SUBSET_OF_STANDARD")
    guard(len(lines)==9 and len({(r["line"],r["gate"]) for r in fs})==50 and
          len({(r["line"],r["gate"]) for r in strict})==26,
          "FROZEN_LINE_GATE_RULE_NOT_50_26")
    source={};market={}
    for r in iter_jsonl(scores):
        rid=str(r.get("race_id") or "")
        guard(rid not in source and r["race_date"]==DAY and
              r["2026_target_results_used"] is False and
              r["ready_for_betting"] is False and
              r["anchor_status"]=="RESEARCH_SCORED_ASOF_UNVERIFIED",
              "UNAPPROVED_OR_CHANGED_ORIGINAL_2026_MODEL:"+rid)
        source[rid]=r
    for r in iter_jsonl(markets):
        rid=str(r.get("race_id") or "")
        guard(rid not in market and r["contract"]==MARKET_RACE and
              r["ready_for_betting"] is False and
              r["2026_target_results_used"] is False and
              r["market_asof_independently_verified"] is False,
              "UNAPPROVED_OR_CHANGED_2026_L175:"+rid)
        market[rid]=r
    guard(len(source)==len(market)==36 and set(source)==set(market),
          "2026_SAVED_SOURCE_NOT_36_RACES")
    return source,market,lines,strict_lines

def run(args):
    guard(not args.output.exists(),"REFUSE_OVERWRITE_L21_REPLAY_REPORT")
    functions={"fixed":args.original_create_cards,"gates":args.original_gates,
       "lab":args.original_odds_lab,"pair":args.original_templates,
       "geometry":args.original_geometry}
    create_cards,gate_flags,geometry=load_frozen_math(functions)
    source,market,lines,strict_lines=load_saved_indices(
        args.pure_index,args.pure_scores,args.market_index,args.market_scores,
        args.frozen_contract,args.backfill)
    output=[];count=Counter();totals=Counter();all_selections=set()
    for rid in sorted(source):
        r=source[rid];m=market[rid]
        originals={str(h["horse_id"]):h for h in r["horses"]}
        ms={str(h["horse_id"]):h for h in m["horses"]}
        guard(len(originals)==len(ms)==r["horse_count"] if "horse_count" in r else
              len(originals)==len(ms)==len(r["horses"]),
              "HORSE_POPULATION_ORIGINAL_L175_CHANGED:"+rid)
        guard(set(originals)==set(ms),"ORIGINAL_L175_HORSE_ID_DRIFT:"+rid)
        market_odds,official_win,missing_quote,quote_timestamp=official_quote_market(rid,args.backfill)
        companions={};win={};raw_rank=[];horses=[]
        for hid,h in originals.items():
            mh=ms[hid];no=h["horse_number"]
            guard(type(no)is int and no==mh["horse_number"] and no not in companions and
                  h["market_rank"]==mh["market_rank"] and
                  h["original_prior_market_win_odds"]==mh["market_win_odds"]==
                    official_win.get(no) and
                  all(type(h[k]) in (int,float) and math.isfinite(h[k])
                      for k in ("P__ANCHOR_FULL","risk","distill_score_z")) and
                  type(h["distill_ai_rank"])is int,
                  "MODEL_AND_JRA_ODDS_MISMATCH_OR_MISSING:"+rid+":"+hid)
            companions[no]={k:h[k] for k in (
                "P__ANCHOR_FULL","risk","distill_score_z","distill_score_gap_to_best",
                "distill_ai_rank","market_rank")}
            companions[no]["horse_id"]=hid
            win[no]=float(official_win[no])
            horses.append({"_horse_id":hid,"P__ANCHOR_FULL":float(h["P__ANCHOR_FULL"]),
                "market_rank":int(h["market_rank"]),
                "final_win_odds":float(win[no]),
                "distill_ai_rank":int(h["distill_ai_rank"])})
        geom=geometry(companions,win)
        ranked=sorted(r["horses"],
                      key=lambda h:(-h["P__ANCHOR_FULL"],h["market_rank"],str(h["horse_id"])))
        pranks={h["horse_number"]:rank for rank,h in enumerate(ranked,1)}
        panel={}
        for n,v in companions.items():
            g=geom[n]
            guard(g["market_rank"]==v["market_rank"] and
                  math.isfinite(g["market_prob_norm"]) and
                  0<g["market_prob_norm"]<1,
                  "ORIGINAL_L21_ANCHOR_GEOMETRY_NOT_IDENTICAL:"+rid)
            panel[n]={**v,"market_prob_norm":g["market_prob_norm"],"P_rank":pranks[n]}
        anchor=int(r["anchor_horse_number"])
        guard(anchor in panel and
              sum(int(h["original_anchor"]) for h in r["horses"])==1 and
              next(h["horse_number"] for h in r["horses"] if h["original_anchor"])==anchor,
              "ORIGINAL_ANCHOR_FULL_AXIS_DRIFT:"+rid)
        grp=pd.DataFrame.from_records(horses)
        pack={"entries":[{"horse_id":h,"horse_number":n} for n,v in companions.items()
                         for h in [v["horse_id"]]],
              "results":[],"payouts":[]}
        raw_top=grp.sort_values(["distill_ai_rank","market_rank","_horse_id"],
                    ascending=[True,True,True],kind="mergesort").iloc[0]["_horse_id"]
        meta={"AI_RAW_TOP_horse":str(raw_top),"VALUE_EV_horse":panel[anchor]["horse_id"]}
        active={};occurrences=0;matched=0;checked=0;miss_odds=0;bands=Counter()
        for tid,ticket,card in create_cards(grp,pack,market_odds,rid,panel[anchor]["horse_id"],meta):
            checked+=1
            line="|".join(map(str,tid))
            rules=lines.get(line)
            if rules is None:continue
            # Exact original STANDARD 50, original gate_flags with original
            # 2025 geometry; strict is diagnostic subset only.
            occ=gate_flags(ticket,panel,anchor)
            selected=rules.intersection(k for k,v in occ.items() if v)
            if not selected:continue
            matched+=len(selected)
            kind,nums=ticket
            price=float(card[0])
            guard(kind in REQUIRED_BETS and
                  any(fam in line for fam in ("TRIO","QUINELLA","TRIFECTA")) and
                  all(n in companions for n in nums) and anchor in nums and
                  math.isfinite(price) and price>0 and
                  card[1]==0,"UNSAFE_OR_UNEXPECTED_L21_TICKET:"+rid)
            key=(kind,tuple(nums))
            occurrences+=1
            if key in active:
                guard(active[key]["odds"]==price,
                      "ODDS_SNAPSHOT_DIFFERENT_PER_IDENTICAL_TICKET:"+rid)
                active[key]["line_gates"].append({"line":line,"gates":sorted(selected)})
                active[key]["strict"]|=bool(
                    strict_lines.get(line,set()).intersection(selected))
                continue
            active[key]={"bet_type":kind,"selection":list(nums),
                "odds":price,"odds_band":str(tid[3]),
                "baseline_yen":100,"strict":bool(
                    strict_lines.get(line,set()).intersection(selected)),
                "line_gates":[{"line":line,"gates":sorted(selected)}],
                "ready_for_betting":False}
            bands[tid[3]]+=1
        # Complete original quotes are not always present: no-vote/scratch
        # combinations fail closed in original ticket_price() -> None.
        unique=list(sorted(active.values(),key=lambda x:(x["bet_type"],x["selection"])))
        for q in unique:
            k=(rid,q["bet_type"],tuple(q["selection"]))
            guard(k not in all_selections,"DUPLICATE_PHYSICAL_TICKET_ACROSS_RACES")
            all_selections.add(k)
        count["races"]+=1;count["horses"]+=len(horses)
        count["unique_tickets"]+=len(unique)
        count["matching_line_gate_events"]+=matched
        count["pre_dedup_ticket_events"]+=occurrences
        count["quoted_template_events"]+=checked
        count["strict_diagnostic_tickets"]+=sum(x["strict"] for x in unique)
        for q in unique:totals[q["bet_type"]]+=1
        output.append({"race_id":rid,"horse_count":len(horses),
          "original_anchor_horse_number":anchor,
          "official_quote_snapshot":quote_timestamp,
          "original_frozen_standard_rules":50,"strict_diagnostic_rules":26,
          "quotation_no_votes_and_scratches_by_type":missing_quote,
          "quoted_template_events":checked,
          "matched_line_gate_events":matched,
          "pre_dedup_selected_ticket_events":occurrences,
          "standard_physical_ticket_count":len(unique),
          "strict_subset_diagnostic_ticket_count":sum(x["strict"] for x in unique),
          "l3_minimum_100_yen_baseline_total":len(unique)*100,
          "l3_300_yen_budget_can_cover_every_ticket":len(unique)<=3,
          "research_only":True,"ready_for_betting":False,
          "tickets":unique})
        print("ORIGINAL_L21_RACE_TICKETS "+json.dumps({
          "race_id":rid,"anchor":anchor,"tickets":len(unique),
          "strict":sum(x["strict"] for x in unique),
          "baseline_yen":len(unique)*100},sort_keys=True),flush=True)
    guard(count["races"]==36 and count["horses"]==471,
          "ORIGINAL_FROZEN_L21_NOT_36R_471H")
    final={"contract":"KEIBA_ORIGINAL_L21_FROZEN_STANDARD_50_2026_1017_RESEARCH_REPLAY_V1",
       "day":DAY,"frozen_policy":"STANDARD","original_frozen_version":"2.1.0",
       "races":36,"horses":471,
       "frozen_standard_rule_count":50,"frozen_strict_subset_rule_count":26,
       "frozen_create_cards_invoked":True,"frozen_gate_flags_invoked":True,
       "original_geometry_invoked":True,"original_template_generator_invoked":True,
       "total_standard_unique_physical_tickets":count["unique_tickets"],
       "bet_types":dict(totals),"races_with_tickets":sum(x["standard_physical_ticket_count"]>0 for x in output),
       "races_exceeding_300_yen_if_all_tickets_min100":sum(
           x["standard_physical_ticket_count"]>3 for x in output),
       "counts":dict(count),"research_race_outputs":output,
       "payout_source_sealed_empty":True,"target_2026_results_or_payouts_read":False,
       "source_asof_independently_verified":False,
       "at_bet_cutoff_quote_verified":False,
       "model_release_approved":False,
       "l3_v3_invoked":False,"ready_for_betting":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(final,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("ORIGINAL_L21_FROZEN_RESEARCH_REPLAY_SUMMARY "+json.dumps({
      "races":36,"horses":471,"standard_rules":50,"strict_rules":26,
      "races_with_tickets":final["races_with_tickets"],
      "physical_tickets":count["unique_tickets"],
      "duplicates_eliminated":count["pre_dedup_ticket_events"]-count["unique_tickets"],
      "types":dict(totals),
      "budget_over_300_yen_races":final["races_exceeding_300_yen_if_all_tickets_min100"],
      "payouts":False,"betting":False},sort_keys=True),flush=True)
    guard(final["ready_for_betting"] is False,"LIVE_BETTING_FORBIDDEN")

def cli():
    ap=argparse.ArgumentParser()
    for field in ("pure-index","pure-scores","market-index","market-scores",
      "frozen-contract","backfill","original-create-cards","original-gates",
      "original-odds-lab","original-templates","original-geometry","output"):
        ap.add_argument("--"+field,type=Path,required=True)
    return ap.parse_args()
if __name__=="__main__":run(cli())
