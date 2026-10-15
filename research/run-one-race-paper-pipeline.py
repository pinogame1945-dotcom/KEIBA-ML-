#!/usr/bin/env python3
"""Offline race-ID -> existing L1 reference -> original L2.1 -> fixed L3 *paper*.

No invented model prices, no original ANCHOR_FULL reconstruction from council.
Missing evidence produces a durable explicit BLOCKED manifest. Quotes do not
represent authenticated JRA publisher/server timestamps and are not executable.
"""
from __future__ import annotations
import argparse,importlib.util,json,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load_file(rel,modname):
    spec=importlib.util.spec_from_file_location(modname,ROOT/"research"/rel)
    module=importlib.util.module_from_spec(spec)
    sys.modules[modname]=module
    spec.loader.exec_module(module)
    return module

BRIDGE=load_file("bridge-mobile-combination-odds.py","l21_combo_bridge")
L21=load_file("l21-frozen-prebet-price.py","l21_paper")
L3=load_file("l3-one-race-money.py","l3_paper")
PREVIEW=load_file("l21-format-price-preview.py","l21_paper_preview")

def save(p, data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def run(rid, outdir, mobile_file=None, original_file=None, limits_file=None, analysis_file=None):
    if not re.fullmatch(r"202[2-5]\d{8}",rid):
        raise ValueError("L2_1_PAPER_RESEARCH_RACE_ID_2022_TO_2025_2026_SEALED")
    root=Path(outdir)
    root.mkdir(parents=True,exist_ok=True)
    stages={
        "L1_HORSE":"NOT_PROVIDED" if not analysis_file else "PROVENANCE_NOT_CHECKED",
        "L1_75_PRODUCTION":"BLOCKED_NO_PROMOTED_ASOF_MARKET_CALIBRATOR",
        "ORIGINAL_ANCHOR_FULL":"MISSING_EXACT_ORIGINAL_PREBET_SOURCE",
        "JRA_MULTI_ODDS":"MISSING_PREBET_SOURCE",
        "L2_1_TICKET":"NOT_STARTED",
        "L3_MONEY":"NOT_STARTED"
    }
    missing=[]
    for name,path in [
       ("KEIBA_MOBILE_OFFICIAL_JRA_CAPTURE",mobile_file),
       ("ORIGINAL_ANCHOR_FULL_L21_HORSE_FEATURES",original_file),
       ("L3_FIXED100_RACE_BUDGET_LIMITS",limits_file)]:
        if not path:missing.append(name)
    if analysis_file:
        a=json.loads(Path(analysis_file).read_text(encoding="utf-8"))
        if a.get("contract")!="KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1" or a.get("race_id")!=rid:
            raise ValueError("L1_EXTERNAL_ANALYSIS_ID_OR_CONTRACT_DRIFT")
        stages["L1_HORSE"]="HISTORICAL_OR_SEPARATE_ANALYSIS_ATTACHED_NOT_LIVE_CERTIFIED"
    report={
        "contract":"KEIBA_ONE_RACE_PAPER_ORCHESTRATION_V1",
        "race_id":rid,"status":"BLOCKED_MISSING_INPUTS",
        "stages":stages,"required_inputs_missing":missing,
        "l21_output_path":None,"l3_output_path":None,
        "source_price_authenticity_independently_verified":False,
        "source_model_asof_independently_verified":False,
        "L1_75_PRODUCTION_PROMOTED":False,
        "actual_bets_placed":False,"2026_sealed_hindsight":True,
        "note":"A staged diagnostic workflow. Not a deployable live betting system."
    }
    output=root/"pipeline-status.json"
    if missing:
        save(output,report)
        print("PAPER_ONE_RACE_BLOCKED_INPUTS",rid,",".join(missing))
        return report
    card=json.loads(Path(mobile_file).read_text(encoding="utf-8"))
    orig=json.loads(Path(original_file).read_text(encoding="utf-8"))
    money=json.loads(Path(limits_file).read_text(encoding="utf-8"))
    mobile_rid=str((card.get("race") or {}).get("canonicalRaceId") or "")
    if mobile_rid!=rid or orig.get("race_id")!=rid:
        raise ValueError("ONE_RACE_MOBILE_AND_ORIGINAL_ANCHOR_ID_DRIFT")
    # If L1 is supplied, prohibit a different horse-number / identity mapping.
    if analysis_file:
        horses=a.get("horses") or []
        originals=orig.get("horses") or []
        left={(str(h.get("horse_id")),int(h.get("horse_number") or 0)) for h in horses}
        right={(str(h.get("horse_id")),int(h.get("horse_number") or 0)) for h in originals}
        if len(left)!=len(horses) or left!=right:
            raise ValueError("L1_VS_ORIGINAL_L21_HORSE_ID_NUMBER_MISMATCH")
    try:
        quotes=BRIDGE.convert(card)
        stages["JRA_MULTI_ODDS"]="ALL_THREE_COMBOS_AND_WIN_SAME_SNAPSHOT_VALIDATED"
        save(root/"JRA-quote-book.json",quotes)
        ticket=L21.execute(card,orig,quotes)
        save(root/"L2_1-quoted-tickets.json",ticket)
        (root/"L2_1-preview.md").write_text(PREVIEW.render(ticket),encoding="utf-8")
        stages["ORIGINAL_ANCHOR_FULL"]="EXACT_SOURCE_ASSERTED_ASOF_NOT_INDEPENDENTLY_VERIFIED"
        stages["L2_1_TICKET"]=ticket["status"]
        portfolio=L3.allocate(ticket,money)
        save(root/"L3-fixed100-paper-portfolio.json",portfolio)
        stages["L3_MONEY"]=portfolio["status"]
        report["l21_output_path"]="L2_1-quoted-tickets.json"
        report["l3_output_path"]="L3-fixed100-paper-portfolio.json"
        report["status"]="PAPER_ONLY_LIMITS_CHECKED" if portfolio["status"]=="PAPER_ONLY_BASELINE_READY" else "BLOCKED_UPSTREAM_OR_BUDGET"
        report["result_ticket_count"]=ticket["physical_ticket_count"]
        report["result_total_baseline_yen"]=ticket["baseline_total_stake_yen"]
        report["result_portfolio_status"]=portfolio["status"]
    except (ValueError,KeyError,TypeError) as exc:
        report["status"]="BLOCKED_CONTRACT_OR_PROVENANCE_DRIFT"
        report["failure_reason"]=str(exc)
        stages["L2_1_TICKET"]="BLOCKED"
        stages["L3_MONEY"]="NOT_STARTED"
    save(output,report)
    print("PAPER_ONE_RACE_STAGE_REPORT",rid,report["status"],
          report.get("result_ticket_count",0),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race",required=True)
    p.add_argument("--mobile-export")
    p.add_argument("--original-features")
    p.add_argument("--limits")
    p.add_argument("--analysis")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    run(a.race,a.out_dir,a.mobile_export,a.original_features,a.limits,a.analysis)

if __name__=="__main__":
    main()
