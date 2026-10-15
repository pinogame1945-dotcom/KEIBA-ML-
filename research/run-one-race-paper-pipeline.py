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
ASOF=load_file("score-original-anchor-asof.py","l21_reconstructed_original_anchor")

def save(p, data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def run(rid, outdir, mobile_file=None, original_file=None, limits_file=None, analysis_file=None,
        original_parameters_file=None, original_ability_risk_file=None, _synthetic_now=None):
    if not re.fullmatch(r"202[2-6]\d{8}",rid):
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
    if original_file and (original_parameters_file or original_ability_risk_file):
        raise ValueError("CANNOT_MIX_SAVED_ANCHOR_INPUT_WITH_RECONSTRUCTED_MODEL")
    reconstruct=bool(original_parameters_file or original_ability_risk_file)
    missing=[]
    for name,path in [
       ("KEIBA_MOBILE_OFFICIAL_JRA_CAPTURE",mobile_file),
       ("L3_FIXED100_RACE_BUDGET_LIMITS",limits_file)]:
        if not path:missing.append(name)
    if reconstruct:
        if not original_parameters_file:missing.append("FROZEN_PRE2025_ANCHOR_FULL_PARAMETERS")
        if not original_ability_risk_file:missing.append("TRUE_PREBET_ORIGINAL_ABILITY_AND_RISK_COMPANIONS")
    elif not original_file:
        missing.append("ORIGINAL_ANCHOR_FULL_L21_HORSE_FEATURES")
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
    orig=(json.loads(Path(original_file).read_text(encoding="utf-8"))
          if original_file else None)
    money=json.loads(Path(limits_file).read_text(encoding="utf-8"))
    mobile_rid=str((card.get("race") or {}).get("canonicalRaceId") or "")
    if mobile_rid!=rid or (orig is not None and orig.get("race_id")!=rid):
        raise ValueError("ONE_RACE_MOBILE_AND_ORIGINAL_ANCHOR_ID_DRIFT")
    try:
        quotes=BRIDGE.convert(card)
        if reconstruct:
            model=json.loads(Path(original_parameters_file).read_text(encoding="utf-8"))
            companions=json.loads(Path(original_ability_risk_file).read_text(encoding="utf-8"))
            prediction=ASOF.calc(model,card,companions,_synthetic_now)
            if prediction["race_id"]!=rid:
                raise ValueError("RECONSTRUCTED_ANCHOR_RACE_ID_DRIFT")
            orig=prediction["model_full_probs"]
            save(root/"L1_75-reconstructed-original-anchor-paper.json",prediction)
            save(root/"L2_1-original-asof-features.json",orig)
            stages["ORIGINAL_ANCHOR_FULL"]="RECONSTRUCTED_PRIOR_MODEL_NEEDS_INDEPENDENT_UPSTREAM_COMPANION_ATTESTATION"
            stages["L1_75_PRODUCTION"]="NOT_PROMOTED_FROZEN_ORIGINAL_COEFF_RECONSTRUCTED_PAPER_ONLY"
        if orig is None or orig.get("race_id")!=rid:
            raise ValueError("MISSING_EXACT_OR_RECONSTRUCTED_ORIGINAL_ANCHOR")
        # Source L1 and L2 horse ID mapping MUST be identical, no drift
        # or quiet fallback from one system's horse number to another.
        if analysis_file:
            horses=a.get("horses") or []
            originals=orig.get("horses") or []
            left={(str(h.get("horse_id")),int(h.get("horse_number") or 0)) for h in horses}
            right={(str(h.get("horse_id")),int(h.get("horse_number") or 0)) for h in originals}
            if len(left)!=len(horses) or left!=right:
                raise ValueError("L1_VS_ORIGINAL_L21_HORSE_ID_NUMBER_MISMATCH")
        stages["JRA_MULTI_ODDS"]="ALL_THREE_COMBOS_AND_WIN_SAME_SNAPSHOT_VALIDATED"
        save(root/"JRA-quote-book.json",quotes)
        ticket=L21.execute(card,orig,quotes)
        save(root/"L2_1-quoted-tickets.json",ticket)
        (root/"L2_1-preview.md").write_text(PREVIEW.render(ticket),encoding="utf-8")
        if not reconstruct:
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
    p.add_argument("--original-anchor-parameters",
                   help="Reconstructed true prior-only original ANCHOR_FULL beta+scaler, never 2025-refitted")
    p.add_argument("--original-ability-risk",
                   help="Complete ORIGINAL timestamped L1 ability / Correction Risk as-of companion horses. No proxy 7 kings.")
    p.add_argument("--out-dir",required=True)
    a=p.parse_args()
    run(a.race,a.out_dir,a.mobile_export,a.original_features,a.limits,a.analysis,
        a.original_anchor_parameters,a.original_ability_risk)

if __name__=="__main__":
    main()
