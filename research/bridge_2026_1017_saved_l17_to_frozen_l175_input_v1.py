#!/usr/bin/env python3
"""Connect real, sealed 2026-10-17 L1.7 scores to FIXED L1.75 market interface.

No 2026 training/evaluation, no retroactive odds, no 7King reranking, no
fabricated market ranks, and NO operational L1.75/L2.1/L3 promotion.
The archived 2021-25 ORIGINAL L1.7 deployment calibrator provides podium p3
for King ranks <= 10 only. Frozen L1.75 market fields remain null if genuine
independently certified PRE_RACE WIN quotes are not provided.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,math
from pathlib import Path
from collections import defaultdict

SOURCE_CONTRACT="ORIGINAL13_7KING_2026_1017_RESEARCH_PRIVATE_V1"
SIGNAL_CONTRACT="KEIBA_2026_7KING_13_OUTSIDER_RESEARCH_ONLY_SIGNAL_V1"
MODEL_CONTRACT="L17_KING_OUTSIDER_PODIUM_CALIBRATOR_MODEL_V1"
MARKET_CONTRACT="L175_MARKET_CONTEXT_V1"
OUTPUT_CONTRACT="KEIBA_2026_1017_L17_TO_FROZEN_L175_RESEARCH_BRIDGE_V1"
ALLOWED_RACES=36
ALLOWED_HORSES=471
FORBIDDEN=("result","finish","payout","profit","return_yen","roi","2026_truth","actual_is_win")
FIELDS={
    "king_borda_score":"borda_score","king_normalized_borda":"normalized_borda",
    "king_mean_rank":"mean_rank","king_rank_std":"rank_std",
    "king_best_rank":"best_rank","king_worst_rank":"worst_rank",
    "king_top1_votes":"top1_votes","king_probability_mean":"probability_mean",
    "king_probability_std":"probability_std",
}

def need(ok,msg):
    if not ok:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as file:
        for part in iter(lambda:file.read(1024*1024),b""):h.update(part)
    return h.hexdigest()

def rows(path):
    op=gzip.open if str(path).endswith(".gz") else open
    with op(path,"rt",encoding="utf8") as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def validate_contract(c):
    need(c.get("contract")==MARKET_CONTRACT and c.get("version")=="1.0.0" and
         c.get("status")=="FIXED_BRIDGE_CONTRACT" and c.get("2026_locked") is True and
         c.get("ranking_policy")=="SEVEN_KING_UNCHANGED" and
         c.get("outsider_policy")=="SIGNAL_ONLY_NO_RERANK" and
         c.get("market_policy")=="JOIN_AFTER_L17_FREEZE","FROZEN_MARKET_BRIDGE_DRIFT")
    need(c.get("market_fields")==["market_win_odds","market_rank"] and
         "reranked_rank" in c.get("prohibited_outputs",[]) and
         "blended_rank" in c.get("prohibited_outputs",[]),"FROZEN_MARKET_OUTPUT_CONTRACT_DRIFT")

def validate_model(m):
    need(m.get("contract")==MODEL_CONTRACT and m.get("training_years")==[2021,2022,2023,2024,2025] and
         m.get("2026_sealed") is True and m.get("odds_used") is False and
         m.get("C")==0.5 and m.get("model")=="sklearn.linear_model.LogisticRegression" and
         len(m.get("coef") or [])==31 and len(m.get("rank_baseline_p3") or {})==10 and
         set(m["rank_baseline_p3"])==set(str(n) for n in range(1,11)),"ORIGINAL_L17_DEPLOYMENT_CALIBRATOR_DRIFT")
    need(all(isinstance(x,(int,float)) and math.isfinite(x) for x in m["coef"]) and
         math.isfinite(m["intercept"]),"INVALID_ORIGINAL_CALIBRATOR_COEFFICIENTS")
    need(all(0<=float(v)<=1 for v in m["rank_baseline_p3"].values()),"INVALID_ORIGINAL_P3_BASELINE")

def p3(rank,score,model):
    # Exact 31-feature design from frozen train_l17_king_outsider_podium_calibrator_wf_v1.py:features()
    need(type(rank) is int and 1<=rank<=10 and type(score) is int and 0<=score<=39,"BAD_ORIGINAL_CALIBRATOR_INPUT")
    features=[0.0]*31
    features[0]=score/10.0
    features[rank]=1.0
    features[10+rank]=score/10.0
    features[20+rank]=(score/10.0)**2
    raw=float(model["intercept"])+sum(w*x for w,x in zip(model["coef"],features))
    result=1/(1+math.exp(-raw)) if raw>=0 else math.exp(raw)/(1+math.exp(raw))
    baseline=float(model["rank_baseline_p3"][str(rank)])
    need(0<=result<=1 and math.isfinite(result),"BAD_DEPLOYMENT_CALIBRATION_RESULT")
    return result,baseline,result-baseline

def build(source_index,source_signal,king_consensus,model_file,contract_file,output):
    need(not output.exists(),"REFUSE_RESEARCH_OUTPUT_OVERWRITE")
    index=json.loads(source_index.read_text(encoding="utf-8"))
    need(index.get("contract")==SOURCE_CONTRACT and
         index.get("races")==ALLOWED_RACES and index.get("horses")==ALLOWED_HORSES and
         index.get("outsiders")==13 and index.get("ready_for_betting") is False and
         index.get("model_release_authorized") is False and
         index.get("2026_results_used") is False and
         index.get("ranking_policy")=="SEVEN_KING_UNCHANGED" and
         index.get("asof_independently_verified") is False,"ORIGINAL13_KING_SOURCE_INDEX_DRIFT")
    need(index.get("files",{}).get(source_signal.name)==sha(source_signal),
         "PRIVATE_ORIGINAL13_7KING_SOURCE_SHA256_DRIFT")
    model=json.loads(model_file.read_text(encoding="utf-8"))
    contract=json.loads(contract_file.read_text(encoding="utf-8"))
    validate_model(model);validate_contract(contract)
    councils={}
    council_total=0
    for race in rows(king_consensus):
        rid=str(race.get("race_id") or "")
        need(rid not in councils and race.get("contract")=="L16_FULL_RANK_CONSENSUS_V1" and
             race.get("expert_count")==7 and race.get("candidate_policy")=="ALL_HORSES",
             "SEVENKING_ORIGINAL_COUNCIL_DRIFT")
        horses=race.get("horses") or []
        need(len(horses)==race.get("field_size") and
             sorted(int(h["consensus_rank"]) for h in horses)==list(range(1,len(horses)+1)),
             "SEVENKING_RANKS_DRIFT")
        councils[rid]={str(h["horse_id"]):h for h in horses}
        council_total+=len(horses)
    need(len(councils)==60 and council_total==786,"ORIGINAL_7KING_COUNCIL_NOT_60R_786H")
    output.parent.mkdir(parents=True,exist_ok=True)
    write=gzip.open if str(output).endswith(".gz") else open
    races=horses=calibrated=0
    seen=set();hist=defaultdict(int)
    with write(output,"wt",encoding="utf-8") as destination:
        for entry in rows(source_signal):
            rid=str(entry.get("race_id") or "")
            need(rid not in seen and rid in councils and rid.startswith("2026") and
                 entry.get("contract")==SIGNAL_CONTRACT and
                 entry.get("prediction_year")==2026 and
                 entry.get("candidate_count")==13 and
                 entry.get("ranking_policy")=="SEVEN_KING_UNCHANGED" and
                 entry.get("outsider_policy")=="SIGNAL_ONLY_NO_RERANK" and
                 entry.get("calibration_available") is False and
                 entry.get("ready_for_betting") is False and
                 entry.get("model_release_authorized") is False and
                 entry.get("2026_results_used") is False and
                 entry.get("l175_l21_l3_invoked") is False,
                 "SOURCE_RACE_OR_RELEASE_DRIFT")
            original=councils[rid]
            signal=entry.get("horses") or []
            need(len(signal)==len(original)==entry.get("horse_count") and
                 {str(z["horse_id"]) for z in signal}==set(original) and
                 sorted(int(z["king_rank"]) for z in signal)==list(range(1,len(signal)+1)),
                 "FROZEN_KING_AND_OUTSIDER_POPULATION_DRIFT")
            out=[]
            for z in sorted(signal,key=lambda z:z["king_rank"]):
                hid=str(z["horse_id"]);base=original[hid];rank=int(z["king_rank"])
                need(type(z["king_rank"]) is int and rank==base["consensus_rank"] and
                     float(z["king_borda_score"])==float(base["borda_score"]) and
                     z["king_probability_mean"]==base["probability_mean"] and
                     z["p3_calibrated"] is None and z["calibration_status"]=="NOT_TRAINED_FOR_2026",
                     "SEVENKING_UNCHANGED_SIGNAL_PROVENANCE_FAIL")
                count=z["outsider_support_count"];score=z["outsider_score"]
                votes=[z["outsider_top1_votes"],z["outsider_top2_votes"],z["outsider_top3_votes"]]
                need(all(type(x) is int and 0<=x<=13 for x in votes+[count,score]) and
                     count==sum(votes) and score==3*votes[0]+2*votes[1]+votes[2] and
                     0<=count<=13 and 0<=score<=39,"ORIGINAL13_OUTSIDER_VOTE_ARITHMETIC_DRIFT")
                horse={"horse_id":hid,"king_rank":rank,
                       "outsider_score":score,"outsider_support_count":count,
                       "outsider_top1_votes":votes[0],"outsider_top2_votes":votes[1],
                       "outsider_top3_votes":votes[2]}
                for key,old in FIELDS.items():
                    value=base[old]
                    need(value is None or (isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)),
                         "BAD_ORIGINAL_COUNCIL_DIAGNOSTIC:"+key)
                    horse[key]=value
                if rank<=10:
                    a,b,c=p3(rank,score,model)
                    horse.update({"p3_calibrated":a,"p3_rank_baseline":b,"p3_delta":c,
                        "calibration_status":"2021_2025_FROZEN_DEPLOYMENT_RESEARCH_ONLY"})
                    calibrated+=1
                else:
                    horse.update({"p3_calibrated":None,"p3_rank_baseline":None,"p3_delta":None,
                        "calibration_status":"OUTSIDE_KING_TOP10_SCOPE"})
                # Authentic pre-race WIN odds were not present in the stored 10/17 signal.
                # A post-race market snapshot is NEVER backfilled into a PRE_RACE L1.75.
                horse.update({"market_win_odds":None,"market_rank":None,
                  "signed_rank_gap":None,"abs_rank_gap":None,"dissent_direction":None,
                  "outsider_alignment_score":None,"outsider_alignment":"NA"})
                need("reranked_rank" not in horse and
                     "blended_rank" not in horse and
                     not any(t in k.lower() for k in horse for t in FORBIDDEN),
                     "PROHIBITED_L175_FIELD")
                out.append(horse)
            need([h["king_rank"] for h in out]==list(range(1,len(out)+1)),
                 "ORIGINAL_7KING_RANK_REORDER_NOT_ALLOWED")
            destination.write(json.dumps({
              "contract":OUTPUT_CONTRACT,"race_id":rid,"prediction_year":2026,
              "l17_upstream_contract":"L17_TO_L2_SIGNAL_V1",
              "l175_frozen_market_contract":MARKET_CONTRACT,
              "market_readiness":"BLOCKED_NO_AUTHENTIC_PRE_RACE_WIN_ODDS",
              "p3_source":MODEL_CONTRACT,"p3_training_years":[2021,2022,2023,2024,2025],
              "p3_calibration_scope":"KING_RANK_1_TO_10_ONLY",
              "original_7king_models_research_only":True,
              "ranking_policy":"SEVEN_KING_UNCHANGED","outsider_policy":"SIGNAL_ONLY_NO_RERANK",
              "2026_target_results_used":False,"2026_model_fit_or_evaluation":False,
              "2026_asof_independently_verified":False,"ready_for_betting":False,
              "l21_l3_invoked":False,"horse_count":len(out),"horses":out,
            },ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
            seen.add(rid);races+=1;horses+=len(out)
            hist[len(out)]+=1
    need(races==36 and horses==471 and calibrated==sum(min(10,n)*c for n,c in hist.items()),
         "L175_SAFE_RESEARCH_36R_471H_OR_TOP10_COUNT_FAILED")
    print("L175_ORIGINAL_2026_1017_P3_BRIDGE_SAFE_RESEARCH_PASS "+
          json.dumps({"races":races,"horses":horses,"frozen_deployment_p3":calibrated,
                      "market_quote_observations":0,"L175_market_complete":False,
                      "ready_for_betting":False,"out":str(output)},sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    for name in ("source-index","source-signal","king-consensus","deployment-model","frozen-market-contract","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    build(a.source_index,a.source_signal,a.king_consensus,a.deployment_model,a.frozen_market_contract,a.output)
if __name__=="__main__":main()
