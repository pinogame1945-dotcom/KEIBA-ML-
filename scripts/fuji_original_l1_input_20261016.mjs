#!/usr/bin/env node
/**
 * Strict actual 2026 Fuji (14 horses) -> ORIGINAL PR#7 one-race inference input
 * CONTRACT ONLY. No models fabricated, no archived 2026 result lookup, no
 * scores, no odds, no override of model registration or frozen definitions.
 * P2 source is real netkeiba+BACKFILL, with source hashes.
 */
import {spawnSync} from "node:child_process";
import {readFile,writeFile,mkdir} from "node:fs/promises";
import path from "node:path";
import {ML_FEATURE_SCHEMA_VERSION,ML_DATASET_VERSION} from "../src/ml-dataset.mjs";
import {applyPredictionPhase,normalizeFeatureSets,normalizeHistoryWindows,selectFeatureFamilies,assertPredictionPhaseSafe} from "../src/l1-feature-contract.mjs";
import {DEFAULT_SMALL_SAMPLE_POLICY} from "../src/small-sample-feature-utils.mjs";
const source=JSON.parse(await readFile("audit-results/fuji-20261017-149-asof-20261016.json","utf8"));
const contract=JSON.parse(await readFile("contracts/l1-feature-catalog-v1.json","utf8"));
const allowed=JSON.parse(await readFile("contracts/l1-feature-set-contract-v1.json","utf8"));
const previous=spawnSync(process.execPath,["scripts/fuji_prebet_phase2_20261016.mjs"],{encoding:"utf8",maxBuffer:20*1024*1024});
if(previous.status!==0)throw Error("PHASE2_INTEGRITY_GATE_FAILED:"+previous.stderr.slice(-900));
const x=previous.stdout.split("\n").find(s=>s.startsWith("PHASE2_JSON "));
if(!x)throw Error("PHASE2_JSON_MISSING");
const joined=JSON.parse(x.slice("PHASE2_JSON ".length));
if(joined.summary.runner_count!==14||joined.summary.top_level_null_cells!==37||joined.summary.candidate_key_collisions_or_type_errors!==0)throw Error("P2_CONTRACT_INTEGRITY");
const knownIds=new Set(source.runners.map(h=>h.horse_id));
const exactFamilyOrder="BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,BACKFILL,AUTO,ACTOR,TIME_PACE"; // PR#7 runOneRace default
const featureSets=normalizeFeatureSets(exactFamilyOrder);
const windows=normalizeHistoryWindows({});
const policy={...DEFAULT_SMALL_SAMPLE_POLICY,actorRecentWindow:windows.actor_recent};
const forbidden=new Set(contract.features.filter(f=>f.l1_allowed===false).flatMap(f=>f.model_keys??[]));
const raceFacts=new Set(["venue_code","meeting_no","meeting_day","race_no","discipline","surface","distance_m","direction"]);
const entryFacts=new Set(["gate","horse_number","sex","age","carried_weight"]);
const rows=[];
for(const h of joined.horses){
 if(!knownIds.has(h.horse_id)||h.feature_count!==134||h.candidate_conflicts.length)throw Error("HORSE_INPUT_NOT_VERIFIED");
 const base={...h.model_candidate_features};
 for(const [key,val] of Object.entries(h.race)){
   if(raceFacts.has(key)){
     if(Object.hasOwn(base,key)&&base[key]!==val)throw Error("RACE_FEATURE_CONFLICT:"+key);
     base[key]=val;
   }
 }
 for(const [key,val] of Object.entries(h.entry)){
   if(entryFacts.has(key)){
     if(Object.hasOwn(base,key)&&base[key]!==val)throw Error("ENTRY_FEATURE_CONFLICT:"+key);
     base[key]=val;
   }
 }
 base.race_date="2026-10-17";
 const safe=applyPredictionPhase(selectFeatureFamilies(base,featureSets),"EARLY");
 assertPredictionPhaseSafe(Object.keys(safe),"EARLY");
 if(Object.keys(safe).some(k=>forbidden.has(k)))throw Error("CATALOG_FORBIDDEN_MODEL_COLUMN");
 if(Object.hasOwn(safe,"weather")||Object.hasOwn(safe,"track_condition")||Object.hasOwn(safe,"body_weight")||Object.hasOwn(safe,"body_weight_diff"))throw Error("FUTURE_RACE_DAY_FIELD_INCLUDED");
 for(const [k,v] of Object.entries(safe)){
   if(v!==null&&!["number","string","boolean"].includes(typeof v))throw Error("UNSUPPORTED_MODEL_VALUE:"+k);
   if(typeof v==="number"&&!Number.isFinite(v))throw Error("NAN_INF:"+k);
 }
 rows.push({ml_dataset_version:ML_DATASET_VERSION,feature_schema_version:ML_FEATURE_SCHEMA_VERSION,
   leakage_policy:"STRICT_PRIOR_DATE_ONLY",race_id:joined.race_id,horse_id:h.horse_id,
   horse_name:h.horse_name,horse_number:h.entry.horse_number,
   prediction_phase:"EARLY",feature_sets:featureSets,history_windows:windows,
   small_sample_policy:policy,
   target:null,market_outcome:null,
   features:safe});
}
if(rows.length!==14||new Set(rows.map(x=>x.horse_id)).size!==14)throw Error("HORSE_COUNT");
rows.sort((a,b)=>a.horse_number-b.horse_number);
const names=Object.keys(rows[0].features).sort().join("|");
if(rows.some(r=>Object.keys(r.features).sort().join("|")!==names))throw Error("MODEL_COLUMN_SCHEMA_DIFF_BY_HORSE");
const race={
 contract:"KEIBA_ONE_RACE_ASOF_V1",status:"FEATURES_PARTIAL_ASOF_MODEL_UNCONFIGURED",
 race_id:"202605040611",race_date:"2026-10-17",phase:"EARLY",
 source_of_truth:"REAL_NETKEIBA_PREBET_PLUS_READONLY_BACKFILL",
 source_pack:null,history_from:"2006-09-02",
 feature_schema_version:ML_FEATURE_SCHEMA_VERSION,
 history:{days:2171,races:joined.summary.cells_checked /* not real race count; overwritten below */},
 asof:{netkeiba_observed_at:source.observed_at_client_utc,netkeiba_sha256:source.source_sha256,
   backfill_latest_race_date:"2026-10-12",not_verified_original_trained_schema:true},
 verified_live_catalog_items:31,verified_history_static_catalog_items:103,
 blocked_due_to_missing_live_info:["weather","track_condition","body_weight","body_weight_diff","LIVE_JRA_ODDS"],
 features_by_horse:rows
};
race.history.races=null; // not recomputed here; never make race-count claim.
const out=path.resolve(process.argv[2]??"/tmp/fuji-original-pr7-asof-input.json");
await mkdir(path.dirname(out),{recursive:true});
await writeFile(out,JSON.stringify(race,null,2)+"\n");
console.log("FUJI_ORIGINAL_L1_INPUT_SHAPE "+JSON.stringify({
 output:out,contract:race.contract,race_id:race.race_id,
 phase:race.phase,horses:rows.length,feature_cols:Object.keys(rows[0].features).length,
 feature_nulls:rows.map(r=>Object.values(r.features).filter(v=>v===null).length),
 original_model_schema_verified:false,trained_model_available:false,
 does_not_contain_results:true,does_not_contain_final_odds:true,
 source_asof:race.asof,
 status:"INPUT_SHAPE_READY_ORIGINAL_MODEL_ARTIFACT_PENDING"}));
