#!/usr/bin/env node
/**
 * 2026-10-17 original Outsider transition history INPUT recovery (not models).
 * Use pinned BACKFILL source (2022-01-01..2026-10-16) and its original
 * backfill-feature-adapter.mjs v2. For each of 471 target runners, reconstruct
 * only the base facts used by ORIGINAL build_l1_nonhorse_dataset_v1.py's
 * horse_signature(). Historical result_status is used solely to identify
 * prior eligible starts; position/target/odds/payout is NEVER copied.
 */
import fs from "node:fs";
import path from "node:path";
import {gunzipSync} from "node:zlib";
import {createHash} from "node:crypto";
import {pathToFileURL} from "node:url";
const DAY="2026-10-17";
const START="2022-01-01";
const END="2026-10-16";
const PREFIX="ml_features/delivery/690/";
const rawKeys=[
 "race_date","venue_code","surface","direction","backfill_course_layout",
 "distance_m","backfill_race_class_normalized","backfill_grade",
 "carried_weight","body_weight"
];
function insist(x,msg){if(!x)throw Error(msg);}
function arg(name){const i=process.argv.indexOf(name);insist(i>=0&&process.argv[i+1],"MISSING_ARG_"+name);return process.argv[i+1];}
function sha(p){return createHash("sha256").update(fs.readFileSync(p)).digest("hex");}
function parseGzip(file){
  return gunzipSync(fs.readFileSync(file)).toString("utf8").split("\n").filter(Boolean).map(JSON.parse);
}
function eligible(e,result){
  if(!result)return false;
  return !["SCRATCHED","EXCLUDED"].includes(String(e.entry_status??"").toUpperCase()) &&
    !["SCRATCHED","EXCLUDED"].includes(String(result.result_status??"").toUpperCase());
}
function numeric(x){
  if(x===null||x===undefined||x==="")return null;
  const n=Number(x);return Number.isFinite(n)?n:null;
}
async function main(){
  const root=arg("--pinned-backfill-root"), output=arg("--output");
  insist(!fs.existsSync(output),"NO_PRIOR_SIGNATURE_OVERWRITE");
  const imp=await import(pathToFileURL(path.join(root,"ml_features/src/backfill-feature-adapter.mjs")).href);
  insist(imp.BACKFILL_FEATURE_ADAPTER_VERSION===2 &&
         typeof imp.readBackfillRaceFeatures==="function","ORIGINAL_BACKFILL_V2_REQUIRED");
  const idx=JSON.parse(fs.readFileSync(path.join(root,PREFIX,"index.json"),"utf8"));
  insist(idx.contract==="KEIBA_BACKFILL_TO_ML_690_INDEX_V1" &&
         idx.feature_count===690&&idx.total_races===60&&idx.total_horses===786&&
         idx.readiness==="ASOF_UNVERIFIED_DO_NOT_BET","PINNED_690_INDEX_DRIFT");
  const raceMap=Object.entries(idx.races).filter(([,x])=>x.date===DAY);
  insist(raceMap.length===36&&raceMap.reduce((a,[,x])=>a+x.starters,0)===471,
         "PINNED_690_36_471_SCOPE_DRIFT");
  const targets=new Map(),ids=new Set();
  for(const [rid,v] of raceMap){
    const p=path.join(root,v.path);
    insist(sha(p)===v.sha256,"ORIGINAL_690_SHA_DRIFT:"+rid);
    const json=JSON.parse(fs.readFileSync(p,"utf8"));
    insist(json.race_id===rid&&json.feature_count===690&&
           json.used_target_results_or_final_odds===false,"ORIGINAL_690_TARGET_META_DRIFT");
    for(const row of json.rows){
      const hid=String(row._horse_id??""),key=rid+"|"+hid;
      const starts=Number(row.prior_starts);
      insist(Number.isInteger(starts)&&starts>=0&&/^\d{8,16}$/.test(hid)&&
             !targets.has(key),"ORIGINAL_690_PRIOR_STARTS_OR_ID_DRIFT");
      targets.set(key,{horse_id:hid,prior_starts:starts,
                       previous_distance_m:numeric(row.previous_distance_m)});
      ids.add(hid);
    }
  }
  insist(targets.size===471,"ORIGINAL_TARGET_471_REQUIRED");
  const daily=path.join(root,"data/daily");
  const files=fs.readdirSync(daily).filter(n=>/^\d{4}-\d\d-\d\d\.jsonl\.gz$/.test(n)&&
    n.slice(0,10)>=START&&n.slice(0,10)<=END).sort();
  insist(files.length===517&&files.filter(n=>n.startsWith("2026-")).length===87,
         "ORIGINAL_2022_2026_STRICT_PRIOR_DAYS_NOT_517");
  const counts=new Map([...ids].map(h=>[h,0]));
  const lastByHorse=new Map();
  const priorPerYear={};
  let rowsSeen=0,stored=0,nonNullWeight=0;
  for(const file of files){
    const date=file.slice(0,10);
    insist(date<DAY,"TARGET_DATE_HISTORY_ACCESS_FORBIDDEN");
    const races=parseGzip(path.join(daily,file))
      .sort((a,b)=>String(a.race?.race_id??"").localeCompare(String(b.race?.race_id??"")));
    const oneDay=new Set();
    for(const row of races){
      const actual=String(row.race?.actual_date??row.race?.scheduled_date??"").slice(0,10);
      insist(actual===date,"PRIOR_RACE_DATE_METADATA_MISMATCH");
      const entries=row.entries??[];
      const map=new Map((row.results??[]).map(z=>[String(z.horse_id??""),z]));
      if(!entries.some(e=>ids.has(String(e.horse_id??""))))continue;
      const derived=imp.readBackfillRaceFeatures(row.race??{},entries);
      for(const e of entries){
        const hid=String(e.horse_id??"");
        if(!ids.has(hid)||!eligible(e,map.get(hid)))continue;
        insist(!oneDay.has(hid),"HORSE_DUPLICATE_SAME_DAY_PRIOR_SIGNATURE");
        oneDay.add(hid);
        const record={
          "race_date":date,
          "venue_code":row.race?.venue_code??null,
          "surface":row.race?.surface??null,
          "direction":row.race?.direction??null,
          "backfill_course_layout":derived.features.backfill_course_layout,
          "distance_m":numeric(row.race?.distance_m),
          "backfill_race_class_normalized":derived.features.backfill_race_class_normalized,
          "backfill_grade":derived.features.backfill_grade,
          "carried_weight":numeric(e.carried_weight),
          "body_weight":numeric(e.body_weight),
        };
        insist(Object.keys(record).length===10 &&
               rawKeys.every(k=>Object.hasOwn(record,k)),"ORIGINAL_HORSE_SIGNATURE_SOURCE_DRIFT");
        if(record.body_weight!==null)nonNullWeight++;
        const arr=lastByHorse.get(hid)??[];
        arr.push(record);
        if(arr.length>3)arr.splice(0,arr.length-3);
        lastByHorse.set(hid,arr);
        counts.set(hid,counts.get(hid)+1);
        priorPerYear[date.slice(0,4)]=(priorPerYear[date.slice(0,4)]??0)+1;
        stored++;rowsSeen++;
      }
    }
  }
  let previousDistanceMatches=0,previousDistanceUncheckable=0;
  const rows=[];
  for(const [key,item] of [...targets].sort(([a],[b])=>a.localeCompare(b))){
    const observed=counts.get(item.horse_id)??0;
    insist(observed===item.prior_starts,
      "PRIOR_HORSE_SIGNATURE_VS_690_PRIOR_STARTS_MISMATCH:"+key+
      ":actual="+observed+":expected="+item.prior_starts);
    const history=lastByHorse.get(item.horse_id)??[];
    insist(history.length===Math.min(3,observed),"PRIOR_HORSE_LAST3_NOT_COMPLETE:"+key);
    if(item.previous_distance_m!==null&&history.length){
      insist(item.previous_distance_m===history.at(-1).distance_m,
             "PRIOR_HORSE_SIGNATURE_DISTANCE_PARITY_MISMATCH:"+key);
      previousDistanceMatches++;
    }else previousDistanceUncheckable++;
    rows.push({
      "contract":"KEIBA_2026_10_17_ORIGINAL_OUTSIDER_5_PRIOR_HORSE_SIGNATURE_SOURCE_V1",
      "race_id":key.split("|")[0],"horse_id":item.horse_id,
      "strict_prior_source_dates":[START,END],
      "observed_prior_starts":observed,
      "last_three_records":history,
      "source_is_original_pinned_backfill":true,
      "2026_target_results_used":false,
      "2026_historical_results_used_for_start_status_only":true,
      "model_ready":false,"ready_for_betting":false,
    });
  }
  insist(rows.length===471,"ORIGINAL_PRIOR_SIGNATURE_471_UNIVERSE_DRIFT");
  fs.mkdirSync(path.dirname(output),{recursive:true});
  fs.writeFileSync(output,rows.map(z=>JSON.stringify(z)).join("\n")+"\n");
  console.log("ORIGINAL_2026_OUTSIDER_FIVE_TRANSITION_SIGNATURES_SOURCE_PASS",JSON.stringify({
    races:36,horses:471,historical_files:files.length,prior_2026_files:87,
    strictly_prior_starts_seen:stored,prior_observations_by_year:priorPerYear,
    source_body_weight_nonnull:nonNullWeight,
    previous_distance_verifiable:previousDistanceMatches,
    previous_distance_not_verifiable:previousDistanceUncheckable,
    original_2026_target_result_used:false,
    prior_features_actual_scoring_done:false,
    all_13_training_done:false,
  }));
}
main().catch(e=>{console.error(e);process.exitCode=1});
