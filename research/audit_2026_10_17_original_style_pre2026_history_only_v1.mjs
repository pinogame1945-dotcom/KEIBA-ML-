#!/usr/bin/env node
/**
 * 2026-10-17 STYLE: original historical-extra-builder.mjs only.
 *
 * STRICT 2026 outcome seal: read dated BACKFILL results ONLY through 2025-12-31.
 * No 2026 daily result file is ever opened, and no 2026 racing outcome is used.
 * This means the STYLE signal is COMPLETE only for target horses whose actual
 * cached prior_starts equals the amount of old (<2026) source history observed.
 * Everyone else is explicitly BLOCKED, never assigned misleading STYLE values.
 *
 * Existing original 690 and king weights are never recomputed/touched. The
 * result is diagnostic and not approved for 13 Outsider scoring or betting.
 */
import fs from "node:fs";
import path from "node:path";
import {gunzipSync} from "node:zlib";
import {pathToFileURL} from "node:url";
import {createHash} from "node:crypto";

const TARGET="2026-10-17";
const HIST_CEILING="2025-12-31";
const EXPECTED_RACES=new Set(["04:01","05:06","08:06"].flatMap(([v,d])=>[]));
// Deliberately write the expected 36 race IDs explicitly by JRA venue/day.
const RACE_IDS=new Set([["04","01"],["05","06"],["08","06"]].flatMap(
  ([venue,day])=>Array.from({length:12},(_,i)=>"2026"+venue+"04"+day+String(i+1).padStart(2,"0"))
));
const STYLE_KEYS=[
  "style_previous_first_position","style_previous_last_position",
  "style_previous_position_gain","style_previous_first_ratio","style_previous_last_ratio",
  "style_recent_races_measured","style_recent_avg_first_position",
  "style_recent_avg_last_position","style_recent_avg_position_gain",
  "style_recent_avg_first_ratio","style_recent_avg_last_ratio",
  "style_recent_front_rate","style_recent_finish_front_rate","style_recent_improve_rate"
];
const INDEX="ml_features/delivery/690/index.json";
const SOURCE_PREFIX="ml_features/delivery/690/2026-10-17/";
const CAPTURE_CEILING=Date.parse("2026-10-16T15:00:00Z");
function require_(x,msg){if(!x)throw Error(msg);}
function arg(name){
  const i=process.argv.indexOf(name);
  require_(i>=0&&process.argv[i+1]&&!process.argv[i+1].startsWith("--"),"MISSING_ARG:"+name);
  return process.argv[i+1];
}
function fileSha(p){return createHash("sha256").update(fs.readFileSync(p)).digest("hex");}
function rows(file){
  const data=fs.readFileSync(file);
  const txt=file.endsWith(".gz")?gunzipSync(data).toString("utf8"):data.toString("utf8");
  return txt.split("\n").filter(s=>s.trim()).map(JSON.parse);
}
function fieldSize(row){
  const results=new Map((row.results||[]).map(r=>[String(r.horse_id??""),r]));
  return (row.entries||[]).filter(e=>{
    const es=String(e.entry_status??"").toUpperCase();
    const result=results.get(String(e.horse_id??""));
    const rs=String(result?.result_status??"").toUpperCase();
    return !!result&&!["SCRATCHED","EXCLUDED"].includes(es)&&!["SCRATCHED","EXCLUDED"].includes(rs);
  }).length;
}
function historicEligible(row,hid){
  const e=(row.entries||[]).find(e=>String(e.horse_id??"")===hid);
  const res=(row.results||[]).find(r=>String(r.horse_id??"")===hid);
  if(!e||!res)return false;
  return !["SCRATCHED","EXCLUDED"].includes(String(e.entry_status??"").toUpperCase())&&
    !["SCRATCHED","EXCLUDED"].includes(String(res.result_status??"").toUpperCase());
}
function loadTarget690(root){
  const index=JSON.parse(fs.readFileSync(path.join(root,INDEX),"utf8"));
  require_(index.contract==="KEIBA_BACKFILL_TO_ML_690_INDEX_V1"&&
       index.feature_count===690&&index.total_races===60&&index.total_horses===786&&
       index.readiness==="ASOF_UNVERIFIED_DO_NOT_BET","FROZEN_690_INDEX_INVALID");
  const selected=Object.entries(index.races).filter(([,e])=>e.date===TARGET);
  require_(selected.length===36&&selected.reduce((a,[,e])=>a+e.starters,0)===471,
           "2026_LIVE_TARGET_690_UNIVERSE_MISMATCH");
  const ids=new Set(),sources=new Map(),seen=new Set();
  for(const [rid,e] of selected){
    require_(RACE_IDS.has(rid)&&e.path===SOURCE_PREFIX+rid+".json"&&
             !seen.has(rid),"LIVE_TARGET_690_RACE_DRIFT");
    seen.add(rid);
    const p=path.join(root,e.path);
    require_(fileSha(p)===e.sha256,"LIVE_TARGET_690_SHA_MISMATCH:"+rid);
    const doc=JSON.parse(fs.readFileSync(p,"utf8"));
    require_(doc.contract==="KEIBA_BACKFILL_TO_ML_690_UNSCORED_V1"&&
             doc.race_date===TARGET&&doc.race_id===rid&&doc.starters===e.starters&&
             doc.feature_count===690&&doc.readiness==="ASOF_UNVERIFIED_DO_NOT_BET"&&
             doc.used_target_results_or_final_odds===false,"LIVE_TARGET_690_CONTRACT_DRIFT");
    for(const z of doc.rows){
      const hid=String(z._horse_id??""),key=rid+"|"+hid;
      const n=Number(z.prior_starts);
      require_(z._race_id===rid&&z._race_date===TARGET&&/^\d{8,16}$/.test(hid)&&
             Number.isInteger(n)&&n>=0&&!sources.has(key),"LIVE_TARGET_HORSE_OR_PRIOR_COUNT_DRIFT:"+key);
      sources.set(key,{prior_starts:n,horse_id:hid,source_race_sha256:e.sha256});
      ids.add(hid);
    }
  }
  require_(seen.size===36&&sources.size===471,"LIVE_SOURCE_EXPECTED_36_471");
  return {ids,sources};
}
function loadPreRaceDayPack(file,sources){
  const arr=rows(file);
  require_(arr.length===36,"PRE_RACE_PACK_NOT_36");
  const races=new Set(),population=new Set();
  for(const row of arr){
    const rid=String(row.race?.race_id??""),date=row.race?.scheduled_date;
    require_(RACE_IDS.has(rid)&&!races.has(rid)&&date===TARGET&&row.race.actual_date==null,
             "PRE_RACE_PACK_RACE_DRIFT");
    const stamp=Date.parse(row.live_source_observed_at);
    require_(Number.isFinite(stamp)&&stamp<CAPTURE_CEILING,
             "PRE_RACE_CAPTURE_NOT_BEFORE_DAY");
    for(const k of ["results","payouts","odds","laps","corners"])
      require_(Array.isArray(row[k])&&row[k].length===0,
               "2026_TARGET_OUTCOME_OR_ODDS_PRESENT:"+k);
    for(const e of row.entries||[]){
      if(e.entry_status!=="STARTED")continue;
      const key=rid+"|"+String(e.horse_id??"");
      require_(sources.has(key)&&!population.has(key),"TARGET_HORSE_NOT_IN_PINNED_690:"+key);
      population.add(key);
    }
    races.add(rid);
  }
  require_(races.size===36&&population.size===471,"TARGET_PACK_36_471_MISMATCH");
  return arr;
}
async function main(){
  const root=arg("--pinned-backfill-root");
  const pack=arg("--pre-race-day-pack");
  const out=arg("--out");
  require_(!fs.existsSync(out),"STYLE_RECOVERY_REFUSE_OVERWRITE");
  const generator=path.join(root,"ml_features/src/historical-extra-builder.mjs");
  const cfg=JSON.parse(fs.readFileSync(path.join(root,"ml_features/contracts/l1-feature-set-contract-v1.json"),"utf8"));
  require_(cfg.history_windows.style_last3f===10&&cfg.history_windows.suitability===20,
           "ORIGINAL_STYLE_HISTORY_WINDOW_DRIFT");
  const {createHistoricalExtraProcessor}=await import(pathToFileURL(generator).href);
  require_(typeof createHistoricalExtraProcessor==="function","ORIGINAL_HISTORICAL_STYLE_BUILDER_MISSING");
  const {ids,sources}=loadTarget690(root);
  const target=loadPreRaceDayPack(pack,sources);
  const processor=createHistoricalExtraProcessor(cfg.history_windows);
  const dir=path.join(root,"data/daily");
  const names=fs.readdirSync(dir).filter(name=>/^\d{4}-\d{2}-\d{2}\.jsonl\.gz$/.test(name)&&
                name.slice(0,10)<=HIST_CEILING).sort();
  require_(names.length>1800&&names[0].startsWith("2006-"),
           "ORIGINAL_HISTORY_SOURCE_NOT_COMPLETE_PRE_2026");
  const targetCounts=new Map([...ids].map(x=>[x,0]));
  let targetHistoryRaces=0,historyFiles=0,historicalObservations=0;
  for(const name of names){
    const date=name.slice(0,10);
    require_(date<TARGET&&date<="2025-12-31","SEALED_2026_RESULT_ACCESS_FORBIDDEN");
    const old=rows(path.join(dir,name));
    const filtered=[];
    for(const row of old){
      const actual=String(row?.race?.actual_date??row?.race?.scheduled_date??"").slice(0,10);
      require_(actual===date,"HISTORICAL_SOURCE_DATE_MISMATCH:"+name);
      if(!(row.entries||[]).some(e=>ids.has(String(e.horse_id??""))))continue;
      filtered.push(row);
      targetHistoryRaces++;
      for(const e of row.entries||[]){
        const hid=String(e.horse_id??"");
        if(ids.has(hid)&&historicEligible(row,hid)){
          targetCounts.set(hid,targetCounts.get(hid)+1);
          historicalObservations++;
        }
      }
    }
    if(filtered.length)processor.processDay(filtered,date,{startDate:TARGET,endDate:TARGET});
    historyFiles++;
  }
  const generated=processor.processDay(target,TARGET,{
    startDate:TARGET,endDate:TARGET,allowUnresultedTargets:true
  });
  require_(generated.size===471,"ORIGINAL_STYLE_BUILDER_NOT_471");
  const expectedStyle=new Set(STYLE_KEYS);
  let full=0,blocked=0,coveredWithStyle=0,measured=0,disagreements=0;
  const report=[];
  for(const [key,src] of [...sources.entries()].sort(([a],[b])=>a.localeCompare(b))){
    const actualCount=targetCounts.get(src.horse_id)||0;
    const expectedCount=src.prior_starts;
    const style=generated.get(key);
    require_(style&&expectedStyle.size===14&&
             [...expectedStyle].every(k=>Object.hasOwn(style,k))&&
             Object.keys(style).filter(k=>k.startsWith("style_")).length===14,
             "ORIGINAL_STYLE_14_COLUMN_SCHEMA_MISMATCH:"+key);
    // If 2026 older-race outcomes are excluded, a mismatch is a hard stop for
    // that horse's STYLE values (not replaced with zero/null as if measured).
    const complete=actualCount===expectedCount;
    if(actualCount>expectedCount)disagreements++;
    if(complete){full++;coveredWithStyle+=Object.values(style).filter(v=>v!==null).length;
                if(style.style_recent_races_measured>0)measured++;}
    else blocked++;
    report.push({
      race_id:key.split("|")[0],horse_id:src.horse_id,
      source_690_sha256:src.source_race_sha256,
      prior_starts_690:expectedCount,pre_2026_prior_races_observed:actualCount,
      original_style_values_complete_without_2026_outcomes:complete,
      style_complete:complete?Object.fromEntries(STYLE_KEYS.map(k=>[k,style[k]])):null,
      excluded_2026_history_if_present:!complete,
      model_ready:false,asof_full_features_verified:false,ready_for_betting:false
    });
  }
  require_(full+blocked===471&&disagreements===0,"HISTORICAL_PRIOR_COUNT_SOURCE_DRIFT");
  fs.mkdirSync(path.dirname(out),{recursive:true});
  fs.writeFileSync(out,report.map(x=>JSON.stringify(x)).join("\n")+"\n");
  console.log("ORIGINAL_2026_10_17_OUTSIDER_STYLE_PRE2026_HISTORY_ONLY_AUDIT",JSON.stringify({
    target_races:36,target_horses:471,historical_daily_files_read:historyFiles,
    historical_races_with_target_horses:targetHistoryRaces,
    pre2026_prior_race_observations:historicalObservations,
    certified_old_only_horse_history_count:full,
    blocked_incomplete_history_count:blocked,
    complete_horses_with_measured_corner_history:measured,
    original_style_feature_count:14,
    total_non_null_style_cells_on_complete_horses:coveredWithStyle,
    any_2026_outcomes_accessed:false,
    original_style_generator_unchanged:true,
    target_model_input_ready:false,
    result_path:out
  }));
}
main().catch(e=>{console.error(e);process.exitCode=1});
