#!/usr/bin/env node
/**
 * Audit-only exact reconstruction of six un-emitted Fuji catalog features.
 * Reuses original approved auto factory functions and the EXACT auto formulae
 * from src/ml-dataset.mjs; reads only 14 horses' PRIOR BACKFILL race records.
 * Does not change original ML modules, does not read future outcomes or odds.
 */
import {readFile,readdir} from "node:fs/promises";
import {gunzipSync} from "node:zlib";
import {fileURLToPath} from "node:url";
import path from "node:path";
import {
  buildRollingFeatureFamily,buildConditionFeatureFamily,
  addFieldRelativeFeatures,buildSafePairFeatures,
} from "../src/auto-feature-factory.mjs";

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"..");
const SOURCE=path.resolve(process.env.BACKFILL_SOURCE_ROOT??"/tmp/keiba-backfill-readonly");
const original=JSON.parse(await readFile(path.join(ROOT,"audit-results/fuji-20261017-backfill-93-10-stage1.json"),"utf8"));
if(original.horses?.length!==14||original.feature_summary?.length!==103)throw new Error("STAGE1_14x103_MISMATCH");
const targets=original.horses, ids=new Set(targets.map(x=>x.horse_id));
const recentByHorse=new Map(targets.map(h=>[h.horse_id,[]]));
const CUTOFF="2026-10-17";
function finite(v){if(v==null||typeof v==="string"&&!v.trim())return null;const n=Number(v);return Number.isFinite(n)?n:null;}
function normType(result){const t=String(result?.margin_type??"").trim().toUpperCase();return t||null;}
function marginLength(result){return normType(result)==="LENGTHS"?finite(result?.margin_lengths):null;}
function smallGap(result){
  const type=normType(result);
  if(type==null||type==="OTHER")return null;
  if(["DEAD_HEAT","NOSE","HEAD","NECK"].includes(type))return true;
  if(type==="LARGE")return false;
  const length=marginLength(result);
  return length==null?null:length<=0.5;
}
function marginTypeRate(items,acceptedTypes){
  const types=items.map(x=>normType(x.result)).filter(Boolean);
  if(!types.length)return null;
  const allow=new Set(acceptedTypes);
  return types.filter(x=>allow.has(x)).length/types.length;
}
// Exact autoHistorySnapshot family semantics copied from frozen ml-dataset.mjs,
// with the original public factory utilities doing every statistical operation.
function autoHistorySnapshot(history,currentRace,limit){
  const recent=history.slice(-limit);
  const finish=recent.map(item=>String(item.result?.result_status??"").toUpperCase()==="FINISHED"
       ?finite(item.result?.official_finish_position):null);
  const last3f=recent.map(item=>finite(item.result?.last_3f));
  const speed=recent.map(item=>{
    const distance=finite(item.race?.distance_m),ms=finite(item.result?.finish_time_ms);
    return distance!=null&&ms!=null&&ms>0?distance/(ms/1000):null;
  });
  const distance=recent.map(item=>finite(item.race?.distance_m));
  const bodyWeight=recent.map(item=>finite(item.entry?.body_weight));
  const carriedWeight=recent.map(item=>finite(item.entry?.carried_weight));
  const lengths=recent.map(item=>marginLength(item.result));
  const types=recent.map(item=>normType(item.result)).filter(Boolean);
  return {
    ...buildRollingFeatureFamily("history_finish_position",finish,{higherIsBetter:false}),
    ...buildRollingFeatureFamily("history_last_3f",last3f,{higherIsBetter:false}),
    ...buildRollingFeatureFamily("history_speed_mps",speed,{higherIsBetter:true}),
    ...buildRollingFeatureFamily("history_distance_m",distance),
    ...buildRollingFeatureFamily("history_body_weight",bodyWeight),
    ...buildRollingFeatureFamily("history_carried_weight",carriedWeight),
    ...buildRollingFeatureFamily("history_margin_lengths",lengths,{higherIsBetter:false}),
    auto_history_margin_type_observation_count:types.length,
    auto_history_margin_lengths_rate:types.length?types.filter(x=>x==="LENGTHS").length/types.length:null,
    auto_history_margin_small_gap_rate:(()=>{
      const items=recent.map(x=>smallGap(x.result)).filter(x=>x===true||x===false);
      return items.length?items.filter(Boolean).length/items.length:null;
    })(),
    auto_history_margin_tight_categorical_rate:marginTypeRate(recent,["DEAD_HEAT","NOSE","HEAD","NECK"]),
    auto_history_margin_large_rate:marginTypeRate(recent,["LARGE"]),
    ...buildConditionFeatureFamily({
      prefix:"margin_gap",history:recent,currentRace,
      valueOf:item=>marginLength(item.result),
      successOf:item=>smallGap(item.result),higherIsBetter:false,
    }),
    ...buildConditionFeatureFamily({
      prefix:"finish",history:recent,currentRace,
      valueOf:item=>finite(item.result?.official_finish_position),
      successOf:item=>{
        const finish=finite(item.result?.official_finish_position);
        return finish==null?null:finish<=3;
      },higherIsBetter:false,
    }),
  };
}
const dayFiles=(await readdir(path.join(SOURCE,"data","daily")))
  .filter(x=>/^\d{4}-\d\d-\d\d\.jsonl\.gz$/.test(x)&&x.slice(0,10)<CUTOFF).sort();
let entriesSeen=0,maxDate=null;
for(const file of dayFiles){
  const txt=gunzipSync(await readFile(path.join(SOURCE,"data/daily",file))).toString("utf8");
  for(const line of txt.split("\n")){
    if(!line.trim())continue;
    const row=JSON.parse(line);
    const date=String(row.race?.actual_date??row.race?.scheduled_date??"").slice(0,10);
    if(!date||date>=CUTOFF)throw new Error("FUTURE_OR_INVALID_ROW:"+date);
    if(maxDate==null||date>maxDate)maxDate=date;
    const results=new Map((row.results??[]).map(x=>[String(x.horse_id),x]));
    for(const e of row.entries??[]){
      const id=String(e.horse_id??"");if(!ids.has(id))continue;
      const r=results.get(id);
      if(!r||["SCRATCHED","EXCLUDED"].includes(String(e.entry_status??"").toUpperCase())||
             ["SCRATCHED","EXCLUDED"].includes(String(r.result_status??"").toUpperCase()))continue;
      const h=recentByHorse.get(id);
      h.push({race:row.race,entry:e,result:r,date});
      if(h.length>20)h.shift();
      entriesSeen++;
    }
  }
}
if(maxDate!==original.backfill_main_latest_race_date)throw new Error("HISTORY_SNAPSHOT_DRIFT:"+maxDate);
const currentRace={
  race_id:"202605040611",venue_code:"05",meeting_no:4,meeting_day:6,
  surface:"TURF",distance_m:1600,direction:"LEFT",course_layout:null,
  track_condition:null,race_class_normalized:"OPEN",grade:"G2",age_min:3,age_max:null,
  age_condition_raw:"3歳以上",weight_rule:"SPECIAL_WEIGHT",
};
const AUTO_FIELD_RELATIVE_SPECS=[
  {key:"network_elo_rating",prefix:"network_elo_rating",direction:"HIGHER_BETTER"},
  {key:"recent_win_rate",prefix:"recent_win_rate",direction:"HIGHER_BETTER"},
  {key:"recent_top3_rate",prefix:"recent_top3_rate",direction:"HIGHER_BETTER"},
  {key:"recent_avg_finish",prefix:"recent_avg_finish",direction:"LOWER_BETTER"},
  {key:"recent_avg_last_3f",prefix:"recent_avg_last_3f",direction:"LOWER_BETTER"},
  {key:"recent_avg_speed_mps",prefix:"recent_avg_speed_mps",direction:"HIGHER_BETTER"},
];
function originalValue(h,key){
  const found=h.features.find(f=>f.key===key);
  if(!found)throw new Error("STAGE1_BASE_KEY_ABSENT:"+h.horse_id+":"+key);
  return found.value;
}
const partial=new Map();
const relativeInput=[];
for(const h of targets){
  const recent=recentByHorse.get(h.horse_id);
  if(!recent.length)throw new Error("NO_PRIOR_HISTORY:"+h.horse_id);
  const auto=autoHistorySnapshot(recent,currentRace,20);
  const currentEntry=h.horse_id;
  const card=JSON.parse(await readFile(path.join(ROOT,"audit-results/fuji-20261017-149-asof-20261016.json"),"utf8"))
    .runners.find(x=>x.horse_id===currentEntry);
  const base={distance_m:1600,body_weight:null,carried_weight:card?.carried_weight};
  const pairs=buildSafePairFeatures({...base,...auto},[
    {a:"distance_m",b:"auto_history_distance_m_mean",prefix:"distance_vs_recent_mean",ops:["diff","ratio","normalized_diff"]},
    {a:"body_weight",b:"auto_history_body_weight_mean",prefix:"body_weight_vs_recent_mean",ops:["diff","ratio","normalized_diff"]},
    {a:"carried_weight",b:"auto_history_carried_weight_mean",prefix:"carried_weight_vs_recent_mean",ops:["diff","ratio","normalized_diff"]},
  ]);
  const last=recent.at(-1);
  partial.set(h.horse_id,{...auto,...pairs,
    history_margin_type:normType(last?.result),
    history_margin_lengths:marginLength(last?.result)});
  relativeInput.push({horse_id:h.horse_id,features:{
    network_elo_rating:originalValue(h,"network_elo_rating"),
    recent_win_rate:originalValue(h,"recent_win_rate"),
    recent_top3_rate:originalValue(h,"recent_top3_rate"),
    recent_avg_finish:originalValue(h,"recent_avg_finish"),
    recent_avg_last_3f:originalValue(h,"recent_avg_last_3f"),
    recent_avg_speed_mps:originalValue(h,"recent_avg_speed_mps"),
  }});
}
const relative=addFieldRelativeFeatures(relativeInput,AUTO_FIELD_RELATIVE_SPECS);
const relativeById=new Map(relative.map(x=>[x.horse_id,x.features]));
const families={
  "auto.rolling_stats":/^auto_history_/,
  "auto.condition_splits":/^auto_(?:margin_gap|finish)_(?:same_|within_)/,
  "auto.field_relative":/^auto_field_/,
  "auto.safe_interactions":/^auto_pair_/,
};
const missing=["auto.rolling_stats","auto.condition_splits","auto.field_relative","auto.safe_interactions","history.margin_type","history.margin_lengths"];
const result={contract:"FUJI_20261017_BACKFILL_SIX_MISSING_V1",
  race_id:"202605040611",source_last_date:maxDate,scan_daily_packs:dayFiles.length,prior_horse_starts:entriesSeen,
  no_future_results:true,no_odds:true,
  horses:[]};
for(const h of targets){
  const id=h.horse_id,computed={...partial.get(id),...relativeById.get(id)};
  const out={horse_id:id,horse_name:h.horse_name,features:[]};
  for(const fid of missing){
    if(families[fid]){
      const keys=Object.keys(computed).filter(k=>families[fid].test(k)).sort();
      const value=Object.fromEntries(keys.map(k=>[k,computed[k]]));
      out.features.push({feature_id:fid,key:"family:"+fid,
        status:keys.length?"BACKFILL_FAMILY_COMPUTED":"FEATURE_FAMILY_NOT_EMITTED",
        value});
    }else{
      const key=fid.replace(".","_");
      out.features.push({feature_id:fid,key,status:computed[key]!=null?"BACKFILL_CALCULATED_VALUE":"BACKFILL_CALCULATED_NULL",
        value:computed[key]??null});
    }
  }
  result.horses.push(out);
}
if(result.horses.length!==14||result.horses.some(h=>h.features.length!==6))throw new Error("AUDIT_SIX_BY_FOURTEEN_INCOMPLETE");
console.log("SIX_SUMMARY "+JSON.stringify({race_id:result.race_id,features:missing,days:dayFiles.length,last_date:maxDate,
  status_counts:Object.fromEntries(missing.map(fid=>[fid,
    Object.fromEntries([...new Set(result.horses.map(h=>h.features.find(f=>f.feature_id===fid).status))]
      .map(status=>[status,result.horses.filter(h=>h.features.find(f=>f.feature_id===fid).status===status).length]))]))}));
console.log("SIX_JSON "+JSON.stringify(result));
