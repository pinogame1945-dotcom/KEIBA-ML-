#!/usr/bin/env node
/**
 * 2026-10-17 Fuji Stakes 14 horses ONLY: 93 prior-date features + 10 static features.
 * No scrape, no 2026-10-17 results, no odds, no model training or inference.
 * Uses unchanged official KEIBA-ML point-in-time feature builders and read-only
 * KEIBA-BACKFILL historical daily and horse profile packs.
 */
import {readFile, readdir} from "node:fs/promises";
import {gunzipSync} from "node:zlib";
import {createHash} from "node:crypto";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {createMlDatasetProcessor} from "../src/ml-dataset.mjs";
import {createHistoricalExtraProcessor} from "../src/historical-extra-builder.mjs";
import {L1_FEATURE_SET_CONTRACT} from "../src/l1-feature-contract.mjs";
import {lineageFromHorseRecord} from "../src/pedigree-feature-builder.mjs";

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"..");
const SOURCE=path.resolve(process.env.BACKFILL_SOURCE_ROOT??"/tmp/keiba-backfill-readonly");
const RID="202605040611", TARGET="2026-10-17", CUTOFF="2026-10-16";
const page=JSON.parse(await readFile(path.join(ROOT,"audit-results/fuji-20261017-149-asof-20261016.json"),"utf8"));
const catalog=JSON.parse(await readFile(path.join(ROOT,"contracts/l1-feature-catalog-v1.json"),"utf8"));
const targets=page.runners??[], ids=new Set(targets.map(x=>String(x.horse_id)));
if(targets.length!==14||ids.size!==14||page.race_id!==RID)throw new Error("FUJI_BASELINE_MISMATCH");
const selected=catalog.features.filter(f=>["STRICT_PRIOR_DATE_ONLY","STATIC_FACT"].includes(f.history_boundary));
if(selected.length!==103||selected.filter(f=>f.history_boundary==="STRICT_PRIOR_DATE_ONLY").length!==93||selected.filter(f=>f.history_boundary==="STATIC_FACT").length!==10)throw new Error("FEATURE_CATALOG_103_SCOPE_DRIFT");

function dateOf(row) {
  return String(row?.race?.actual_date??row?.race?.scheduled_date??"").slice(0,10);
}
function unpack(bytes){return gunzipSync(bytes).toString("utf8").split("\n").filter(Boolean).map(JSON.parse);}
function scalar(v){return v==null||typeof v==="number"||typeof v==="boolean"||typeof v==="string";}
const ancestry=new Map(), profiles=new Map(), seenPackFiles=[];
const horsesDir=path.join(SOURCE,"data/horses");
const horseFiles=(await readdir(horsesDir)).filter(x=>/^horse-\d{4}-\d{2}-\d{2}-\d+\.jsonl\.gz$/.test(x)).sort();
let horsePackRecords=0;
for(let i=0;i<horseFiles.length;i++){
  const file=horseFiles[i], records=unpack(await readFile(path.join(horsesDir,file)));
  for(const rec of records){
    if(rec?.kind!=="horse"||!rec.horse_id)continue;
    const id=String(rec.horse_id);
    if(!ancestry.has(id))ancestry.set(id,lineageFromHorseRecord(rec));
    if(ids.has(id)&&!profiles.has(id))profiles.set(id,{...rec,source_pack:file});
    horsePackRecords++;
  }
  if((i+1)%600===0)console.error("HORSE_PACK_PROGRESS",i+1,horseFiles.length,"profiles",profiles.size);
}
console.error("HORSE_PACK_DONE",JSON.stringify({files:horseFiles.length,records:horsePackRecords,distinct_lineage:ancestry.size,target_profiles:profiles.size}));

const windows=L1_FEATURE_SET_CONTRACT.history_windows;
const ml=createMlDatasetProcessor({
  historyWindows:windows,
  includeAutoFeatures:false, // auto family not falsely represented as calculated
  includeBackfillFeatures:false,
  includePedigreeFeatures:true,
  includeActorFeatures:true,
  includeOpponentRelationshipFeatures:true,
  includeTimePaceFeatures:true,
  lineageByHorse:ancestry,
});
const extras=createHistoricalExtraProcessor(windows);
const dir=path.join(SOURCE,"data/daily");
const daily=(await readdir(dir)).filter(x=>/^\d{4}-\d{2}-\d{2}\.jsonl\.gz$/.test(x)&&x.slice(0,10)<TARGET).sort();
if(!daily.length)throw new Error("BACKFILL_DAILY_PACKS_MISSING");
let raceCount=0, entryCount=0, currentHorseStarts=new Map(targets.map(x=>[String(x.horse_id),0])), maxSourceDate=null, sourceDays=0;
let invalidFuture=0, daySeen=new Set();
for(let i=0;i<daily.length;i++){
  const file=daily[i], rows=unpack(await readFile(path.join(dir,file)));
  const grouped=new Map();
  for(const r of rows){
    const date=dateOf(r);
    if(!date||date>TARGET)throw new Error("SOURCE_INVALID_OR_FUTURE_DATE:"+date);
    if(date===TARGET){invalidFuture++;continue;}
    const group=grouped.get(date)??[];group.push(r);grouped.set(date,group);
    raceCount++;entryCount+=(r.entries??[]).length;
    for(const e of r.entries??[])if(ids.has(String(e.horse_id)))currentHorseStarts.set(String(e.horse_id),currentHorseStarts.get(String(e.horse_id))+1);
  }
  for(const [date,rs] of [...grouped].sort(([a],[b])=>a.localeCompare(b))){
    if(daySeen.has(date))throw new Error("DUPLICATE_HISTORY_DATE:"+date);
    daySeen.add(date);
    ml.processDay(rs,date,{startDate:TARGET,endDate:TARGET});
    extras.processDay(rs,date,{startDate:TARGET,endDate:TARGET});
    if(maxSourceDate==null||date>maxSourceDate)maxSourceDate=date;
    sourceDays++;
  }
  if((i+1)%300===0)console.error("DAILY_PROGRESS",i+1,daily.length,"races",raceCount,"rows",entryCount,"latest",maxSourceDate);
}
if(invalidFuture)throw new Error("FUTURE_RACE_DATA_APPEARED");
console.error("HISTORY_LOADED",JSON.stringify({days:sourceDays,races:raceCount,entries:entryCount,last_date:maxSourceDate,activeHorseStarts:Object.fromEntries(currentHorseStarts)}));

const race={
  race_id:RID,actual_date:TARGET,scheduled_date:TARGET,venue_code:"05",meeting_no:4,meeting_day:6,race_no:11,
  discipline:"FLAT",surface:"TURF",distance_m:1600,direction:"LEFT",weather:null,track_condition:null,
  course_layout:null,course_laps:null,race_class_normalized:"OPEN",grade:"G2",age_min:3,age_max:null,
  age_condition_raw:"3歳以上",weight_rule:"SPECIAL_WEIGHT",international:true,designated:true,field_size:14
};
const entryRows=targets.map(t=>({horse_id:t.horse_id,horse_name:t.horse_name,gate:t.gate,
  horse_number:t.horse_number,sex:t.sex,age:t.age,carried_weight:t.carried_weight,
  jockey_id:t.jockey_id,trainer_id:t.trainer_id,entry_status:"STARTED"}));
const synthetic={race,entries:entryRows,
  // Only to pass historical TRAINING dataset builder's existence gate. Neither
  // finish nor odds is invented; all synthetic target/market fields discarded.
  results:entryRows.map(e=>({horse_id:e.horse_id,result_status:"PENDING",official_finish_position:null})),laps:[]};
const generated=ml.processDay([synthetic],TARGET,{startDate:TARGET,endDate:TARGET});
const extraMap=extras.processDay([synthetic],TARGET,{startDate:TARGET,endDate:TARGET});
if(generated.length!==14||new Set(generated.map(r=>String(r.horse_id))).size!==14||extraMap.size!==14)
  throw new Error("FUTURE_ASOF_FEATURE_ROWS_MISMATCH:"+generated.length+"/"+extraMap.size);

const horseIndex=new Map(generated.map(row=>[String(row.horse_id),{
  ...row.features,...(extraMap.get(RID+"|"+row.horse_id)??{})
}]));
const mapping={
  "jockey.win_rate":"actor_jockey_all_win_rate",
  "jockey.top3_rate":"actor_jockey_all_top3_rate",
  "jockey.venue_score":"actor_jockey_venue_effective_top3_rate",
  "jockey.distance_score":"actor_jockey_distance_band_effective_top3_rate",
  "jockey.surface_score":"actor_jockey_surface_effective_top3_rate",
  "horse_jockey.starts":"actor_horse_jockey_all_starts",
  "horse_jockey.top3_rate":"actor_horse_jockey_all_top3_rate",
  "trainer.win_rate":"actor_trainer_all_win_rate",
  "trainer.top3_rate":"actor_trainer_all_top3_rate",
  "trainer.recent_form":"actor_trainer_recent_effective_top3_rate"
};
const families={
  "opponent.strong_opponent_performance":/^opponent_strong_/,
  "opponent.beaten_strength":/^opponent_strong_opponent_beat_rate$/,
  "opponent.relationship_v2":/^opponent_(relationship_|history_|strong_|current_field_|close_)/,
  "pedigree.point_in_time_stats_v1":/^ped_/,
  "actor.point_in_time_stats_v1":/^actor_/,
  "time_pace.normalized_v1":/^timepace_/,
  "auto.rolling_stats":/^auto_(history_|rolling_)/,
  "auto.condition_splits":/^auto_.*_(same_|within_)/,
  "auto.field_relative":/^auto_field_/,
  "auto.safe_interactions":/^auto_pair_/
};
const ancestorSlots={
  "pedigree.sire_id":[1,0],"pedigree.dam_id":[1,1],
  "pedigree.siresire_id":[2,0],"pedigree.damsire_id":[2,2]
};
const profileFields={
  "horse.birth_date":"birth_date","horse.coat_color":"coat_color","horse.birthplace":"birthplace",
  "horse.owner_id":"owner_id","horse.breeder_id":"breeder_id"
};
function keyOf(fid){
  if(mapping[fid])return mapping[fid];
  if(fid.startsWith("history."))return fid.slice("history.".length);
  if(fid.startsWith("opponent."))return "opponent_"+fid.slice(9);
  if(fid.startsWith("network."))return "network_"+fid.slice(8);
  if(fid.startsWith("lap."))return "lap_"+fid.slice(4);
  if(fid.startsWith("style."))return "style_"+fid.slice(6);
  if(fid.startsWith("distance."))return "distx_"+fid.slice(9);
  return null;
}
function featureValue(f, id){
  const fid=f.feature_id,feat=horseIndex.get(id)??{},p=profiles.get(id);
  const base={feature_id:fid,history_boundary:f.history_boundary,per_horse_id:id,
    source:"KEIBA_BACKFILL",value:null,key:null,status:null,
    reason:null};
  if(f.history_boundary==="STATIC_FACT"){
    if(!p){base.status="HORSE_PROFILE_PACK_MISSING";base.reason="No existing BACKFILL horse record found";return base;}
    if(fid==="pedigree.known_nodes"){
      const nodes=p.pedigree??[];base.key="pedigree.nodes";base.value=nodes.filter(n=>n.ancestor_name||n.ancestor_id).length;
    } else if(ancestorSlots[fid]){
      const [gen,slot]=ancestorSlots[fid];const n=(p.pedigree??[]).find(x=>Number(x.generation)===gen&&Number(x.slot)===slot);
      base.key="pedigree.generation_slot";base.value=n?.ancestor_id??null;
    }else if(profileFields[fid]){
      base.key="profile."+profileFields[fid];base.value=p.profile?.[profileFields[fid]]??null;
    }else{base.status="UNMAPPED_STATIC_CATALOG_ID";return base;}
    base.status=base.value!=null?"BACKFILL_STATIC_VALUE_FOUND":"BACKFILL_STATIC_NULL";
    return base;
  }
  if(families[fid]){
    const keys=Object.keys(feat).filter(k=>families[fid].test(k)).sort();
    if(!keys.length){base.status="FEATURE_FAMILY_NOT_EMITTED";base.reason="Original unchanged builder returned no keys for this catalog family";return base;}
    const obj=Object.fromEntries(keys.map(k=>[k,feat[k]]));
    base.value=obj;base.key="family:"+fid;base.status=Object.values(obj).some(v=>v!=null)?"BACKFILL_FAMILY_COMPUTED":"BACKFILL_FAMILY_ALL_NULL";return base;
  }
  const k=keyOf(fid);
  if(!k){base.status="UNMAPPED_CATALOG_ID";base.reason="No 1-to-1 original feature key mapping verified";return base;}
  base.key=k;
  if(!Object.hasOwn(feat,k)){base.status="FEATURE_NOT_EMITTED";base.reason="Requested feature key not present in original builder output";return base;}
  const v=feat[k];
  if(!scalar(v)){base.status="INVALID_NONSCALAR";return base;}
  base.value=v;base.status=v==null?"BACKFILL_CALCULATED_NULL":"BACKFILL_CALCULATED_VALUE";
  return base;
}
const results={contract:"FUJI_20261017_BACKFILL_93_10_ASOF_V1",race_id:RID,race_date:TARGET,
  collected_at_utc:new Date().toISOString(),strict_prior_date_cutoff:CUTOFF,
  backfill_main_latest_race_date:maxSourceDate,source_daily_packs:daily.length,
  source_races:raceCount,source_entries:entryCount,
  horse_pack_files:horseFiles.length,horse_pack_records:horsePackRecords,
  unique_pedigrees:ancestry.size,
  output_count:generated.length,temporary_pending_results_for_feature_gate:true,
  actual_current_race_results_consumed:false,market_odds_consumed:false,
  model_training:false,scope_catalog_features:{history:93,static:10,total:103},
  horses:[],feature_summary:[]};
const byFeature=new Map(selected.map(f=>[f.feature_id,[]]));
for(const t of targets){
  const id=String(t.horse_id);const profile=profiles.get(id)??null;
  const item={horse_id:id,horse_name:t.horse_name,prior_entries_in_backfill:currentHorseStarts.get(id)??0,
    horse_profile_pack:profile?.source_pack??null,features:[]};
  for(const f of selected){const row=featureValue(f,id);item.features.push(row);byFeature.get(f.feature_id).push(row);}
  results.horses.push(item);
}
for(const f of selected){
  const rows=byFeature.get(f.feature_id),statuses={};
  for(const v of rows)statuses[v.status]=(statuses[v.status]??0)+1;
  results.feature_summary.push({feature_id:f.feature_id,history_boundary:f.history_boundary,
    status_counts:statuses,numeric_present:rows.filter(x=>x.status==="BACKFILL_CALCULATED_VALUE"||x.status==="BACKFILL_STATIC_VALUE_FOUND").length,
    family_computed:rows.filter(x=>x.status==="BACKFILL_FAMILY_COMPUTED").length,
    unresolved:rows.filter(x=>["UNMAPPED_CATALOG_ID","FEATURE_NOT_EMITTED","FEATURE_FAMILY_NOT_EMITTED","HORSE_PROFILE_PACK_MISSING"].includes(x.status)).length});
}
if(results.feature_summary.length!==103||results.horses.some(x=>x.features.length!==103))throw new Error("AUDIT_103_BY_14_CONTRACT_FAILED");
console.log("AUDIT_SUMMARY "+JSON.stringify({
  contract:results.contract,race_id:RID,daily_packs:daily.length,latest:maxSourceDate,
  profiles_found:profiles.size,features:results.feature_summary.length,
  targets:results.horses.length,unresolved_features:results.feature_summary.filter(x=>x.unresolved).map(x=>x.feature_id)
}));
console.log("AUDIT_JSON_BEGIN");
console.log(JSON.stringify(results));
console.log("AUDIT_JSON_END");
