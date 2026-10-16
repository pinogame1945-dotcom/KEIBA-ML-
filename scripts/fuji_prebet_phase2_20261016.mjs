#!/usr/bin/env node
/**
 * Phase 2: actual Fuji 14-horse join and type/NULL audit for exactly 31+103
 * pre-race feature catalog rows. No model execution; no inference code changes.
 */
import {readFile} from "node:fs/promises";
const read=async p=>JSON.parse(await readFile(p,"utf8"));
const live=await read("audit-results/fuji-20261017-149-asof-20261016.json");
const backfill=await read("audit-results/fuji-20261017-backfill-93-10-final.json");
const catalog=await read("contracts/l1-feature-catalog-v1.json");
const RID="202605040611";
if(live.race_id!==RID||backfill.race_id!==RID||live.runners?.length!==14||backfill.horses?.length!==14)throw Error("RACE_ID_OR_RUNNER_COUNT_MISMATCH");
const liveItems=live.feature_audit.filter(x=>["OBSERVED_NETKEIBA_PAGE","OBSERVED_OR_DERIVED_FROM_PAGE"].includes(x.status));
if(liveItems.length!==31)throw Error("31_ACTUAL_LIVE_FEATURES_EXPECTED_GOT_"+liveItems.length);
const historic=catalog.features.filter(x=>x.history_boundary==="STRICT_PRIOR_DATE_ONLY"||x.history_boundary==="STATIC_FACT");
if(historic.length!==103||backfill.feature_summary?.length!==103)throw Error("103_BACKFILL_FEATURES_NOT_FOUND");
const allNames=new Set([...liveItems.map(x=>x.feature_id),...historic.map(x=>x.feature_id)]);
if(allNames.size!==134)throw Error("FEATURE_ID_COLLISION");
const allowedIds=new Set(live.runners.map(x=>String(x.horse_id)));
const historicIds=new Set(backfill.horses.map(x=>String(x.horse_id)));
if(allowedIds.size!==14||historicIds.size!==14||[...allowedIds].some(id=>!historicIds.has(id)))throw Error("RUNNER_ID_MISMATCH");
const byHorse=new Map(backfill.horses.map(x=>[String(x.horse_id),x]));
const sourceById=new Map(liveItems.map(x=>[x.feature_id,x]));
const catalogById=new Map(catalog.features.map(f=>[f.feature_id,f]));
let entryCount=0,raceCount=0;
const output={
  contract:"FUJI_20261017_31_PLUS_103_JOIN_AUDIT_V1",race_id:RID,
  as_of:"2026-10-16",source_race_sha256:live.source_sha256,
  source_backfill_latest:backfill.backfill_main_latest_race_date,
  includes_post_race_values:false,includes_final_odds:false,excludes_coat_color_substitution:true,
  feature_catalog_count:134,horses:[],summary:null};
const scalar=v=>v==null||["string","number","boolean"].includes(typeof v);
function addFlat(out,key,val,origin,conflicts){
 if(!key||typeof key!=="string"){conflicts.push({key,origin,reason:"invalid key"});return}
 if(!scalar(val)){conflicts.push({key,origin,reason:"non-scalar in flat model-candidate"});return}
 if(Object.hasOwn(out,key)){
   if(out[key]!==val)conflicts.push({key,origin,reason:"value conflict",previous:out[key],incoming:val});
 }else out[key]=val;
}
for(const runner of live.runners){
 const id=String(runner.horse_id),historical=byHorse.get(id);
 if(historical.features?.length!==103)throw Error("103_HISTORICAL_ROWS_MISSING:"+id);
 const map={};const modelCandidate={};const conflicts=[];
 const raceInputs={},entryInputs={};
 for(const item of liveItems){
   let value=item.observed_value;
   if(item.status==="OBSERVED_NETKEIBA_PAGE"){
     if(!Array.isArray(value))throw Error("PER_HORSE_VALUE_NOT_ARRAY:"+item.feature_id);
     const found=value.filter(x=>String(x.horse_id)===id);
     if(found.length!==1)throw Error("PER_HORSE_VALUE_CARDINALITY:"+item.feature_id+":"+id);
     value=found[0].value;entryCount++;
   }else raceCount++;
   const f=catalogById.get(item.feature_id);
   if(!f||!scalar(value)||value==null)throw Error("LIVE_FIELD_INVALID:"+item.feature_id);
   map[item.feature_id]={value,source:"LIVE_NETKEIBA",status:item.status};
   const field=f.source_field;
   if(!field||!field.includes("."))throw Error("LIVE_SOURCE_MAPPING_ABSENT:"+item.feature_id);
   if(item.feature_id.startsWith("race."))raceInputs[item.feature_id.slice(5)]=value;
   if(item.feature_id.startsWith("entry."))entryInputs[item.feature_id.slice(6)]=value;
 }
 for(const f of historic){
   const rows=historical.features.filter(x=>x.feature_id===f.feature_id);
   if(rows.length!==1)throw Error("HISTORIC_FEATURE_ID_CARDINALITY:"+f.feature_id);
   const r=rows[0],value=r.value;
   if(value!==null&&!scalar(value)&&!(Array.isArray(value)===false&&typeof value==="object"))throw Error("FEATURE_VALUE_TYPE_UNEXPECTED:"+f.feature_id);
   map[f.feature_id]={value,source:"BACKFILL",status:r.status,key:r.key};
   if(r.key?.startsWith("family:")){
     if(value==null||typeof value!=="object"||Array.isArray(value))throw Error("FAMILY_NONOBJECT:"+f.feature_id);
     for(const [k,v] of Object.entries(value))addFlat(modelCandidate,k,v,f.feature_id,conflicts);
   }else if(r.key){
     addFlat(modelCandidate,r.key,value,f.feature_id,conflicts);
   }else conflicts.push({feature_id:f.feature_id,reason:"missing original model key mapping"});
 }
 if(Object.keys(map).length!==134)throw Error("134_FIELDS_NOT_JOINED");
 for(const [k,v] of Object.entries(entryInputs))if(runner[k]!==v)throw Error("ENTRY_VALUE_CHANGED:"+k);
 if(entryInputs.horse_id!==id)throw Error("ENTRY_HORSE_ID_MISMATCH");
 if(entryInputs.jockey_id&&!/^\d{5}$/.test(entryInputs.jockey_id))throw Error("JOCKEY_ID_STRING_LOST_LEADING_ZERO");
 if(entryInputs.trainer_id&&!/^\d{5}$/.test(entryInputs.trainer_id))throw Error("TRAINER_ID_STRING_LOST_LEADING_ZERO");
 const nullIds=Object.entries(map).filter(([,x])=>x.value==null).map(([k])=>k);
 output.horses.push({
   horse_id:id,horse_name:runner.horse_name,race:raceInputs,entry:entryInputs,
   catalog_features:map,model_candidate_features:modelCandidate,
   feature_count:Object.keys(map).length,
   scalar_null_count:nullIds.length,scalar_null_feature_ids:nullIds,
   model_candidate_feature_count:Object.keys(modelCandidate).length,
   model_candidate_null_count:Object.values(modelCandidate).filter(x=>x===null).length,
   candidate_conflicts:conflicts});
}
const nulls=output.horses.reduce((n,h)=>n+h.scalar_null_count,0);
const conflicts=output.horses.reduce((n,h)=>n+h.candidate_conflicts.length,0);
output.summary={runner_count:14,live_features:31,backfill_features:103,feature_count_per_horse:134,
  cells_checked:14*134,source_entry_values_checked:entryCount,source_race_values_checked:raceCount,
  top_level_null_cells:nulls,candidate_key_collisions_or_type_errors:conflicts,
  candidate_cols_min:Math.min(...output.horses.map(x=>x.model_candidate_feature_count)),
  candidate_cols_max:Math.max(...output.horses.map(x=>x.model_candidate_feature_count)),
  model_inference_tested:false,model_input_schema_compatibility_not_proven:true,
  expected_nulls_from_source_103:37};
if(conflicts)console.error("PHASE2_COLLISIONS "+JSON.stringify(output.horses.filter(x=>x.candidate_conflicts.length).map(x=>({horse:x.horse_name,conflicts:x.candidate_conflicts.slice(0,15)})).slice(0,3)));
if(nulls!==37||conflicts!==0||entryCount!==112||raceCount!==322||output.horses.length!==14)throw Error("JOIN_INTEGRITY:"+JSON.stringify(output.summary));
console.log("PHASE2_SUMMARY "+JSON.stringify(output.summary));
console.log("PHASE2_JSON "+JSON.stringify(output));
