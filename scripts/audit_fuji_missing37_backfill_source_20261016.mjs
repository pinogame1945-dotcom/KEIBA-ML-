#!/usr/bin/env node
// Diagnosis only; reads existing KEIBA-BACKFILL /data/horses and /data/daily.
// NO changes to data, feature or model production code.
import {readFile,readdir} from "node:fs/promises";
import {gunzipSync} from "node:zlib";
import path from "node:path";
import {fileURLToPath} from "node:url";
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"..");
const SRC=path.resolve(process.env.BACKFILL_SOURCE_ROOT??"/tmp/keiba-backfill-readonly");
const baseline=JSON.parse(await readFile(path.join(ROOT,"audit-results/fuji-20261017-149-asof-20261016.json"),"utf8"));
const horses=baseline.runners, ids=new Set(horses.map(h=>h.horse_id));
if(ids.size!==14||horses.length!==14)throw Error("Expected fourteen actual Fuji entrants");
const byId=new Map(horses.map(h=>[h.horse_id,{
  horse_id:h.horse_id,horse_name:h.horse_name,fuji_jockey_id:h.jockey_id,
  profiles:[],previous:[],all_previous_count:0,
  same_jockey_prior_count:0,
}]));
const packs=(await readdir(path.join(SRC,"data/horses"))).filter(f=>/^horse-\d{4}-\d\d-\d\d-\d+\.jsonl\.gz$/.test(f)).sort();
const global={pack_records:0,pack_non_null_coat:0,pack_raw_coat_text:0,pack_profile_keys:{},found_ids:[],profile_sources_missing:[]};
for(let i=0;i<packs.length;i++){
  const file=packs[i],txt=gunzipSync(await readFile(path.join(SRC,"data/horses",file))).toString("utf8");
  for(const line of txt.split("\n")){
    if(!line.trim())continue;
    const row=JSON.parse(line);
    if(row.kind!=="horse")continue;
    global.pack_records++;
    const p=row.profile??{};
    if(p.coat_color)global.pack_non_null_coat++;
    if(p.profile_raw_json?.includes("毛色"))global.pack_raw_coat_text++;
    if(global.pack_records<=30000)for(const key of Object.keys(p))global.pack_profile_keys[key]=(global.pack_profile_keys[key]??0)+1;
    if(!ids.has(String(row.horse_id)))continue;
    let raw={};try{raw=JSON.parse(p.profile_raw_json??"{}")}catch{}
    byId.get(String(row.horse_id)).profiles.push({
      pack:file,profile_coat_color:p.coat_color??null,
      raw_coat_color:raw["毛色"]??null,
      raw_color_keys:Object.fromEntries(Object.entries(raw).filter(([k])=>/色|毛|産地|父/.test(k))),
      raw_profile_keys:Object.keys(raw).slice(0,30),
      profile_keys:Object.keys(p),
      sex:p.sex??null,kind:row.kind,
    });
  }
}
for(const h of byId.values())if(!h.profiles.length)global.profile_sources_missing.push(h.horse_id);
const days=(await readdir(path.join(SRC,"data/daily"))).filter(f=>/^\d{4}-\d\d-\d\d\.jsonl\.gz$/.test(f)&&f.slice(0,10)<"2026-10-17").sort();
let rows=0,maxDate=null;
for(const file of days){
  const txt=gunzipSync(await readFile(path.join(SRC,"data/daily",file))).toString("utf8");
  for(const line of txt.split("\n")){
    if(!line.trim())continue;
    const r=JSON.parse(line);
    const date=String(r.race?.actual_date??r.race?.scheduled_date??"").slice(0,10);
    if(!date||date>="2026-10-17")throw Error("INVALID_DAILY_DATE "+date);
    if(!maxDate||date>maxDate)maxDate=date;
    rows++;
    const results=new Map((r.results??[]).map(x=>[String(x.horse_id),x]));
    for(const e of r.entries??[]){
      const id=String(e.horse_id??"");
      if(!ids.has(id))continue;
      const h=byId.get(id),result=results.get(id)??null;
      h.all_previous_count++;
      if(String(e.jockey_id??"")===h.fuji_jockey_id)h.same_jockey_prior_count++;
      h.previous.push({
        race_id:r.race?.race_id,date,
        current_race_distance_m:r.race?.distance_m,
        jockey_id:e.jockey_id,horse_id:e.horse_id,
        result_status:result?.result_status??null,
        finish_position:result?.official_finish_position??null,
        margin_raw:result?.margin_raw??null,
        margin_type:result?.margin_type??null,
        margin_lengths:result?.margin_lengths??null,
        margin_parser_version:r.margin_parser_version??null,
        has_margin_raw:result?.margin_raw!=null&&result?.margin_raw!=="",
      });
    }
  }
}
const out={
  contract:"FUJI_20261017_MISSING_37_SOURCE_DIAGNOSIS_V1",
  source_date_through:maxDate,pack_counts:{daily:days.length,horse:packs.length},race_rows:rows,
  global:{...global},
  horses:[],
};
for(const h of byId.values()){
  h.previous.sort((a,b)=>a.date.localeCompare(b.date)||a.race_id.localeCompare(b.race_id));
  const last=h.previous.at(-1);
  const previous20=h.previous.slice(-20);
  const stats={
    previous20_len:previous20.length,
    shorter_starts:previous20.filter(x=>Number(x.current_race_distance_m)<1600&&x.result_status==="FINISHED").length,
    longer_starts:previous20.filter(x=>Number(x.current_race_distance_m)>1600&&x.result_status==="FINISHED").length,
    same_jockey_prior_count:h.same_jockey_prior_count,
    margin_raw_present:previous20.filter(x=>x.has_margin_raw).length,
    margin_type_present:previous20.filter(x=>x.margin_type!=null).length,
    margin_lengths_present:previous20.filter(x=>x.margin_lengths!=null).length,
    distinct_margin_parser_versions:[...new Set(previous20.map(x=>x.margin_parser_version))],
  };
  out.horses.push({
    horse_id:h.horse_id,horse_name:h.horse_name,
    profiles_count:h.profiles.length,
    profiles_first:h.profiles[0]??null,
    profiles_latest:h.profiles.at(-1)??null,
    any_profile_coat_present:h.profiles.some(p=>p.profile_coat_color!=null),
    any_profile_raw_coat_present:h.profiles.some(p=>p.raw_coat_color!=null),
    all_previous_count:h.all_previous_count,stats,
    recent_3:h.previous.slice(-3),
    recent_20_margin_types:previous20.reduce((acc,r)=>{const k=String(r.margin_type);acc[k]=(acc[k]??0)+1;return acc},{}),
  });
}
const overview={
  contract:out.contract,source_through:maxDate,horse_packs:packs.length,daily_packs:days.length,
  all_profile_records:global.pack_records,all_profile_non_null_coat:global.pack_non_null_coat,
  all_raw_profile_coat_text:global.pack_raw_coat_text,
  target_profiles_found:out.horses.filter(h=>h.profiles_count>0).length,
  target_any_non_null_coat:out.horses.filter(h=>h.any_profile_coat_present).length,
  target_any_raw_coat:out.horses.filter(h=>h.any_profile_raw_coat_present).length,
  margin_last_raw_missing:out.horses.filter(h=>h.recent_3.at(-1)?.margin_raw==null).map(h=>h.horse_name),
};
console.log("DIAG_OVERVIEW "+JSON.stringify(overview));
console.log("DIAG_JSON "+JSON.stringify(out));
