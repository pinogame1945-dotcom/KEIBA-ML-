import {readFile,readdir,access} from "node:fs/promises";
import {resolve,relative,isAbsolute,join} from "node:path";
import {gunzipSync} from "node:zlib";
import {createMlDatasetProcessor,ML_FEATURE_SCHEMA_VERSION} from "./ml-dataset.mjs";
import {createHistoricalExtraState} from "./single-race-historical-extra.mjs";
import {lineageFromHorseRecord} from "./pedigree-feature-builder.mjs";
import {
  applyPredictionPhase,normalizeFeatureSets,normalizeHistoryWindows,
  normalizePredictionPhase,selectFeatureFamilies
} from "./l1-feature-contract.mjs";
import {DEFAULT_SMALL_SAMPLE_POLICY} from "./small-sample-feature-utils.mjs";

const RACE_ID=/^\d{12}$/;
const DATE_RE=/^\d{4}-\d{2}-\d{2}$/;
const RESULT_FIELDS=[
  "race_id","actual_date","scheduled_date","venue_code","meeting_no",
  "meeting_day","race_no","discipline","surface","distance_m","direction",
  "field_size","course_layout","course_laps","race_class_normalized",
  "grade","age_min","age_max","sex_condition","weight_rule","mixed",
  "international","special_designated","designated"
];
// This is the archived ENTRY whitelist. It intentionally excludes final odds,
// final popularity, actual body weight, entry result fields, race outcomes.
const ENTRY_FIELDS=[
  "horse_id","horse_name","horse_number","gate","sex","age","carried_weight",
  "jockey_id","trainer_id","entry_status"
];

export function parseRaceId(value){
  const raw=String(value??"").trim();
  if(RACE_ID.test(raw))return raw;
  try{
    const url=new URL(raw);
    if(!["race.netkeiba.com","db.netkeiba.com"].includes(url.hostname))
      throw new Error("only netkeiba URLs supported");
    const id=url.searchParams.get("race_id")??url.pathname.match(/\/race\/(\d{12})\/?/)?.[1];
    if(id&&RACE_ID.test(id))return id;
  }catch(error){
    if(error?.message==="only netkeiba URLs supported")throw error;
  }
  throw new Error("race must be a 12-digit canonical race ID or netkeiba race URL");
}

export function verifyRaceManifestPath(root,p){
  if(typeof p!=="string"||!p||isAbsolute(p)||p.includes("\\")||
    p.split("/").some(t=>t===".."||t==="."||!t))
    throw new Error("invalid manifest pack file path: "+String(p));
  const safe=resolve(root,p),rel=relative(resolve(root),safe);
  if(rel.startsWith("..")||isAbsolute(rel))throw new Error("path escapes BACKFILL: "+p);
  return safe;
}

export function stripTargetOutcome(row,{phase="EARLY"}={}){
  const race=Object.fromEntries(RESULT_FIELDS
    .filter(k=>Object.hasOwn(row.race??{},k))
    .map(k=>[k,row.race[k]]));
  if(phase!=="FINAL"&&phase!=="EARLY")throw new Error("invalid phase");
  // Final archive does not prove when race-day weather / current body
  // weight was published. Historical replay is EARLY only.
  const entries=(row.entries??[]).map(entry=>
    Object.fromEntries(ENTRY_FIELDS
      .filter(k=>Object.hasOwn(entry,k))
      .map(k=>[k,entry[k]])));
  if(!RACE_ID.test(String(race.race_id??"")))throw new Error("invalid race id");
  if(entries.length<1)throw new Error("no archived starters for "+race.race_id);
  return {race,entries};
}

export async function loadPack(file){
  const data=await readFile(file);
  const plain=file.endsWith(".gz")?gunzipSync(data).toString("utf8"):data.toString("utf8");
  return plain.split("\n").filter(Boolean).map(line=>JSON.parse(line));
}

export async function findArchivedRace(root,raceId,manifest){
  const year=Number(raceId.slice(0,4));
  // Reschedules may cross a month; search by the canonical race year and the next one.
  const candidates=Object.entries(manifest.days??{})
    .filter(([date,item])=>item?.status==="SUCCESS" &&
      Number(date.slice(0,4))>=year&&Number(date.slice(0,4))<=year+1)
    .sort(([a],[b])=>a.localeCompare(b));
  const found=[];
  for(const [date,item] of candidates){
    if(!item.file)throw new Error("manifest SUCCESS day lacks file: "+date);
    const rows=await loadPack(verifyRaceManifestPath(root,item.file));
    for(const row of rows)if(String(row?.race?.race_id??"")===raceId)
      found.push({date,row,source_file:item.file});
  }
  if(found.length!==1)throw new Error("ARCHIVE_RACE_ID_NOT_UNIQUE: "+raceId+" (count="+found.length+")");
  return found[0];
}

async function loadLineage(root,manifest,historyRaceIds){
  const data=new Map();
  for(const pack of Object.values(manifest.horse_packs??{})){
    if(pack?.status!=="SUCCESS"||!pack.file)continue;
    const rows=await loadPack(verifyRaceManifestPath(root,pack.file));
    for(const row of rows){
      const id=String(row?.horse_id??"");
      if(!historyRaceIds.has(id)||data.has(id)||
         !Array.isArray(row.pedigree)||row.pedigree.length!==62)continue;
      data.set(id,lineageFromHorseRecord(row));
    }
    if(data.size===historyRaceIds.size)break;
  }
  return data;
}

export async function runOneRace({
  sourceRoot,raceId,historyFrom="2021-01-01",featureSets="BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,BACKFILL,AUTO,ACTOR,TIME_PACE",
  predictionPhase="EARLY",historyWindows={},smallSamplePolicy={},manifestOverride=null,
  // For deterministic integration tests; production reads source packs unchanged.
  onDay=null,
}){
  const root=resolve(sourceRoot);
  const id=parseRaceId(raceId);
  if(!DATE_RE.test(historyFrom))throw new Error("invalid history-from date");
  const phase=normalizePredictionPhase(predictionPhase);
  if(phase!=="EARLY")throw new Error(
    "ARCHIVED_FINAL_AS_OF_UNVERIFIED: FINAL requires pre-race timestamped source, unavailable in result packs");
  const families=normalizeFeatureSets(featureSets);
  const windows=normalizeHistoryWindows(historyWindows);
  const policy={...DEFAULT_SMALL_SAMPLE_POLICY,actorRecentWindow:windows.actor_recent,...smallSamplePolicy};
  const manifest=manifestOverride??JSON.parse(await readFile(join(root,"data/manifest.json"),"utf8"));
  const found=await findArchivedRace(root,id,manifest);
  const date=found.date;
  if(date<historyFrom)throw new Error("target predates history start");
  const target=stripTargetOutcome(found.row,{phase});
  const sourceDays=Object.entries(manifest.days??{})
    .filter(([d,p])=>d>=historyFrom&&d<date&&p?.status==="SUCCESS")
    .sort(([a],[b])=>a.localeCompare(b));
  const allHistoryIds=new Set(target.entries.map(e=>String(e.horse_id??"")));
  let scanDays=0,scanRaces=0,anomalousDates=0;
  // Strict completeness gate prevents silent hole-as-no-history errors.
  for(let d=historyFrom;d<date;d=new Date(Date.parse(d+"T00:00:00Z")+86400000).toISOString().slice(0,10)){
    if(manifest.days?.[d]?.status==="SUCCESS")continue;
    if(manifest.non_meeting_days?.[d]?.status==="CONFIRMED_NO_JRA")continue;
    if(manifest.schedule_exception_days?.[d]?.status==="NO_RACES_HELD")continue;
    throw new Error("HISTORY_DAY_NOT_VERIFIED: "+d);
  }
  let cached=null;
  if(families.includes("PEDIGREE")){
    // Collect IDs from archived race entries only. Lineage is stable/static.
    for(const [,pack] of sourceDays){
      const rows=await loadPack(verifyRaceManifestPath(root,pack.file));
      for(const row of rows)for(const entry of row.entries??[])
        if(entry?.horse_id)allHistoryIds.add(String(entry.horse_id));
    }
    cached=await loadLineage(root,manifest,allHistoryIds);
  }
  const proc=createMlDatasetProcessor({
    historyWindows:windows,includeAutoFeatures:families.includes("AUTO"),
    includeBackfillFeatures:families.includes("BACKFILL"),
    includePedigreeFeatures:families.includes("PEDIGREE"),
    includeActorFeatures:families.includes("ACTOR"),
    includeOpponentRelationshipFeatures:families.includes("OPPONENT"),
    includeTimePaceFeatures:families.includes("TIME_PACE"),
    lineageByHorse:cached??new Map(),smallSamplePolicy:policy,
  });
  const extra=createHistoricalExtraState(windows);
  for(const [d,pack] of sourceDays){
    const rows=await loadPack(verifyRaceManifestPath(root,pack.file));
    for(const row of rows){
      // Manifest path owns actual race date; no lookahead across reschedules.
      if(String(row?.race?.actual_date??row?.race?.scheduled_date??"").slice(0,10)!==d)
        throw new Error("HISTORY_PACK_DATE_DRIFT: "+d);
    }
    proc.processDay(rows,d,{startDate:"9999-01-01",endDate:"9999-12-31"});
    extra.commitDay(rows,d);
    scanDays++;scanRaces+=rows.length;
    if(onDay)await onDay(d,scanDays);
  }
  const rows=proc.predictDay([target],date);
  const byHorse=new Map(target.entries.map(e=>[String(e.horse_id),e]));
  const selected=rows.map(row=>{
    const entry=byHorse.get(String(row.horse_id));
    const more=extra.snapshot(target,entry,date);
    const safe=applyPredictionPhase(selectFeatureFamilies({
      ...row.features,...more,
    },families),phase);
    return {...row,horse_name:entry.horse_name??null,horse_number:entry.horse_number??null,
      prediction_phase:phase,feature_sets:families,history_windows:windows,
      small_sample_policy:policy,features:safe};
  }).sort((a,b)=>(a.horse_number??99)-(b.horse_number??99)||a.horse_id.localeCompare(b.horse_id));
  if(!selected.length)throw new Error("no eligible runners");
  const schema=new Set(Object.keys(selected[0].features));
  for(const row of selected)if(Object.keys(row.features).length!==schema.size||
    Object.keys(row.features).some(k=>!schema.has(k)))
    throw new Error("feature column mismatch between horses");
  return{
    contract:"KEIBA_ONE_RACE_ASOF_V1",status:"FEATURES_READY_NOT_YET_SCORED",
    race_id:id,race_date:date,phase,source_of_truth:"KEIBA-BACKFILL",
    source_pack:found.source_file,history_from:historyFrom,
    feature_schema_version:ML_FEATURE_SCHEMA_VERSION,
    limitation:"Final archived entries are not timestamped pre-race snapshots; field and IDs are replay assumptions. No target outcome or final odds are loaded into model features.",
    history:{days:scanDays,races:scanRaces,missing_lineage:families.includes("PEDIGREE")?
      allHistoryIds.size-cached.size:null},
    features_by_horse:selected,
  };
}
