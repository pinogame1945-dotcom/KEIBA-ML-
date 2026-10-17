#!/usr/bin/env node
/**
 * Original BACKFILL v2 race-context recovery for 2026-10-17 Outsider.
 *
 * Strictly read-only source: a JRA pre-race live daily pack already produced
 * by the PINNED KEIBA-BACKFILL live_json_to_daily_pack_v1.py. Dynamic import
 * of that SAME pinned repo's backfill-feature-adapter.mjs: no reimplementation.
 *
 * Results, current/outcome odds, historical STYLE and 10/10 data are NOT used.
 * Produces 471 horse rows with race-side backfill_* metadata ONLY.
 * Missing raw source fields stay null (no pretending to have full 13 inputs).
 */
import fs from 'node:fs';
import path from 'node:path';
import url from 'node:url';
import zlib from 'node:zlib';

const CUT_OFF_MS=Date.parse("2026-10-16T15:00:00Z");
const DAY="2026-10-17";
const ALLOWED=new Set(["STARTED","SCRATCHED","EXCLUDED"]);
const EXPECTED_FIELDS=[
  "backfill_field_size","backfill_course_layout","backfill_course_laps",
  "backfill_race_class_normalized","backfill_grade","backfill_age_min",
  "backfill_age_max","backfill_sex_condition","backfill_weight_rule",
  "backfill_mixed","backfill_international","backfill_special_designated",
  "backfill_designated"
];
function demand(ok, reason) {if(!ok) throw Error(reason);}
function getArg(name) {
  const at=process.argv.indexOf(name);
  demand(at>=0 && process.argv[at+1] && !process.argv[at+1].startsWith("--"),
         "REQUIRED_ARG_"+name);
  return process.argv[at+1];
}
function readPack(p) {
  const raw=fs.readFileSync(p);
  const txt=p.endsWith(".gz")?zlib.gunzipSync(raw).toString("utf8"):raw.toString("utf8");
  return txt.trim().split("\n").map(x=>JSON.parse(x));
}
function validBefore(value,name){
  demand(typeof value==="string" && /(?:Z|[+-]\d\d:\d\d)$/.test(value) &&
         Number.isFinite(Date.parse(value)) && Date.parse(value)<CUT_OFF_MS,
         "PRE_RACE_CAPTURE_NOT_PROVEN:"+name);
}
async function main() {
  const input=getArg("--daily-pack");
  const moduleFile=getArg("--original-backfill-adapter");
  const output=getArg("--output");
  demand(!fs.existsSync(output),"REFUSE_OVERWRITE_ORIGINAL_RACE_CONTEXT");
  demand(path.basename(moduleFile)==="backfill-feature-adapter.mjs",
         "BACKFILL_ORIGINAL_ADAPTER_REQUIRED");
  const original=await import(url.pathToFileURL(path.resolve(moduleFile)).href);
  demand(original.BACKFILL_FEATURE_ADAPTER_VERSION===2 &&
         typeof original.readBackfillRaceFeatures==="function",
         "ORIGINAL_BACKFILL_V2_SCHEMA_MISMATCH");
  const records=readPack(input);
  demand(records.length===36,"LIVE_20261017_36_RACES_REQUIRED");
  let starters=0;
  const seenRaces=new Set();
  const perHorse=[];
  const nullCoverage=Object.fromEntries(EXPECTED_FIELDS.map(k=>[k,0]));
  const warnings=new Set();
  for(const doc of records){
    const race=doc.race??{};
    const rid=String(race.race_id??"");
    demand(/^2026(?:04|05|08)04\d{4}$/.test(rid) &&
           String(race.scheduled_date)===DAY && race.actual_date==null &&
           !seenRaces.has(rid),"PRE_RACE_LIVE_RACE_ID_DRIFT:"+rid);
    demand(doc.live_source_status==="PRE_RACE_OBSERVED_ASOF_UNVERIFIED",
           "ONLY_ORIGINAL_PRE_RACE_SOURCE_ACCEPTED:"+rid);
    validBefore(doc.live_source_observed_at,rid);
    for(const key of ["results","payouts","odds","laps","corners"]){
      demand(Array.isArray(doc[key]) && doc[key].length===0,
             "OUTCOMES_OR_LIVE_MARKET_FORBIDDEN_IN_OUTSIDER_SOURCE:"+rid+":"+key);
    }
    const entries=doc.entries??[];
    demand(entries.length>=3 && entries.length<=18,"RACE_ENTRIES_NOT_VALID:"+rid);
    const allowedIds=new Set();
    const allIds=new Set();
    for (const ent of entries) {
      const hid=String(ent.horse_id??"");
      demand(/^\d{8,16}$/.test(hid) && !allIds.has(hid) &&
             ALLOWED.has(String(ent.entry_status??"")),
             "LIVE_RACE_RUNNER_ID_OR_STATUS_DRIFT:"+rid);
      allIds.add(hid);
      if(ent.entry_status==="STARTED")allowedIds.add(hid);
    }
    const {features,warnings:originalWarnings}=original.readBackfillRaceFeatures(race,entries);
    demand(Object.keys(features).sort().join("|")===
           [...EXPECTED_FIELDS].sort().join("|"),
           "FROZEN_BACKFILL_ADAPTER_13_COLUMN_DRIFT");
    for(const warning of originalWarnings)warnings.add(warning);
    const liveContext={
      "race_date":DAY,
      "venue_code":race.venue_code,
      "meeting_no":race.meeting_no,"meeting_day":race.meeting_day,
      "race_no":race.race_no,
      "discipline":race.discipline,"surface":race.surface,
      "distance_m":race.distance_m,"direction":race.direction,
      "weather":race.weather,"track_condition":race.track_condition,
      ...features
    };
    for(const k of EXPECTED_FIELDS)if(liveContext[k]===null)nullCoverage[k]++;
    for(const ent of entries.filter(z=>z.entry_status==="STARTED")){
      perHorse.push({
        "contract":"KEIBA_2026_10_17_ORIGINAL_OUTSIDER_RACE_CONTEXT_V1",
        "race_id":rid,"horse_id":String(ent.horse_id),
        "original_backfill_adapter_version":2,
        "source_capture_utc":doc.live_source_observed_at,
        "source_origin":"PINNED_JRA_LIVE_PRE_RACE_DAY_PACK",
        "raw_race_context_observed_before_target_date":true,
        "derived_style_history_available":false,
        "derived_13_outsider_all_features_ready":false,
        "independent_full_input_asof_verified":false,
        "ready_for_betting":false,
        "race_features":liveContext,
      });
      starters++;
    }
    seenRaces.add(rid);
  }
  demand(seenRaces.size===36 && starters===471,
         "ORIGINAL_OUTSIDER_RACE_CONTEXT_36_471_MISMATCH");
  demand(!fs.existsSync(output),"RACE_CONTEXT_OUTPUT_OVERWRITE_REFUSED");
  fs.mkdirSync(path.dirname(output),{recursive:true});
  fs.writeFileSync(output,perHorse.map(x=>JSON.stringify(x)).join("\n")+"\n");
  console.log("ORIGINAL_2026_OUTSIDER_BACKFILL_CONTEXT_PRE_RACE_36_471_PASS",JSON.stringify({
    "race_count":36,"horse_rows":471,"original_backfill_fields":EXPECTED_FIELDS.length,
    "fields_with_any_null":Object.entries(nullCoverage).filter(([,n])=>n>0).map(([k])=>k),
    "observed_input_source":"2026-10-16 prior to 15:00 UTC",
    "original_builder_applied":true,"original_style_history_derived":false,
    "full_13_outsider_ready":false,"2026_outcomes_read":false,
    "output":output
  }));
}
main().catch(err=>{console.error(err);process.exitCode=1});
