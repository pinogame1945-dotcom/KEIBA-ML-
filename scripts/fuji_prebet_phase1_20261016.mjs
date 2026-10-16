#!/usr/bin/env node
// Phase 1: inspect only Fuji course_layout, course_laps, age_max using unchanged
// KEIBA-BACKFILL race-meta normalization (read-only dependency).
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
const source=JSON.parse(await readFile("audit-results/fuji-20261017-149-asof-20261016.json","utf8"));
if(source.race_id!=="202605040611"||source.runners?.length!==14)throw Error("Wrong Fuji baseline");
const originalPath=resolve(process.env.BACKFILL_RACE_META_PATH??"/tmp/backfill-race-meta/src/race-meta.mjs");
const {normalizeRaceMeta,validateRaceMetaFields}=await import(pathToFileURL(originalPath).href);
const raw=source.raw_page;
const normalized=normalizeRaceMeta({
  course_meta_raw:raw.raw_course,
  race_condition_raw:raw.raw_condition,
  race_meta_raw:raw.raw_course+" "+raw.raw_condition,
  race_name:raw.race_name
});
validateRaceMetaFields(normalized);
const observed={
  contract:"FUJI_PREBET_PHASE1_20261016",
  race_id:source.race_id,
  original_backfill_normalizer:"src/race-meta.mjs",
  source_page_observation:source.observed_at_client_utc,
  exact_source_course_text:raw.raw_course,
  exact_source_condition_text:raw.raw_condition,
  fields:{
    "race.course_layout":{value:normalized.course_layout,meaning:"BACKFILL normalizer: when course text contains no 内 or 外 -> NORMAL",verified:true},
    "race.course_laps":{value:normalized.course_laps,meaning:"No 周 count appears in raw course text; normalizer emits null. Do not invent an integer.",verified:true},
    "race.age_max":{value:normalized.age_max,meaning:"3歳以上 has no age ceiling; normalizer returns null; not an extraction failure.",verified:true}
  },
  age_min:normalized.age_min,
  age_condition_type:normalized.age_condition_type,
  additional_mismatch_not_changed:normalized.designated!==true?"Legacy (指) versus [指] designated-flag normalization mismatch":null
};
if(normalized.course_layout!=="NORMAL"||normalized.course_laps!==null||normalized.age_max!==null||normalized.age_min!==3||normalized.age_condition_type!=="THREE_YEAR_PLUS")throw Error("Original BACKFILL normalized differently; inspect before proceeding");
console.log("PHASE1_RESULT "+JSON.stringify(observed));
