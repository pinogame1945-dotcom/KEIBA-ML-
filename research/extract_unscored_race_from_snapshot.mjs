#!/usr/bin/env node
// Metadata-resolved 2025 yearly snapshot -> UNSCORED target race features ONLY.
// Never return/read historical targets, final prices, cached AI ranks or result.
// EARLY projection uses the original EARLY/FINAL source contract; no claim of true as-of
// provenance until the source's historical capture times are independently proven.
import { createReadStream, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createGunzip } from 'node:zlib';
import { createInterface } from 'node:readline';
import {
  normalizeFeatureSets, selectFeatureFamilies, applyPredictionPhase,
  assertPredictionPhaseSafe
} from '../src/l1-feature-contract.mjs';

const get = (k) => {const i=process.argv.indexOf(k);return i>=0?process.argv[i+1]:null;};
const path = get('--input');
const rid = get('--race-id');
const out = get('--output');
const expectedRows = Number(get('--manifest-rows'));
const expectedMin = get('--manifest-min-date');
const expectedMax = get('--manifest-max-date');
if (!Number.isSafeInteger(expectedRows) || expectedRows<=0 ||
    !/^2025-\d{2}-\d{2}$/.test(expectedMin||'') ||
    !/^2025-\d{2}-\d{2}$/.test(expectedMax||'')) {
  throw new Error('AUTHORITATIVE_MANIFEST_COUNT_AND_DATES_REQUIRED');
}
if (!path || !out || !/^2025[0-9]{8}$/.test(rid || '')) {
  throw new Error('REQUIRES_EXACT_2025_RACE_ID_INPUT_AND_OUTPUT');
}
const sets = normalizeFeatureSets('BASE,OPPONENT,NETWORK,LAP,STYLE,DISTANCE,BACKFILL,AUTO,ACTOR,TIME_PACE');
const extraUnsafe = /^(?:_finish|_is_win|_is_top3|target(?:_|$)|_target(?:_|$)|finish(?:_|$)|final(?:_|$)|market_final_|result(?:_|$)|payout(?:_|$)|postrace_|after_race_|outcome_|P__|distill_|risk(?:_|$)|actual_start_time$|market_rank$|win_odds_at_cutoff$)/i;
const files = createReadStream(path);
const decoded = path.endsWith('.gz') ? files.pipe(createGunzip()) : files;
const rl = createInterface({input:decoded, crlfDelay:Infinity});
let rowsRead=0, raceLines=0, leaked=0, month=null, ymd=null, sourceFeatures=0;
let earliest=null, latest=null;
const horses = new Map();
for await (const line of rl) {
  if (!line.trim()) continue;
  rowsRead++;
  const row=JSON.parse(line);
  const sourceDate=String((row.features||{}).race_date || row.race_date || '').slice(0,10);
  if (!/^2025-\d{2}-\d{2}$/.test(sourceDate)) throw new Error('MANIFEST_SOURCE_DATE_INVALID');
  earliest = earliest===null || sourceDate<earliest?sourceDate:earliest;
  latest = latest===null || sourceDate>latest?sourceDate:latest;
  if (String(row.race_id || '') !== rid) continue;
  raceLines++;
  const horseId=String(row.horse_id || '');
  if (!/^\d{8,16}$/.test(horseId)) throw new Error('CANONICAL_TARGET_HORSE_ID_REQUIRED');
  if (horses.has(horseId)) throw new Error('DUPLICATE_TARGET_HORSE');
  // Do NOT access row.target, finish, payout, odds, or historical model scores.
  const orig=(row.features && typeof row.features==='object' && !Array.isArray(row.features))?row.features:{};
  const date=String(orig.race_date || row.race_date || '').slice(0,10);
  if (!/^2025-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(date))) {
    throw new Error('INVALID_TARGET_DATE');
  }
  if (ymd !== null && ymd!==date) throw new Error('TARGET_RACE_DATE_MISMATCH');
  ymd=date;
  const clean=applyPredictionPhase(selectFeatureFamilies(orig,sets), 'EARLY');
  delete clean.race_date;
  const rawKeys=Object.keys(clean);
  sourceFeatures+=Object.keys(orig).length;
  for (const key of rawKeys) {
    if (extraUnsafe.test(key)) {delete clean[key];leaked++;}
  }
  assertPredictionPhaseSafe(Object.keys(clean),'EARLY');
  clean.race_month=Number(date.slice(5,7));
  clean.race_day_of_year=Math.round((Date.parse(date)-Date.UTC(2025,0,1))/86400000)+1;
  if (!Number.isInteger(clean.race_day_of_year) || clean.race_day_of_year<1 || clean.race_day_of_year>366) {
    throw new Error('RACE_DAY_OF_YEAR_INVALID');
  }
  horses.set(horseId,{_race_id:rid,_horse_id:horseId,_race_date:date,...clean});
}
if (rowsRead!==expectedRows || earliest!==expectedMin || latest!==expectedMax) {
  throw new Error('KAGGLE_COMPRESSION_EQUIVALENT_MANIFEST_COUNTS_OR_DATE_DRIFT:'+
    rowsRead+' '+earliest+' '+latest);
}
console.log('SNAPSHOT_2025_MANIFEST_ROW_DATE_RANGE_VERIFIED',rowsRead,earliest,latest);
if (raceLines===0) throw new Error('TARGET_RACE_NOT_PRESENT_IN_2025_SNAPSHOT');
if (horses.size<3 || horses.size>24) throw new Error('TARGET_STARTER_COUNT_NEEDS_INVESTIGATION');
const keys = Array.from(new Set(Array.from(horses.values()).flatMap(x=>Object.keys(x))))
  .filter(x=>!x.startsWith('_')).sort();
const unexpected = keys.filter(x=>extraUnsafe.test(x));
if (unexpected.length) throw new Error('PRE_RACE_LEAKED_TARGET_KEYS:'+unexpected.join(','));
const digest=createHash('sha256').update(JSON.stringify(Array.from(horses.values()))).digest('hex');
const output={
  contract:'KEIBA_UNSCORED_RACE_FEATURE_EXTRACT_V1',
  race_id:rid,
  race_date:ymd,
  source:'KAGGLE_2025_YEARLY_SNAPSHOT_METADATA_RESOLVED',
  status:'EARLY_PROJECTED_BUT_ASOF_AND_FULL_FIELD_UNVERIFIED',
  source_prediction_phase:'EARLY_PROJECTED_FROM_HISTORICAL_SNAPSHOT',
  used_target_labels:false,
  used_results_payouts_final_odds:false,
  used_historical_oos_target_predictions:false,
  actual_source_capture_time_independently_verified:false,
  official_full_starter_list_compared:false,
  l2_1_ready:false,
  l3_v3_ready:false,
  feature_key_count:keys.length,
  source_manifest_count_and_date_range_verified:true,
  blinded_feature_panel_sha256:digest,
  rows:Array.from(horses.values())
};
writeFileSync(out,JSON.stringify(output)+'\n',{encoding:'utf8',flag:'wx'});
console.log('TARGET_SNAPSHOT_EXTRACT_PROBE',JSON.stringify({
  race_id:rid,race_date:ymd,horses:horses.size,feature_keys:keys.length,
  banned_fields_removed:leaked,snapshot_horse_lines:raceLines,
  historical_results_used:false,certified_prebet:false,
  blinded_features_sha256:digest,
  status:output.status
}));
