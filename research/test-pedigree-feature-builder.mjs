#!/usr/bin/env node
import { createReadStream } from "node:fs";
import { readdir } from "node:fs/promises";
import { createGunzip } from "node:zlib";
import { createInterface } from "node:readline";
import path from "node:path";
import {
  lineageFromHorseRecord,
  createPedigreeFeatureState,
} from "../src/pedigree-feature-builder.mjs";

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

function raceDate(row) {
  const raw = row?.race?.actual_date ?? row?.race?.scheduled_date ?? "";
  const s = String(raw).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(s) ? s : null;
}

async function readGzipJsonl(file, onRow) {
  const input = createReadStream(file).pipe(createGunzip());
  const rl = createInterface({ input, crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    await onRow(JSON.parse(line));
  }
}

const sourceRoot = path.resolve(arg("--source-root", "../KEIBA-BACKFILL"));
const historyStart = arg("--history-start");
const emitStart = arg("--emit-start");
const end = arg("--end");
if (!historyStart || !emitStart || !end) throw new Error("--history-start, --emit-start, --end are required");
if (!(historyStart <= emitStart && emitStart <= end)) throw new Error("invalid date order");

const dailyDir = path.join(sourceRoot, "data", "daily");
const horseDir = path.join(sourceRoot, "data", "horses");

const dailyNames = (await readdir(dailyDir))
  .filter(name => /^\d{4}-\d{2}-\d{2}\.jsonl\.gz$/.test(name))
  .filter(name => {
    const d = name.slice(0, 10);
    return d >= historyStart && d <= end;
  })
  .sort();

const races = [];
const targetHorseIds = new Set();
for (const name of dailyNames) {
  await readGzipJsonl(path.join(dailyDir, name), row => {
    const date = raceDate(row);
    if (!date || date < historyStart || date > end) return;
    races.push({ date, row });
    for (const entry of row?.entries ?? []) {
      if (entry?.horse_id) targetHorseIds.add(String(entry.horse_id));
    }
  });
}
races.sort((a, b) => a.date.localeCompare(b.date) || String(a.row?.race?.race_id ?? "").localeCompare(String(b.row?.race?.race_id ?? "")));

const lineageByHorse = new Map();
const horseNames = (await readdir(horseDir))
  .filter(name => /^horse-\d{4}-\d{2}-\d{2}-\d+\.jsonl\.gz$/.test(name))
  .sort();

let horsePackFilesRead = 0;
let horseRecordsRead = 0;
for (const name of horseNames) {
  await readGzipJsonl(path.join(horseDir, name), record => {
    horseRecordsRead += 1;
    const horseId = String(record?.horse_id ?? "");
    if (!targetHorseIds.has(horseId) || lineageByHorse.has(horseId)) return;
    lineageByHorse.set(horseId, lineageFromHorseRecord(record));
  });
  horsePackFilesRead += 1;
  if (lineageByHorse.size === targetHorseIds.size) break;
}

const state = createPedigreeFeatureState();
let raceCount = 0;
let starterRows = 0;
let emitRows = 0;
let sireKnown = 0;
let damsireKnown = 0;
let sireHistoryRows = 0;
let damsireHistoryRows = 0;
let totalFeatureKeys = 0;
const examples = [];

let at = 0;
while (at < races.length) {
  const date = races[at].date;
  const day = [];
  while (at < races.length && races[at].date === date) day.push(races[at++].row);

  const updates = [];
  for (const row of day) {
    raceCount += 1;
    const resultByHorse = new Map((row?.results ?? []).map(r => [String(r?.horse_id ?? ""), r]));
    for (const entry of row?.entries ?? []) {
      const horseId = String(entry?.horse_id ?? "");
      const result = resultByHorse.get(horseId);
      if (!horseId || !result) continue;
      const finish = Number(result?.official_finish_position);
      if (!Number.isFinite(finish) || finish < 1) continue;
      starterRows += 1;
      const lineage = lineageByHorse.get(horseId) ?? { sire_key: null, damsire_key: null };

      if (date >= emitStart) {
        const features = state.snapshot(lineage, row?.race ?? {});
        emitRows += 1;
        if (lineage.sire_key) sireKnown += 1;
        if (lineage.damsire_key) damsireKnown += 1;
        if ((features.ped_sire_all_starts ?? 0) > 0) sireHistoryRows += 1;
        if ((features.ped_damsire_all_starts ?? 0) > 0) damsireHistoryRows += 1;
        totalFeatureKeys += Object.keys(features).length;
        if (examples.length < 3) {
          examples.push({
            date,
            race_id: row?.race?.race_id ?? null,
            horse_id: horseId,
            sire_known: Boolean(lineage.sire_key),
            damsire_known: Boolean(lineage.damsire_key),
            sire_starts: features.ped_sire_all_starts ?? 0,
            sire_top3_rate: features.ped_sire_all_top3_rate ?? null,
            damsire_starts: features.ped_damsire_all_starts ?? 0,
            damsire_top3_rate: features.ped_damsire_all_top3_rate ?? null,
          });
        }
      }
      updates.push({ lineage, race: row?.race ?? {}, result });
    }
  }

  // Same-day results are committed only after every prediction row for that date.
  for (const item of updates) state.add(item.lineage, item.race, item.result);
}

console.log("PEDIGREE_FEATURE_BUILDER_TEST");
console.log(JSON.stringify({
  history_start: historyStart,
  emit_start: emitStart,
  end,
  daily_files: dailyNames.length,
  races: raceCount,
  starter_rows: starterRows,
  target_horses: targetHorseIds.size,
  lineage_horses_found: lineageByHorse.size,
  lineage_coverage: targetHorseIds.size ? lineageByHorse.size / targetHorseIds.size : null,
  horse_pack_files_read: horsePackFilesRead,
  horse_records_read: horseRecordsRead,
  lineage_source_policy: "STATIC_PEDIGREE_MAY_BE_BACKFILLED_AFTER_RACE_DATE",
  emit_rows: emitRows,
  sire_known_rate: emitRows ? sireKnown / emitRows : null,
  damsire_known_rate: emitRows ? damsireKnown / emitRows : null,
  sire_prior_history_rate: emitRows ? sireHistoryRows / emitRows : null,
  damsire_prior_history_rate: emitRows ? damsireHistoryRows / emitRows : null,
  avg_feature_keys: emitRows ? totalFeatureKeys / emitRows : null,
  stat_rows: state.stat_rows(),
  examples,
}, null, 2));
