#!/usr/bin/env node
import { readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { gunzipSync } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ML_DATASET_VERSION,
  ML_FEATURE_SCHEMA_VERSION,
  ML_LEAKAGE_POLICY,
  createMlDatasetProcessor,
} from "../src/ml-dataset.mjs";
import { createHistoricalExtraProcessor } from "../src/historical-extra-builder.mjs";
import {
  applyPredictionPhase,
  normalizeFeatureSets,
  selectFeatureFamilies,
  L1_FEATURE_SET_CONTRACT,
} from "../src/l1-feature-contract.mjs";
import { lineageFromHorseRecord } from "../src/pedigree-feature-builder.mjs";
import { DEFAULT_SMALL_SAMPLE_POLICY } from "../src/small-sample-feature-utils.mjs";
import { createYearlySnapshotWriter } from "../src/yearly-snapshot-store.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");

function arg(name) {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : null;
}

function dateArg(name, fallback) {
  const value = arg(name) ?? fallback;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error("invalid " + name + ": " + value);
  return value;
}

function sha256Text(text) {
  return createHash("sha256").update(text).digest("hex");
}

async function sha256File(file) {
  return sha256Text(await readFile(file));
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

function stableHash(value) {
  return sha256Text(JSON.stringify(canonicalize(value)));
}

function raceDate(row) {
  const value = row?.race?.actual_date ?? row?.race?.scheduled_date ?? "";
  const text = String(value).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

function parseDailyFile(file) {
  const text = gunzipSync(file).toString("utf8").trim();
  return text ? text.split("\n").filter(Boolean).map(JSON.parse) : [];
}

function groupByDate(rows) {
  const sorted = [...rows]
    .map(row => ({ row, date: raceDate(row) }))
    .filter(item => item.date)
    .sort((a, b) => a.date.localeCompare(b.date) ||
      String(a.row?.race?.race_id ?? "").localeCompare(String(b.row?.race?.race_id ?? "")));
  const groups = [];
  let current = null;
  for (const item of sorted) {
    if (!current || current.date !== item.date) {
      current = { date: item.date, rows: [] };
      groups.push(current);
    }
    current.rows.push(item.row);
  }
  return groups;
}

async function collectTargetHorseIds(dailyDir, dailyNames) {
  const ids = new Set();
  let races = 0;
  for (const name of dailyNames) {
    const rows = parseDailyFile(await readFile(path.join(dailyDir, name)));
    races += rows.length;
    for (const row of rows) {
      for (const entry of row?.entries ?? []) {
        const horseId = String(entry?.horse_id ?? "");
        if (horseId) ids.add(horseId);
      }
    }
  }
  return { ids, races };
}

async function loadLineageMap(sourceRoot, targetHorseIds) {
  const horseDir = path.join(sourceRoot, "data", "horses");
  const horseNames = (await readdir(horseDir))
    .filter(name => /^horse-\d{4}-\d{2}-\d{2}-\d+\.jsonl\.gz$/.test(name))
    .sort();
  const map = new Map();
  let filesRead = 0;
  let recordsRead = 0;
  for (const name of horseNames) {
    const text = gunzipSync(await readFile(path.join(horseDir, name))).toString("utf8").trim();
    if (text) {
      for (const line of text.split("\n")) {
        if (!line.trim()) continue;
        recordsRead += 1;
        const record = JSON.parse(line);
        const horseId = String(record?.horse_id ?? "");
        if (!horseId || !targetHorseIds.has(horseId) || map.has(horseId)) continue;
        map.set(horseId, lineageFromHorseRecord(record));
      }
    }
    filesRead += 1;
    if (map.size === targetHorseIds.size) break;
  }
  return { map, files_read: filesRead, records_read: recordsRead, targets: targetHorseIds.size };
}

const sourceRoot = path.resolve(arg("--source-root") ?? "../KEIBA-BACKFILL");
const outRoot = path.resolve(arg("--out-dir") ?? "out/yearly-snapshots");
const sourceStart = dateArg("--source-start", "2022-01-01");
const sourceEnd = dateArg("--source-end", "2025-12-31");
const emitStart = dateArg("--emit-start", sourceStart);
const emitEnd = dateArg("--emit-end", sourceEnd);
const sourceSha = arg("--source-sha");
const mlSourceSha = arg("--ml-source-sha");
const predictionPhase = String(arg("--prediction-phase") ?? "FINAL").toUpperCase();
const firstModelReadyYear = Number(arg("--first-model-ready-year") ?? (Number(sourceStart.slice(0, 4)) + 1));
const allFeatureSets = Object.keys(L1_FEATURE_SET_CONTRACT.feature_sets);
const featureSets = normalizeFeatureSets(arg("--feature-sets") ?? allFeatureSets.join(","));
const historyWindows = { ...L1_FEATURE_SET_CONTRACT.history_windows };
const smallSamplePolicy = {
  ratePriorStrength: Number(arg("--shrinkage-rate-strength") ?? DEFAULT_SMALL_SAMPLE_POLICY.ratePriorStrength),
  meanPriorStrength: Number(arg("--shrinkage-mean-strength") ?? DEFAULT_SMALL_SAMPLE_POLICY.meanPriorStrength),
  minSpecificObservations: Number(arg("--min-specific-observations") ?? DEFAULT_SMALL_SAMPLE_POLICY.minSpecificObservations),
  actorRecentWindow: historyWindows.actor_recent,
};

if (!sourceSha) throw new Error("--source-sha is required");
if (!mlSourceSha) throw new Error("--ml-source-sha is required");
if (!Number.isInteger(firstModelReadyYear)) throw new Error("--first-model-ready-year must be an integer");
if (sourceStart > emitStart) throw new Error("--source-start must be <= --emit-start");
if (emitStart > emitEnd) throw new Error("--emit-start must be <= --emit-end");
if (emitEnd > sourceEnd) throw new Error("--emit-end must be <= --source-end");

const featureContractPath = path.join(ROOT, "contracts", "l1-feature-set-contract-v1.json");
const smallSampleContractPath = path.join(ROOT, "contracts", "l1-small-sample-contract-v1.json");
const generation = {
  source_backfill_sha: sourceSha,
  ml_source_sha: mlSourceSha,
  dataset_version: ML_DATASET_VERSION,
  feature_schema_version: ML_FEATURE_SCHEMA_VERSION,
  leakage_policy: ML_LEAKAGE_POLICY,
  prediction_phase: predictionPhase,
  feature_sets: featureSets,
  history_windows: historyWindows,
  small_sample_policy: smallSamplePolicy,
  feature_contract_sha256: await sha256File(featureContractPath),
  small_sample_contract_sha256: await sha256File(smallSampleContractPath),
  source_start: sourceStart,
  source_end: sourceEnd,
  emit_start: emitStart,
  emit_end: emitEnd,
};
generation.generation_id = stableHash(generation).slice(0, 16);

const dailyDir = path.join(sourceRoot, "data", "daily");
const dailyNames = (await readdir(dailyDir))
  .filter(name => /^\d{4}-\d{2}-\d{2}\.jsonl\.gz$/.test(name))
  .sort()
  .filter(name => {
    const date = name.slice(0, 10);
    return date >= sourceStart && date <= sourceEnd;
  });

console.log("L1_YEARLY_SNAPSHOT_BUILD_START");
console.log(JSON.stringify({
  generation,
  source_files: dailyNames.length,
  streaming: true,
  persistent_upload: false,
}, null, 2));

const startedAt = Date.now();
const targetScan = await collectTargetHorseIds(dailyDir, dailyNames);
const lineage = featureSets.includes("PEDIGREE")
  ? await loadLineageMap(sourceRoot, targetScan.ids)
  : { map: new Map(), files_read: 0, records_read: 0, targets: 0 };

const mlProcessor = createMlDatasetProcessor({
  historyWindows,
  includeAutoFeatures: featureSets.includes("AUTO"),
  includeBackfillFeatures: featureSets.includes("BACKFILL"),
  includePedigreeFeatures: featureSets.includes("PEDIGREE"),
  includeActorFeatures: featureSets.includes("ACTOR"),
  includeOpponentRelationshipFeatures: featureSets.includes("OPPONENT"),
  includeTimePaceFeatures: featureSets.includes("TIME_PACE"),
  lineageByHorse: lineage.map,
  smallSamplePolicy,
});
const extraProcessor = createHistoricalExtraProcessor(historyWindows);
const generationDir = path.join(outRoot, generation.generation_id);
const writer = await createYearlySnapshotWriter({
  outDir: generationDir,
  generation,
  firstModelReadyYear,
});

let sourceRaces = 0;
let emittedRows = 0;
let lapRows = 0;
let styleRows = 0;
let processedDates = 0;

for (const name of dailyNames) {
  const rows = parseDailyFile(await readFile(path.join(dailyDir, name)));
  sourceRaces += rows.length;
  for (const group of groupByDate(rows)) {
    const baseRows = mlProcessor.processDay(group.rows, group.date, {
      startDate: emitStart,
      endDate: emitEnd,
    });
    const extras = extraProcessor.processDay(group.rows, group.date, {
      startDate: emitStart,
      endDate: emitEnd,
    });

    const staged = baseRows.map(row => {
      const extra = extras.get(String(row.race_id) + "|" + String(row.horse_id)) ?? {};
      const selectedExtra = {};
      for (const [key, value] of Object.entries(extra)) {
        if (featureSets.includes("LAP") && key.startsWith("lap_")) selectedExtra[key] = value;
        if (featureSets.includes("STYLE") && key.startsWith("style_")) selectedExtra[key] = value;
        if (featureSets.includes("DISTANCE") && key.startsWith("distx_")) selectedExtra[key] = value;
      }
      if ((selectedExtra.lap_recent_races_measured ?? 0) > 0) lapRows += 1;
      if ((selectedExtra.style_recent_races_measured ?? 0) > 0) styleRows += 1;
      const selected = selectFeatureFamilies({
        ...row.features,
        ...selectedExtra,
      }, featureSets);
      return {
        ...row,
        prediction_phase: predictionPhase,
        feature_sets: featureSets,
        history_windows: historyWindows,
        small_sample_policy: smallSamplePolicy,
        features: applyPredictionPhase(selected, predictionPhase),
      };
    });

    await writer.writeRows(staged);
    emittedRows += staged.length;
    processedDates += 1;
  }
}

const finishedAt = Date.now();
const result = await writer.close({
  source_files: dailyNames.length,
  source_races: sourceRaces,
  emitted_rows: emittedRows,
  lap_rows: lapRows,
  lap_coverage: emittedRows ? lapRows / emittedRows : 0,
  style_rows: styleRows,
  style_coverage: emittedRows ? styleRows / emittedRows : 0,
  pedigree_lineage_targets: lineage.targets,
  pedigree_lineage_found: lineage.map.size,
  pedigree_horse_pack_files_read: lineage.files_read,
  pedigree_horse_records_read: lineage.records_read,
  processed_dates: processedDates,
  target_scan_races: targetScan.races,
  processor_state_sizes: mlProcessor.debugStateSizes(),
  extra_state_sizes: extraProcessor.debugStateSizes(),
  total_build_ms: finishedAt - startedAt,
  streaming: true,
  persistent_upload: false,
});
const manifest = result.manifest;

console.log("L1_YEARLY_SUPERSET_SNAPSHOT_READY");
console.log(JSON.stringify({
  manifest: result.manifestPath,
  generation_id: generation.generation_id,
  feature_sets: featureSets,
  first_model_ready_year: firstModelReadyYear,
  years: manifest.years,
  total_bytes: manifest.total_bytes,
  total_rows: manifest.total_rows,
  total_build_ms: manifest.total_build_ms,
  source_races: manifest.source_races,
  pedigree_lineage_targets: manifest.pedigree_lineage_targets,
  pedigree_lineage_found: manifest.pedigree_lineage_found,
  processor_state_sizes: manifest.processor_state_sizes,
  persistent_upload: false,
}, null, 2));
