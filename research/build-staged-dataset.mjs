import { mkdir, readFile, readdir } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { once } from "node:events";
import path from "node:path";
import { gunzipSync, createGzip } from "node:zlib";
import { buildMlDataset } from "../src/ml-dataset.mjs";

function arg(name) {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : null;
}

function dateArg(name, fallback) {
  const value = arg(name) ?? fallback;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`invalid ${name}: ${value}`);
  return value;
}

const sourceRoot = path.resolve(arg("--source-root") ?? "../KEIBA-BACKFILL");
const output = path.resolve(arg("--output") ?? "out/l1-dataset.jsonl.gz");
const sourceStart = dateArg("--source-start", "2021-01-01");
const sourceEnd = dateArg("--source-end", "2025-12-31");
const emitStart = dateArg("--emit-start", "2022-01-01");
const emitEnd = dateArg("--emit-end", "2025-12-31");
const historyLimit = Number(arg("--history-limit") ?? 5);

if (!Number.isInteger(historyLimit) || historyLimit < 1 || historyLimit > 100) {
  throw new Error("--history-limit must be an integer from 1 to 100");
}
if (sourceStart > emitStart) throw new Error("--source-start must be <= --emit-start");
if (emitStart > emitEnd) throw new Error("--emit-start must be <= --emit-end");
if (emitEnd > sourceEnd) throw new Error("--emit-end must be <= --source-end");

function finite(value) {
  if (value == null) return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  const clean = values.map(finite).filter(v => v != null);
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

function rate(items, predicate) {
  return items.length ? items.filter(predicate).length / items.length : null;
}

function raceDate(row) {
  const value = row?.race?.actual_date ?? row?.race?.scheduled_date ?? "";
  const text = String(value).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

function resultMap(row) {
  return new Map((row?.results ?? []).map(r => [String(r?.horse_id ?? ""), r]));
}

function eligible(entry, result) {
  if (!entry || !result) return false;
  const es = String(entry.entry_status ?? "").toUpperCase();
  const rs = String(result.result_status ?? "").toUpperCase();
  return !["SCRATCHED", "EXCLUDED"].includes(es) && !["SCRATCHED", "EXCLUDED"].includes(rs);
}

function lapSummary(laps) {
  const values = (laps ?? []).map(x => finite(x?.lap_seconds)).filter(v => v != null && v > 0);
  if (!values.length) return null;
  const first = values.slice(0, Math.min(3, values.length));
  const last = values.slice(Math.max(0, values.length - 3));
  const avg = mean(values);
  const variance = mean(values.map(v => (v - avg) ** 2));
  return {
    first3: mean(first),
    last3: mean(last),
    last1: values.at(-1),
    variance,
    delta_last3_first3: mean(last) - mean(first),
  };
}

function parseCorners(raw, fieldSize) {
  const positions = String(raw ?? "").match(/\d+/g)?.map(Number).filter(Number.isFinite) ?? [];
  if (!positions.length) return null;
  const first = positions[0];
  const last = positions.at(-1);
  const size = finite(fieldSize);
  return {
    first,
    last,
    gain: first - last,
    first_ratio: size && size > 0 ? first / size : null,
    last_ratio: size && size > 0 ? last / size : null,
    front: first <= 3,
    finished_front: last <= 3,
    improved: last < first,
  };
}

function historicalExtras(history, currentDistance) {
  const recent = history.slice(-historyLimit).reverse();
  const finished = recent.filter(x =>
    String(x.result?.result_status ?? "").toUpperCase() === "FINISHED" &&
    finite(x.result?.official_finish_position) != null
  );

  const lapItems = recent.map(x => lapSummary(x.laps)).filter(Boolean);
  const prevLap = lapItems[0] ?? null;
  const styleItems = recent.map(x => parseCorners(x.result?.corner_raw, x.fieldSize)).filter(Boolean);
  const prevStyle = styleItems[0] ?? null;

  const current = finite(currentDistance);
  const sameBand = finished.filter(x => {
    const d = finite(x.race?.distance_m);
    return d != null && current != null && Math.abs(d - current) <= 200;
  });
  const shorter = finished.filter(x => {
    const d = finite(x.race?.distance_m);
    return d != null && current != null && d < current;
  });
  const longer = finished.filter(x => {
    const d = finite(x.race?.distance_m);
    return d != null && current != null && d > current;
  });
  const distances = finished.map(x => finite(x.race?.distance_m)).filter(v => v != null);

  return {
    lap_previous_first3_avg: prevLap?.first3 ?? null,
    lap_previous_last3_avg: prevLap?.last3 ?? null,
    lap_previous_last1: prevLap?.last1 ?? null,
    lap_previous_variance: prevLap?.variance ?? null,
    lap_previous_delta_last3_first3: prevLap?.delta_last3_first3 ?? null,
    lap_recent_races_measured: lapItems.length,
    lap_recent_avg_first3: mean(lapItems.map(x => x.first3)),
    lap_recent_avg_last3: mean(lapItems.map(x => x.last3)),
    lap_recent_avg_last1: mean(lapItems.map(x => x.last1)),
    lap_recent_avg_variance: mean(lapItems.map(x => x.variance)),
    lap_recent_avg_delta_last3_first3: mean(lapItems.map(x => x.delta_last3_first3)),

    style_previous_first_position: prevStyle?.first ?? null,
    style_previous_last_position: prevStyle?.last ?? null,
    style_previous_position_gain: prevStyle?.gain ?? null,
    style_previous_first_ratio: prevStyle?.first_ratio ?? null,
    style_previous_last_ratio: prevStyle?.last_ratio ?? null,
    style_recent_races_measured: styleItems.length,
    style_recent_avg_first_position: mean(styleItems.map(x => x.first)),
    style_recent_avg_last_position: mean(styleItems.map(x => x.last)),
    style_recent_avg_position_gain: mean(styleItems.map(x => x.gain)),
    style_recent_avg_first_ratio: mean(styleItems.map(x => x.first_ratio)),
    style_recent_avg_last_ratio: mean(styleItems.map(x => x.last_ratio)),
    style_recent_front_rate: rate(styleItems, x => x.front),
    style_recent_finish_front_rate: rate(styleItems, x => x.finished_front),
    style_recent_improve_rate: rate(styleItems, x => x.improved),

    distx_recent_avg_distance_m: mean(distances),
    distx_current_minus_recent_avg_m:
      current != null && distances.length ? current - mean(distances) : null,
    distx_same_band_starts: sameBand.length,
    distx_same_band_top3_rate: rate(sameBand, x => Number(x.result.official_finish_position) <= 3),
    distx_shorter_starts: shorter.length,
    distx_shorter_top3_rate: rate(shorter, x => Number(x.result.official_finish_position) <= 3),
    distx_longer_starts: longer.length,
    distx_longer_top3_rate: rate(longer, x => Number(x.result.official_finish_position) <= 3),
  };
}

function buildHistoricalExtraMap(rows) {
  const normalized = rows
    .map(row => ({ row, date: raceDate(row) }))
    .filter(x => x.date)
    .sort((a, b) => a.date.localeCompare(b.date) ||
      String(a.row?.race?.race_id ?? "").localeCompare(String(b.row?.race?.race_id ?? "")));

  const historyByHorse = new Map();
  const extras = new Map();
  let index = 0;

  while (index < normalized.length) {
    const date = normalized[index].date;
    const day = [];
    while (index < normalized.length && normalized[index].date === date) {
      day.push(normalized[index].row);
      index += 1;
    }

    for (const row of day) {
      const raceId = String(row?.race?.race_id ?? "");
      const results = resultMap(row);
      for (const entry of row?.entries ?? []) {
        const horseId = String(entry?.horse_id ?? "");
        const result = results.get(horseId);
        if (!raceId || !horseId || !eligible(entry, result)) continue;
        if (date < emitStart || date > emitEnd) continue;
        const history = historyByHorse.get(horseId) ?? [];
        extras.set(
          raceId + "|" + horseId,
          historicalExtras(history, row?.race?.distance_m),
        );
      }
    }

    for (const row of day) {
      const results = resultMap(row);
      const fieldSize = (row?.entries ?? []).filter(entry => {
        const result = results.get(String(entry?.horse_id ?? ""));
        return eligible(entry, result);
      }).length;
      for (const entry of row?.entries ?? []) {
        const horseId = String(entry?.horse_id ?? "");
        const result = results.get(horseId);
        if (!horseId || !eligible(entry, result)) continue;
        const history = historyByHorse.get(horseId) ?? [];
        history.push({
          race: row.race ?? {},
          result,
          laps: row.laps ?? [],
          fieldSize,
        });
        if (history.length > 100) history.splice(0, history.length - 100);
        historyByHorse.set(horseId, history);
      }
    }
  }
  return extras;
}

async function loadPedigreeFacts(targetHorseIds) {
  const files = (await readdir(path.join(sourceRoot, "data", "horses")))
    .filter(name => name.endsWith(".jsonl.gz"))
    .sort();
  const facts = new Map();
  let parsedRecords = 0;
  for (const name of files) {
    const zipped = await readFile(path.join(sourceRoot, "data", "horses", name));
    const text = gunzipSync(zipped).toString("utf8").trim();
    if (!text) continue;
    for (const line of text.split("\n")) {
      const row = JSON.parse(line);
      const horseId = String(row?.horse_id ?? "");
      if (!targetHorseIds.has(horseId)) continue;
      parsedRecords += 1;
      const nodes = row?.pedigree ?? [];
      const at = (generation, slot) =>
        nodes.find(x => Number(x?.generation) === generation && Number(x?.slot) === slot) ?? null;
      const sire = at(1, 0);
      const dam = at(1, 1);
      const sireSire = at(2, 0);
      const damSire = at(2, 2);
      facts.set(horseId, {
        pedigree_sire_id: sire?.ancestor_id ?? null,
        pedigree_dam_id: dam?.ancestor_id ?? null,
        pedigree_siresire_id: sireSire?.ancestor_id ?? null,
        pedigree_damsire_id: damSire?.ancestor_id ?? null,
        pedigree_known_nodes: nodes.filter(x => x?.ancestor_id || x?.ancestor_name).length,
      });
    }
  }
  return { facts, files: files.length, matchedRecords: parsedRecords };
}

const dailyNames = (await readdir(path.join(sourceRoot, "data", "daily")))
  .filter(name => /^\d{4}-\d{2}-\d{2}\.jsonl\.gz$/.test(name))
  .sort()
  .filter(name => {
    const date = name.slice(0, 10);
    return date >= sourceStart && date <= sourceEnd;
  });

const raceRows = [];
for (const name of dailyNames) {
  const zipped = await readFile(path.join(sourceRoot, "data", "daily", name));
  const text = gunzipSync(zipped).toString("utf8").trim();
  if (text) raceRows.push(...text.split("\n").map(JSON.parse));
}

const base = buildMlDataset(raceRows, {
  startDate: emitStart,
  endDate: emitEnd,
  historyLimit,
});
const extras = buildHistoricalExtraMap(raceRows);
const targetHorseIds = new Set(base.map(row => String(row.horse_id)));
const pedigree = await loadPedigreeFacts(targetHorseIds);

let lapRows = 0;
let styleRows = 0;
let pedigreeRows = 0;
const staged = base.map(row => {
  const extra = extras.get(String(row.race_id) + "|" + String(row.horse_id)) ?? {};
  const ped = pedigree.facts.get(String(row.horse_id)) ?? {
    pedigree_sire_id: null,
    pedigree_dam_id: null,
    pedigree_siresire_id: null,
    pedigree_damsire_id: null,
    pedigree_known_nodes: 0,
  };
  if ((extra.lap_recent_races_measured ?? 0) > 0) lapRows += 1;
  if ((extra.style_recent_races_measured ?? 0) > 0) styleRows += 1;
  if ((ped.pedigree_known_nodes ?? 0) > 0) pedigreeRows += 1;
  return {
    ...row,
    features: {
      ...row.features,
      ...extra,
      ...ped,
    },
  };
});

await mkdir(path.dirname(output), { recursive: true });
const gzip = createGzip({ level: 6 });
const sink = createWriteStream(output);
gzip.pipe(sink);
for (const row of staged) {
  if (!gzip.write(JSON.stringify(row) + "\n")) {
    await once(gzip, "drain");
  }
}
gzip.end();
await once(sink, "close");

console.log("ML_STAGED_DATASET_READY");
console.log(JSON.stringify({
  output,
  source_root: sourceRoot,
  source_start: sourceStart,
  source_end: sourceEnd,
  emit_start: emitStart,
  emit_end: emitEnd,
  history_limit: historyLimit,
  source_files: dailyNames.length,
  source_races: raceRows.length,
  rows: staged.length,
  races: new Set(staged.map(row => row.race_id)).size,
  horses: targetHorseIds.size,
  lap_rows: lapRows,
  lap_coverage: staged.length ? lapRows / staged.length : 0,
  style_rows: styleRows,
  style_coverage: staged.length ? styleRows / staged.length : 0,
  horse_pack_files: pedigree.files,
  pedigree_rows: pedigreeRows,
  pedigree_coverage: staged.length ? pedigreeRows / staged.length : 0,
  pedigree_horses: pedigree.facts.size,
  pedigree_matched_records: pedigree.matchedRecords,
}, null, 2));
