#!/usr/bin/env node
import { createReadStream } from "node:fs";
import { mkdir, readdir, writeFile } from "node:fs/promises";
import { createGunzip } from "node:zlib";
import { createInterface } from "node:readline";
import path from "node:path";
import { createBackfillReadinessAccumulator } from "../src/backfill-readiness.mjs";

function arg(name) {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : null;
}

function numberArg(name, fallback) {
  const raw = arg(name);
  if (raw == null) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) throw new Error(`invalid ${name}: ${raw}`);
  return n;
}

function dateArg(name, fallback) {
  const raw = arg(name) ?? fallback;
  if (raw == null) return null;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(raw)) throw new Error(`invalid ${name}: ${raw}`);
  return raw;
}

function raceDate(row) {
  const raw = row?.race?.actual_date ?? row?.race?.scheduled_date ?? "";
  const text = String(raw).slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

async function readGzipJsonl(file, onRow) {
  const input = createReadStream(file).pipe(createGunzip());
  const rl = createInterface({ input, crlfDelay: Infinity });
  let lineNo = 0;
  for await (const line of rl) {
    lineNo += 1;
    if (!line.trim()) continue;
    try {
      await onRow(JSON.parse(line));
    } catch (error) {
      throw new Error(`${file}:${lineNo}: ${error.message}`);
    }
  }
}

const sourceRoot = path.resolve(arg("--source-root") ?? "../KEIBA-BACKFILL");
const start = dateArg("--start", null);
const end = dateArg("--end", null);
const reportOut = path.resolve(arg("--report-out") ?? "out/backfill-readiness.json");
const minCoreKnownCoverage = numberArg("--min-core-known-coverage", 0.98);
const maxInvalidRate = numberArg("--max-invalid-rate", 0);
const maxYearGap = numberArg("--max-year-gap", 0.10);

if (start && end && start > end) throw new Error("--start must be <= --end");

const dailyDir = path.join(sourceRoot, "data", "daily");
const files = (await readdir(dailyDir))
  .filter(name => /^\d{4}-\d{2}-\d{2}\.jsonl\.gz$/.test(name))
  .sort()
  .filter(name => {
    const date = name.slice(0, 10);
    return (!start || date >= start) && (!end || date <= end);
  });

if (!files.length) throw new Error("no BACKFILL daily files matched");

const acc = createBackfillReadinessAccumulator();
let rows = 0;

for (const name of files) {
  await readGzipJsonl(path.join(dailyDir, name), row => {
    const date = raceDate(row);
    if (!date) return;
    if (start && date < start) return;
    if (end && date > end) return;
    acc.add(row);
    rows += 1;
  });
}

const report = acc.finish({
  minCoreKnownCoverage,
  maxInvalidRate,
  maxYearGap,
});

report.source = {
  root: sourceRoot,
  start,
  end,
  daily_files: files.length,
  race_rows: rows,
};

await mkdir(path.dirname(reportOut), { recursive: true });
await writeFile(reportOut, JSON.stringify(report, null, 2) + "\n", "utf8");

console.log("BACKFILL_READINESS_REPORT");
console.log(JSON.stringify({
  ready_for_l1_research: report.ready_for_l1_research,
  failed_gate_count: report.failed_gate_count,
  source: report.source,
  core_fields: Object.fromEntries(
    report.core_fields.map(field => [
      field,
      {
        known_coverage: report.overall.fields[field].known_coverage,
        missing_rate: report.overall.fields[field].missing_rate,
        unknown_rate: report.overall.fields[field].unknown_rate,
        invalid_rate: report.overall.fields[field].invalid_rate,
      },
    ]),
  ),
  margin: report.overall.margin,
  report_out: reportOut,
}, null, 2));

if (!report.ready_for_l1_research) process.exitCode = 2;
