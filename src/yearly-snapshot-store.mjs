import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, stat, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createGunzip, createGzip } from "node:zlib";
import { createInterface } from "node:readline";
import { once } from "node:events";
import path from "node:path";

export const YEARLY_SNAPSHOT_STORE_VERSION = 1;

function yearOf(row) {
  const date = String(row?.race_date ?? row?.features?.race_date ?? "").slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) return null;
  return { year: Number(date.slice(0, 4)), date };
}

async function sha256File(file) {
  const hash = createHash("sha256");
  const stream = createReadStream(file);
  stream.on("data", chunk => hash.update(chunk));
  await once(stream, "end");
  return hash.digest("hex");
}

function snapshotName(year) {
  return "snapshot-" + year + ".jsonl.gz";
}

export async function createYearlySnapshotWriter({
  outDir,
  generation,
  firstModelReadyYear,
  manifestName = "manifest.json",
} = {}) {
  if (!outDir) throw new Error("outDir is required");
  if (!generation || typeof generation !== "object") throw new Error("generation metadata is required");
  if (!Number.isInteger(firstModelReadyYear)) throw new Error("firstModelReadyYear must be an integer");

  await mkdir(outDir, { recursive: true });
  const writers = new Map();
  const statsByYear = new Map();
  let closed = false;

  function ensureYear(year) {
    if (writers.has(year)) return writers.get(year);
    const file = path.join(outDir, snapshotName(year));
    const gzip = createGzip({ level: 6 });
    const sink = createWriteStream(file);
    gzip.pipe(sink);
    const value = { file, gzip, sink };
    writers.set(year, value);
    statsByYear.set(year, {
      rows: 0,
      races: new Set(),
      horses: new Set(),
      min_date: null,
      max_date: null,
    });
    return value;
  }

  async function writeRows(rows) {
    if (closed) throw new Error("yearly snapshot writer is already closed");
    for (const row of rows ?? []) {
      const info = yearOf(row);
      if (!info) throw new Error("snapshot row missing valid race_date");
      const writer = ensureYear(info.year);
      const ys = statsByYear.get(info.year);
      ys.rows += 1;
      if (row.race_id != null) ys.races.add(String(row.race_id));
      if (row.horse_id != null) ys.horses.add(String(row.horse_id));
      ys.min_date = ys.min_date == null || info.date < ys.min_date ? info.date : ys.min_date;
      ys.max_date = ys.max_date == null || info.date > ys.max_date ? info.date : ys.max_date;
      if (!writer.gzip.write(JSON.stringify(row) + "\n")) await once(writer.gzip, "drain");
    }
  }

  async function close(extraManifest = {}) {
    if (closed) throw new Error("yearly snapshot writer is already closed");
    closed = true;
    for (const { gzip } of writers.values()) gzip.end();
    await Promise.all([...writers.values()].map(({ sink }) => once(sink, "close")));

    const years = [];
    for (const year of [...writers.keys()].sort((a, b) => a - b)) {
      const writer = writers.get(year);
      const ys = statsByYear.get(year);
      const fileStat = await stat(writer.file);
      years.push({
        year,
        file: path.basename(writer.file),
        bytes: fileStat.size,
        sha256: await sha256File(writer.file),
        rows: ys.rows,
        races: ys.races.size,
        horses: ys.horses.size,
        min_date: ys.min_date,
        max_date: ys.max_date,
        model_ready: year >= firstModelReadyYear,
        history_status: year >= firstModelReadyYear
          ? "COMPLETE_FROM_INCLUDED_WARMUP"
          : "PROVISIONAL_TRUNCATED_PRIOR_HISTORY",
      });
    }

    const manifest = {
      contract: "L1_YEARLY_SUPERSET_SNAPSHOT_V1",
      store_version: YEARLY_SNAPSHOT_STORE_VERSION,
      generation,
      first_model_ready_year: firstModelReadyYear,
      years,
      total_bytes: years.reduce((sum, item) => sum + item.bytes, 0),
      total_rows: years.reduce((sum, item) => sum + item.rows, 0),
      ...extraManifest,
    };
    const manifestPath = path.join(outDir, manifestName);
    await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n", "utf8");
    return { manifest, manifestPath };
  }

  return { writeRows, close };
}

export async function partitionDatasetByYear({
  inputDataset,
  outDir,
  generation,
  firstModelReadyYear,
  manifestName = "manifest.json",
} = {}) {
  if (!inputDataset) throw new Error("inputDataset is required");
  const writer = await createYearlySnapshotWriter({
    outDir,
    generation,
    firstModelReadyYear,
    manifestName,
  });
  const gunzip = createGunzip();
  const source = createReadStream(inputDataset);
  source.pipe(gunzip);
  const rl = createInterface({ input: gunzip, crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    await writer.writeRows([JSON.parse(line)]);
  }
  return writer.close();
}
