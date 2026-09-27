#!/usr/bin/env node
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { createGzip } from "node:zlib";
import { once } from "node:events";
import os from "node:os";
import path from "node:path";
import { partitionDatasetByYear } from "../src/yearly-snapshot-store.mjs";

const dir = await mkdtemp(path.join(os.tmpdir(), "l1-yearly-snapshot-"));
const input = path.join(dir, "input.jsonl.gz");
const out = path.join(dir, "out");
const gzip = createGzip();
const sink = createWriteStream(input);
gzip.pipe(sink);
for (const row of [
  { race_date: "2022-12-31", race_id: "R1", horse_id: "H1", features: { x: 1 } },
  { race_date: "2023-01-01", race_id: "R2", horse_id: "H1", features: { x: 2 } },
  { race_date: "2023-01-01", race_id: "R2", horse_id: "H2", features: { x: 3 } },
  { race_date: "2024-02-03", race_id: "R3", horse_id: "H3", features: { x: 4 } },
]) gzip.write(JSON.stringify(row) + "\n");
gzip.end();
await once(sink, "close");

const { manifest } = await partitionDatasetByYear({
  inputDataset: input,
  outDir: out,
  generation: { generation_id: "TEST", feature_schema_version: 8 },
  firstModelReadyYear: 2023,
});

assert.deepEqual(manifest.years.map(x => x.year), [2022, 2023, 2024]);
assert.equal(manifest.years[0].model_ready, false);
assert.equal(manifest.years[0].history_status, "PROVISIONAL_TRUNCATED_PRIOR_HISTORY");
assert.equal(manifest.years[1].model_ready, true);
assert.equal(manifest.years[1].rows, 2);
assert.equal(manifest.years[1].races, 1);
assert.equal(manifest.years[1].horses, 2);
assert.equal(manifest.years[2].model_ready, true);
assert.ok(manifest.years.every(x => x.bytes > 0));
assert.ok(manifest.years.every(x => /^[a-f0-9]{64}$/.test(x.sha256)));

const persisted = JSON.parse(await readFile(path.join(out, "manifest.json"), "utf8"));
assert.equal(persisted.contract, "L1_YEARLY_SUPERSET_SNAPSHOT_V1");
assert.equal(persisted.total_rows, 4);

await rm(dir, { recursive: true, force: true });
console.log("L1_YEARLY_SNAPSHOT_SMOKE_OK");
