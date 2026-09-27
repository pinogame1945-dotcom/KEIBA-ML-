#!/usr/bin/env node
import { createReadStream, createWriteStream } from "node:fs";
import { createGunzip, createGzip } from "node:zlib";
import { createInterface } from "node:readline";
import { once } from "node:events";
import {
  applyPredictionPhase,
  normalizeFeatureSets,
  normalizePredictionPhase,
  selectFeatureFamilies,
} from "../src/l1-feature-contract.mjs";

function arg(name) {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : null;
}

const inputs = String(arg("--inputs") ?? "").split(",").map(x => x.trim()).filter(Boolean);
const output = arg("--output");
const featureSets = normalizeFeatureSets(arg("--feature-sets"));
const predictionPhase = normalizePredictionPhase(arg("--prediction-phase"));

if (!inputs.length) throw new Error("--inputs is required");
if (!output) throw new Error("--output is required");

const gzip = createGzip({ level: 6 });
const sink = createWriteStream(output);
gzip.pipe(sink);

let rows = 0;
for (const input of inputs) {
  const source = createReadStream(input);
  const decoded = input.endsWith(".gz") ? source.pipe(createGunzip()) : source;
  const rl = createInterface({ input: decoded, crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    const projected = {
      ...row,
      prediction_phase: predictionPhase,
      feature_sets: featureSets,
      features: applyPredictionPhase(
        selectFeatureFamilies(row.features ?? {}, featureSets),
        predictionPhase,
      ),
    };
    if (!gzip.write(JSON.stringify(projected) + "\n")) await once(gzip, "drain");
    rows += 1;
  }
}
gzip.end();
await once(sink, "close");
console.log(JSON.stringify({
  contract: "L1_SNAPSHOT_PROJECTION_V1",
  feature_sets: featureSets,
  prediction_phase: predictionPhase,
  rows,
  output,
}, null, 2));
