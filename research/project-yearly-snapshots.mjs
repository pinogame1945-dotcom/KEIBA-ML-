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
const actorPrefixes = String(arg("--actor-prefixes") ?? "")
  .split(",").map(x => x.trim()).filter(Boolean);
const ALLOWED_ACTOR_PREFIXES = new Set([
  "actor_jockey_",
  "actor_trainer_",
  "actor_horse_jockey_",
]);
for (const prefix of actorPrefixes) {
  if (!ALLOWED_ACTOR_PREFIXES.has(prefix)) {
    throw new Error("invalid --actor-prefixes value: " + prefix);
  }
}
if (actorPrefixes.length && !featureSets.includes("ACTOR")) {
  throw new Error("--actor-prefixes requires ACTOR feature set");
}

const autoSlices = String(arg("--auto-slices") ?? "")
  .split(",").map(x => x.trim().toUpperCase()).filter(Boolean);
const ALLOWED_AUTO_SLICES = new Set(["ROLLING", "CONDITION", "FIELD", "PAIR"]);
for (const slice of autoSlices) {
  if (!ALLOWED_AUTO_SLICES.has(slice)) {
    throw new Error("invalid --auto-slices value: " + slice);
  }
}
if (autoSlices.length && !featureSets.includes("AUTO")) {
  throw new Error("--auto-slices requires AUTO feature set");
}

function keepAutoFeature(key) {
  if (!key.startsWith("auto_") || !autoSlices.length) return true;
  if (key.startsWith("auto_field_")) return autoSlices.includes("FIELD");
  if (key.startsWith("auto_pair_")) return autoSlices.includes("PAIR");
  if (/^auto_(?:margin_gap|finish)_(?:same_|within_)/.test(key)) {
    return autoSlices.includes("CONDITION");
  }
  return autoSlices.includes("ROLLING");
}

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
      research_context: {
        ...(row.research_context ?? {}),
        race_class_normalized: row.features?.backfill_race_class_normalized ?? null,
      },
      features: (() => {
        const phased = applyPredictionPhase(
          selectFeatureFamilies(row.features ?? {}, featureSets),
          predictionPhase,
        );
        if (!actorPrefixes.length && !autoSlices.length) return phased;
        return Object.fromEntries(Object.entries(phased).filter(([key]) => {
          if (key.startsWith("actor_") && actorPrefixes.length) {
            return actorPrefixes.some(prefix => key.startsWith(prefix));
          }
          if (key.startsWith("auto_")) return keepAutoFeature(key);
          return true;
        }));
      })(),
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
  actor_prefixes: actorPrefixes,
  auto_slices: autoSlices,
  rows,
  output,
}, null, 2));
