#!/usr/bin/env node
import { mkdir, readFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ML_DATASET_VERSION,
  ML_FEATURE_SCHEMA_VERSION,
  ML_LEAKAGE_POLICY,
} from "../src/ml-dataset.mjs";
import { partitionDatasetByYear } from "../src/yearly-snapshot-store.mjs";
import { L1_FEATURE_SET_CONTRACT } from "../src/l1-feature-contract.mjs";
import { DEFAULT_SMALL_SAMPLE_POLICY } from "../src/small-sample-feature-utils.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");

function arg(name) {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : null;
}

function dateArg(name, fallback) {
  const value = arg(name) ?? fallback;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error(`invalid ${name}: ${value}`);
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

async function run(command, args) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args.map(String), {
      cwd: ROOT,
      stdio: "inherit",
      env: process.env,
    });
    child.on("error", reject);
    child.on("exit", code => code === 0 ? resolve() : reject(new Error(`${command} exited ${code}`)));
  });
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
const featureSets = (arg("--feature-sets") ?? allFeatureSets.join(","))
  .split(",").map(x => x.trim().toUpperCase()).filter(Boolean);
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

const generationDir = path.join(outRoot, generation.generation_id);
const tempDir = path.join(generationDir, ".tmp");
const tempDataset = path.join(tempDir, "superset-all-years.jsonl.gz");
await mkdir(tempDir, { recursive: true });

const buildArgs = [
  "research/build-staged-dataset.mjs",
  "--source-root", sourceRoot,
  "--output", tempDataset,
  "--source-start", sourceStart,
  "--source-end", sourceEnd,
  "--emit-start", emitStart,
  "--emit-end", emitEnd,
  "--feature-sets", featureSets.join(","),
  "--prediction-phase", predictionPhase,
  "--history-recent-form", historyWindows.recent_form,
  "--history-style-last3f", historyWindows.style_last3f,
  "--history-suitability", historyWindows.suitability,
  "--history-opponent", historyWindows.opponent,
  "--history-auto-rolling", historyWindows.auto_rolling,
  "--history-actor-recent", historyWindows.actor_recent,
  "--history-time-pace", historyWindows.time_pace,
  "--shrinkage-rate-strength", smallSamplePolicy.ratePriorStrength,
  "--shrinkage-mean-strength", smallSamplePolicy.meanPriorStrength,
  "--min-specific-observations", smallSamplePolicy.minSpecificObservations,
];

console.log("L1_YEARLY_SNAPSHOT_BUILD_START");
console.log(JSON.stringify({ generation, generation_dir: generationDir }, null, 2));
const startedAt = Date.now();
await run(process.execPath, buildArgs);
const generatedAt = Date.now();

const { manifest, manifestPath } = await partitionDatasetByYear({
  inputDataset: tempDataset,
  outDir: generationDir,
  generation: {
    ...generation,
    superset_build_ms: generatedAt - startedAt,
  },
  firstModelReadyYear,
});

await rm(tempDir, { recursive: true, force: true });
const finishedAt = Date.now();
manifest.total_build_ms = finishedAt - startedAt;
await import("node:fs/promises").then(({ writeFile }) =>
  writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n", "utf8")
);

console.log("L1_YEARLY_SUPERSET_SNAPSHOT_READY");
console.log(JSON.stringify({
  manifest: manifestPath,
  generation_id: generation.generation_id,
  feature_sets: featureSets,
  first_model_ready_year: firstModelReadyYear,
  years: manifest.years,
  total_bytes: manifest.total_bytes,
  total_rows: manifest.total_rows,
  superset_build_ms: generatedAt - startedAt,
  total_build_ms: finishedAt - startedAt,
  persistent_upload: false,
}, null, 2));
