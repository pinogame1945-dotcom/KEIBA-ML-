import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONTRACT_PATH = path.join(HERE, "..", "contracts", "l1-feature-set-contract-v1.json");
export const L1_FEATURE_SET_CONTRACT = JSON.parse(readFileSync(CONTRACT_PATH, "utf8"));

export const FEATURE_PREFIXES = Object.fromEntries(
  Object.entries(L1_FEATURE_SET_CONTRACT.feature_sets).map(([name, spec]) => [name, spec.prefixes ?? []]),
);
export const ALL_FAMILY_PREFIXES = [...new Set(Object.values(FEATURE_PREFIXES).flat())];

function unique(items) {
  return [...new Set(items)];
}

export function normalizePredictionPhase(value) {
  const phase = String(value ?? L1_FEATURE_SET_CONTRACT.default_prediction_phase).trim().toUpperCase();
  if (!Object.hasOwn(L1_FEATURE_SET_CONTRACT.prediction_phases, phase)) {
    throw new Error(`invalid prediction phase: ${value}`);
  }
  return phase;
}

export function normalizeFeatureSets(raw, legacyStage = null) {
  let requested;
  if (raw != null && String(raw).trim()) {
    requested = String(raw).split(",").map(v => v.trim().toUpperCase()).filter(Boolean);
  } else if (legacyStage != null && String(legacyStage).trim()) {
    requested = L1_FEATURE_SET_CONTRACT.legacy_stage_map[String(legacyStage).trim()];
    if (!requested) throw new Error(`invalid legacy stage: ${legacyStage}`);
  } else {
    requested = L1_FEATURE_SET_CONTRACT.default_feature_sets;
  }
  requested = unique(requested);
  const invalid = requested.filter(name => !Object.hasOwn(FEATURE_PREFIXES, name));
  if (invalid.length) throw new Error(`invalid feature set(s): ${invalid.join(", ")}`);
  if (!requested.includes("BASE")) requested.unshift("BASE");
  return requested;
}

export function legacyStageToFeatureSets(stage) {
  return normalizeFeatureSets(null, stage);
}

export function normalizeHistoryWindows(overrides = {}, legacyHistoryLimit = null) {
  const defaults = L1_FEATURE_SET_CONTRACT.history_windows;
  const out = { ...defaults };
  if (legacyHistoryLimit != null) {
    const legacy = Number(legacyHistoryLimit);
    if (!Number.isInteger(legacy) || legacy < 1 || legacy > 100) {
      throw new Error("legacy history limit must be an integer from 1 to 100");
    }
    for (const key of Object.keys(L1_FEATURE_SET_CONTRACT.history_window_bounds)) out[key] = legacy;
  }
  for (const [key, value] of Object.entries(overrides ?? {})) {
    if (!Object.hasOwn(defaults, key)) throw new Error(`unknown history window: ${key}`);
    if (defaults[key] === "ALL") {
      if (String(value).toUpperCase() !== "ALL") throw new Error(`${key} history window must stay ALL`);
      out[key] = "ALL";
      continue;
    }
    const n = Number(value);
    const [min, max] = L1_FEATURE_SET_CONTRACT.history_window_bounds[key];
    if (!Number.isInteger(n) || n < min || n > max) {
      throw new Error(`${key} history window must be an integer from ${min} to ${max}`);
    }
    out[key] = n;
  }
  return out;
}

export function selectedPrefixes(featureSets) {
  return unique(featureSets.flatMap(name => FEATURE_PREFIXES[name] ?? []));
}

export function selectFeatureFamilies(features, featureSets) {
  const allowedPrefixes = selectedPrefixes(featureSets);
  const out = {};
  for (const [key, value] of Object.entries(features ?? {})) {
    const familyPrefix = ALL_FAMILY_PREFIXES.find(prefix => key.startsWith(prefix));
    if (familyPrefix == null || allowedPrefixes.includes(familyPrefix)) out[key] = value;
  }
  return out;
}

export function phaseBlockedKeys(features, predictionPhase) {
  const phase = normalizePredictionPhase(predictionPhase);
  const policy = L1_FEATURE_SET_CONTRACT.prediction_phases[phase];
  const exact = new Set(policy.blocked_exact_model_keys ?? []);
  const patterns = (policy.blocked_model_key_patterns ?? []).map(raw => new RegExp(raw));
  return Object.keys(features ?? {}).filter(key => exact.has(key) || patterns.some(re => re.test(key)));
}

export function applyPredictionPhase(features, predictionPhase) {
  const blocked = new Set(phaseBlockedKeys(features, predictionPhase));
  return Object.fromEntries(Object.entries(features ?? {}).filter(([key]) => !blocked.has(key)));
}

export function assertPredictionPhaseSafe(featureNames, predictionPhase) {
  const probe = Object.fromEntries((featureNames ?? []).map(name => [name, true]));
  const bad = phaseBlockedKeys(probe, predictionPhase).sort();
  if (bad.length) throw new Error(`prediction phase ${normalizePredictionPhase(predictionPhase)} blocked model columns: ${bad.join(", ")}`);
}
