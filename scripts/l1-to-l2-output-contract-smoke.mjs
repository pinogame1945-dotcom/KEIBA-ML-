#!/usr/bin/env node
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const contract = JSON.parse(await readFile(new URL("../contracts/l1-to-l2-output-contract-v1.json", import.meta.url), "utf8"));
const featureContract = JSON.parse(await readFile(new URL("../contracts/l1-feature-set-contract-v1.json", import.meta.url), "utf8"));
const trainer = await readFile(new URL("../research/train-staged-lightgbm.py", import.meta.url), "utf8");

assert.equal(contract.contract, "L1_TO_L2_OUTPUT_CONTRACT_V1");
assert.equal(contract.principles.ability_uses_odds, false);
assert.equal(contract.principles.pre_race_only, true);
assert.equal(contract.principles.outcomes_joined_separately_for_l2_training, true);
assert.equal(contract.principles.market_data_joined_separately_with_timestamp_policy, true);

for (const field of [
  "raw_win_probability",
  "race_normalized_win_probability",
  "predicted_rank",
  "raw_margin_logit",
  "bias_logit",
  "family_contribution_logit",
  "family_abs_contribution",
  "family_abs_share",
  "race_summary",
  "model_sha256",
  "training_config_sha256",
]) {
  assert.ok(contract.required_fields.includes(field), "missing required field: " + field);
}

for (const forbidden of [
  "actual_is_win",
  "actual_finish_position",
  "final_win_odds",
  "final_popularity",
  "payouts",
]) {
  assert.ok(contract.forbidden_payload_fields.includes(forbidden), "missing forbidden field: " + forbidden);
}

assert.equal(contract.canonical_feature_families_from, "contracts/l1-feature-set-contract-v1.json");
assert.ok(Object.keys(featureContract.feature_sets).includes("BASE"));
assert.ok(Object.keys(featureContract.feature_sets).includes("PEDIGREE"));
assert.ok(Object.keys(featureContract.feature_sets).includes("TIME_PACE"));

for (const token of [
  "--l2-output",
  "L1_TO_L2_OUTPUT_CONTRACT_V1",
  "pred_contrib=True",
  "family_contribution_logit",
  "family_abs_contribution",
  "family_abs_share",
  "normalized_entropy",
]) {
  assert.ok(trainer.includes(token), "trainer missing token: " + token);
}

console.log("L1_TO_L2_OUTPUT_CONTRACT_SMOKE_OK");
