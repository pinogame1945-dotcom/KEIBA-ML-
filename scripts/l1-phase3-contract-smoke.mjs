#!/usr/bin/env node
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const contract = JSON.parse(await readFile(new URL("../contracts/l1-phase3-research-v1.json", import.meta.url), "utf8"));
const runner = await readFile(new URL("../research/run_phase3_selection.py", import.meta.url), "utf8");
const walk = await readFile(new URL("../research/run_walk_forward.py", import.meta.url), "utf8");
const trainer = await readFile(new URL("../research/train-staged-lightgbm.py", import.meta.url), "utf8");
const workflow = await readFile(new URL("../.github/workflows/l1-phase3-research.yml", import.meta.url), "utf8");

assert.equal(contract.contract, "L1_PHASE3_RESEARCH_V1");
assert.equal(contract.locked_year, 2026);
assert.equal(contract.production_promotion_allowed, false);
assert.equal(contract.selection_policy.outer_holdout_must_not_influence_feature_selection, true);
assert.equal(contract.selection_policy.feature_selection_scope, "FOLD_TRAIN_ONLY");
assert.equal(contract.selection_policy.ability_uses_odds, false);
assert.equal(contract.summary_contract.name, "L1_PHASE3_RESEARCH_SUMMARY_V1");
assert.ok(contract.summary_contract.forbidden_payloads.includes("trained model bytes"));
assert.ok(contract.summary_contract.forbidden_payloads.includes("OOF row dumps"));

for (const family of [
  "OPPONENT", "NETWORK", "LAP", "STYLE", "DISTANCE",
  "BACKFILL", "AUTO", "PEDIGREE", "ACTOR", "TIME_PACE",
]) {
  assert.ok(contract.base_plus_one_families.includes(family));
}

assert.ok(runner.includes("L1_PHASE3_RESEARCH_PLAN"));
assert.ok(runner.includes("L1_PHASE3_RESEARCH_DONE"));
assert.ok(runner.includes("LOCKED_YEAR"));
assert.ok(runner.includes("2026"));
assert.ok(runner.includes("base_plus_one"));
assert.ok(runner.includes("windows"));
assert.ok(runner.includes("shrinkage"));
assert.ok(runner.includes("feature_selection"));
assert.ok(runner.includes("--max-execute-candidates"));
assert.ok(runner.includes("research-summary.json"));
assert.ok(runner.includes("ordered_candidate_ids_by_primary_metric"));
assert.ok(runner.includes("persistent_model_or_oof_upload"));

assert.ok(walk.includes("LOCKED_RESEARCH_YEAR = 2026"));
assert.ok(walk.includes("2026 research lock"));
assert.ok(walk.includes("--feature-selection"));
assert.ok(walk.includes("--fs-max-missing-rate"));
assert.ok(walk.includes("--fs-max-correlation"));
assert.ok(walk.includes("--fs-min-inner-gain-fraction"));

assert.ok(trainer.includes("LOCKED_RESEARCH_YEAR = 2026"));
assert.ok(trainer.includes("L1_TRAIN_ONLY_FEATURE_SELECTION_V1"));
assert.ok(trainer.includes("feature_selection_train_v1"));
assert.ok(trainer.includes("dropped_missing"));
assert.ok(trainer.includes("dropped_constant"));
assert.ok(trainer.includes("dropped_correlation"));
assert.ok(trainer.includes("dropped_inner_gain"));
assert.ok(trainer.includes("inner_valid"));

assert.ok(workflow.includes("runs-on: ubuntu-latest"));
assert.ok(workflow.includes("mode:"));
assert.ok(workflow.includes("matrix:"));
assert.ok(workflow.includes("base_plus_one"));
assert.ok(workflow.includes("feature_selection"));
assert.ok(workflow.includes("--source-sha"));
assert.ok(workflow.includes("--execute"));
assert.equal(workflow.includes("actions/upload-artifact"), false);

console.log("L1_PHASE3_CONTRACT_SMOKE_OK");
