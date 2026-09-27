import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const walk = await readFile(new URL("../research/run_walk_forward.py", import.meta.url), "utf8");
const workflow = await readFile(new URL("../.github/workflows/l1-walk-forward.yml", import.meta.url), "utf8");

assert.ok(walk.includes("L1_BACKFILL_READINESS_GATE"));
assert.ok(walk.includes("L1_BACKFILL_READINESS_BLOCKED"));
assert.ok(walk.includes("scripts/audit-backfill-readiness.mjs"));
assert.ok(walk.includes("--min-core-known-coverage"));
assert.ok(walk.includes("--max-invalid-rate"));
assert.ok(walk.includes("--max-year-gap"));
assert.ok(walk.includes("--require-source-integrity"));
assert.ok(walk.includes("BACKFILL_READINESS_V3"));
assert.ok(walk.includes("source integrity not verified"));

const planAt = walk.indexOf("if a.plan_only:");
const gateAt = walk.indexOf("L1_BACKFILL_READINESS_GATE");
assert.ok(planAt >= 0);
assert.ok(gateAt > planAt);

const workflowGate = workflow.indexOf("- name: BACKFILL readiness gate");
const setupPython = workflow.indexOf("- name: Setup Python");
const install = workflow.indexOf("- name: Install ML dependencies");
const runTraining = workflow.indexOf("- name: Run L1 walk-forward");

assert.ok(workflowGate >= 0);
assert.ok(workflowGate < setupPython);
assert.ok(workflowGate < install);
assert.ok(workflowGate < runTraining);
assert.ok(workflow.includes("--min-core-known-coverage 0.98"));
assert.ok(workflow.includes("--max-invalid-rate 0"));
assert.ok(workflow.includes("--max-year-gap 0.10"));
assert.ok(workflow.includes("--require-source-integrity"));

console.log("L1_READINESS_GATE_SMOKE_OK");


assert.equal(workflow.includes("actions/upload-artifact"), false);
assert.equal(workflow.includes("distance_v1"), false);
assert.ok(workflow.includes("prediction_phase:"));
assert.ok(workflow.includes("- EARLY"));
assert.ok(workflow.includes("- FINAL"));
assert.ok(workflow.includes("feature_sets:"));
assert.match(workflow, /default:\s*["']?BASE,OPPONENT,NETWORK,LAP,STYLE["']?/);
assert.ok(workflow.includes("--feature-sets"));
assert.ok(workflow.includes("--history-recent-form"));
assert.ok(workflow.includes("--history-style-last3f"));
assert.ok(workflow.includes("--history-suitability"));
assert.ok(workflow.includes("--history-opponent"));
assert.ok(workflow.includes("--history-auto-rolling"));
assert.ok(workflow.includes("--readiness-report out/backfill-readiness.json"));
assert.ok(workflow.includes("--source-sha \"$BACKFILL_SHA\""));
assert.ok(walk.includes("resolve_feature_sets"));
assert.ok(walk.includes("resolve_history_windows"));
assert.ok(walk.includes("validate_readiness_report"));
assert.ok(walk.includes("BACKFILL SHA mismatch"));
