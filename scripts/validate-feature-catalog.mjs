import { readFile } from "node:fs/promises";

const path = new URL("../contracts/l1-feature-catalog-v1.json", import.meta.url);
const catalog = JSON.parse(await readFile(path, "utf8"));

const required = catalog.required_columns_per_feature ?? [];
const allowedStatus = new Set([
  "AVAILABLE","DERIVABLE","BACKFILL_REQUIRED","MISSING_SOURCE","PROHIBITED_L1","L2_ONLY"
]);
const allowedKind = new Set(["RAW","DERIVED","AUTO_DISCOVERY"]);
const allowedAndroid = new Set(["STATE","TODAY","FIELD","NONE"]);

if (catalog?.principles?.ability_uses_odds !== false) {
  throw new Error("L1 ability_uses_odds must be false");
}
if (catalog?.principles?.leakage_policy !== "STRICT_PRIOR_DATE_ONLY") {
  throw new Error("leakage policy mismatch");
}

const ids = new Set();
for (const row of catalog.features ?? []) {
  for (const key of required) {
    if (!(key in row)) throw new Error(`${row.feature_id ?? "<unknown>"}: missing ${key}`);
  }
  if (ids.has(row.feature_id)) throw new Error(`duplicate feature_id: ${row.feature_id}`);
  ids.add(row.feature_id);
  if (!allowedStatus.has(row.status)) throw new Error(`${row.feature_id}: bad status ${row.status}`);
  if (!allowedKind.has(row.kind)) throw new Error(`${row.feature_id}: bad kind ${row.kind}`);
  if (!allowedAndroid.has(row.android_mode)) throw new Error(`${row.feature_id}: bad android_mode ${row.android_mode}`);
  if (row.status === "L2_ONLY" && row.l1_allowed !== false) {
    throw new Error(`${row.feature_id}: L2_ONLY must be forbidden in L1`);
  }
  if (row.status === "PROHIBITED_L1" && row.l1_allowed !== false) {
    throw new Error(`${row.feature_id}: PROHIBITED_L1 must have l1_allowed=false`);
  }
}

for (const forbidden of [
  "market.final_win_odds",
  "market.final_popularity",
  "market.payout",
  "race.actual_start_time",
  "entry.jockey_id",
  "entry.trainer_id",
  "pedigree.sire_id",
  "pedigree.dam_id",
  "pedigree.siresire_id",
  "pedigree.damsire_id",
]) {
  const row = (catalog.features ?? []).find(x => x.feature_id === forbidden);
  if (!row || row.l1_allowed !== false) throw new Error(`${forbidden}: must be forbidden in L1`);
}

const expectedCounts = {
  total: (catalog.features ?? []).length,
  available: (catalog.features ?? []).filter(x => x.status === "AVAILABLE").length,
  derivable: (catalog.features ?? []).filter(x => x.status === "DERIVABLE").length,
  backfill_required: (catalog.features ?? []).filter(x => x.status === "BACKFILL_REQUIRED").length,
  prohibited_l1: (catalog.features ?? []).filter(x => x.status === "PROHIBITED_L1").length,
  l2_only: (catalog.features ?? []).filter(x => x.status === "L2_ONLY").length,
};
if (JSON.stringify(catalog.counts) !== JSON.stringify(expectedCounts)) {
  throw new Error("catalog counts are stale");
}

console.log("L1_FEATURE_CATALOG_OK");
console.log(JSON.stringify({
  version: catalog.catalog_version,
  features: (catalog.features ?? []).length,
  counts: catalog.counts
}, null, 2));
