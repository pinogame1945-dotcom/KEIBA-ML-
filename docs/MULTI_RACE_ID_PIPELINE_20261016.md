# Multiple race IDs -> frozen L2.1/L3: operating notes

## Goal

One request can contain 1..60 netkeiba 12-digit race IDs, in any comma, space,
newline or text-file combination. The ID is an **identifier**, not an excuse
for the assistant to scrape netkeiba. The engine only reads registered Kaggle
snapshots and explicit pre-race input files; it never fetches netkeiba pages.

```bash
python3 research/run-races-by-id.py \
  --race-ids '202506050811,202507010212' \
  --session-dir "${RUNNER_TEMP:-/tmp}/keiba-id-batch-session" \
  --out-dir "${RUNNER_TEMP:-/tmp}/keiba-id-batch-results"
```

On Actions, run `Multi-race ID batch paper pipeline (free CPU)`.
While the new workflow is on a feature branch only, updating
`config/race-ids.txt` on that feature branch is the supported push-trigger
path. `workflow_dispatch` becomes available once GitHub can resolve the
workflow on its default branch; do not merge into frozen L2.1 just to enable
manual dispatch. The target feature branch is
`feature/multi-race-id-pipeline-20261016`.

`KAGGLE_API_TOKEN` must be present to score the seven historical prior-year
trained LightGBM models. Actions reads the existing secret
`KEIBA_ML_SNAPSHOTS`. The actual yearly feature data is the saved FINAL-phase
archive and **not independently certified as available pre-bet**.
Rank-sum comparison is not Council/router and not ANCHOR_FULL.

## Why this is fast for multiple IDs

Within the one free standard CPU runner, the snapshot metadata/manifest is
resolved to a real remote filename. The downloaded annual snapshot is streamed
**once per year**, extracting all requested races. The same models and Kaggle
metadata are reused across races in that same year. There is no guessed
`.jsonl` then `.gz` probing and no second download for a cached source.
Only 429/5xx/transient network issues are retried; 404 stops. Large snapshot
data is released after the year's scoring when multiple years are requested,
so the temporary disk budget remains bounded.

## Paper L2.1 and L3 input contract

For each ID the orchestration *always* reaches the frozen paper L2.1/L3
handoff. If its source data are absent, the result is **BLOCKED**, not
fabricated. Provide separately recorded original as-of files, not result-day
FINAL odds or a stand-in seven-king probability.

Sample optional `batch-inputs.json`:

```json
{
  "contract": "KEIBA_ID_BATCH_INPUTS_V1",
  "races": {
    "202506050811": {
      "mobile_export": "./captures/202506050811-jra-prebet.json",
      "original_anchor_parameters": "./models/original-prior-anchor.json",
      "original_ability_risk": "./captures/202506050811-original-ability-risk.json",
      "l3_limits": "./limits/l3-flat100.json"
    }
  }
}
```

Paths in the manifest are relative to the manifest's directory. Original
`original_features` may replace the pair of original anchor parameters and
ability/risk companions, **but must not be mixed with them**. Existing guards
validate horse identity, official JRA one-time four-bettype quotes, timestamps,
cutoff, all starters, the original frozen 50 rules, and flat-100 all-or-none
L3 limits. Use `--input-manifest batch-inputs.json`, optionally
`--require-l3` to return nonzero when at least one L3 paper portfolio is
blocked. A common `--l3-limits path.json` is also accepted.

Outputs: per-ID `prior-year-seven-kings.json` if scored; per-ID
`l21-l3/pipeline-status.json` and paper ticket/portfolio JSONs when source
guards pass; consolidated `batch-summary.json` with completed/block reason,
ticket counts and budget totals across only READY portfolios. No model
training, no bet placement, no 2026 outcome reads, no odds scraping.

Offline tests (no Kaggle or JRA input) run with:

```bash
python3 scripts/multi-race-id-smoke.py
python3 research/run-races-by-id.py \
  --race-ids "202506050811,202507010212" --skip-kings \
  --out-dir /tmp/keiba-batch-offline
```

These tests prove ID batching/individual blocking, **not** authenticated
live L1->L3 trading readiness. Genuine as-of ability/Correction Risk and
same-time JRA odds remain unverified sources in the current project.
