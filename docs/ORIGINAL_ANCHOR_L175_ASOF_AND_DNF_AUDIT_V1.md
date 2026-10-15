# L1.75 ORIGINAL ANCHOR_FULL : genuine coefficients, as-of adapter and DNF survivorship audit

Last checked: 2026-10-15. This is **not a production promotion**.

## Concrete reconstruction evidence

- Original historical model: `MARKET_FIXED_ANCHOR_RESIDUAL_V2_FROM_CACHE`, variant `ANCHOR_FULL`, 19 ability/risk/market-interaction features. Source: `research/run_market_fixed_anchor_residual_from_cache_v2.py`.
- True archived `residual-coefficients.csv` for target year **2025** were fitted only on 2022–2024 races. Lambda was chosen using **prior years only** with chronological 80/20 split. Do not substitute retrospectively trained 2025-risk or 2025 seven-king binaries.
- Coefficient values and training-only median/mean/population-standard deviation are independently reconstructed from the original 16MB `fair-base-bf811fa2eab73db0.csv.gz` cache. `research/replay-original-anchor-coefficients-2025.py` reproduced **all 16 saved P__ANCHOR_FULL values in real 202507010212 with maximum difference 1.1102230246251565e-16**, without reading a single 2025 label for scoring.
- Run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38752875344 . Fresh full original-source parameter JSON can be generated *in runner ephemeral storage* via `--export-parameters` and used for as-of scoring. No paid model registry or stored new artifact.

### Reconstruct exact original prior-only anchor parameter pack

```sh
python3 research/replay-original-anchor-coefficients-2025.py \
  --cache "$RUNNER_TEMP/original-anchor/fair-base.csv.gz" \
  --coefficients "$RUNNER_TEMP/original-anchor/coefficients.csv" \
  --lambda-selection "$RUNNER_TEMP/original-anchor/lambda.csv" \
  --predictions "$RUNNER_TEMP/original-anchor/saved-oos.csv.gz" \
  --race 202507010212 \
  --out "$RUNNER_TEMP/original-anchor/proven.json" \
  --export-parameters "$RUNNER_TEMP/original-anchor/prior-model.json"
```

Files above are **metadata-verified GitHub originals** from frozen branch; the one-shot workflow downloads each exact name and reuses them in a session.

## Strict as-of input to reconstructed original ANCHOR_FULL

`research/score-original-anchor-asof.py` recomputes the original 19 features with *CURRENT observed JRA market* (rather than historical FINAL market) and original pre-bet ability/risk companion values, then uses the exact prior-only `ANCHOR_FULL` 2025 coefficients and normalizer.

Important mandatory inputs:

- Original mobile exporter `KEIBA_MOBILE_L21_PREBET_EXPORT_V1`, **one snapshot** containing all ACTIVE runners and all combinations of WIN/QUINELLA/TRIO/TRIFECTA, a freshly checked nonfinal JRA state and a verified separate bet-cutoff timestamp. No purchased-fraction shortcuts or fabricated odds.
- Companion `KEIBA_L1_ORIGINAL_ABILITY_RISK_ASOF_V1` from the *actual same-generation-original* L1 ability and Correction Risk models. Complete `horse_id/horse_number/distill_ai_rank/distill_score_z/distill_score_gap_to_best/risk`, source as-of timestamp, `feature_source: ORIGINAL_PREBET_ABILITY_AND_RISK`. 7King ranks or L1.7 podium probabilities **cannot** substitute these inputs.
- The ability+Correction Risk input must be observed before JRA prices and the original ANCHOR_FULL probability must be recomputed **after** those prices and **before** the bet cutoff. Current actual clock is used; no manual timestamp override is exposed through the command-line.
- ANCHOR_FULL axis is determined by the original frozen `argmax(P__ANCHOR_FULL * WIN odds)`, with deterministic probability/market-rank/horse-ID tie breaks. Every original 50-gate L2.1 ticket then prices against the same exact market quote and passes to L3 `BASELINE_FLAT_100_ALL_OR_NONE` (no dropping, repricing or ROI optimization).

```sh
python3 research/run-one-race-paper-pipeline.py \
  --race 2026XXXXXXXX \
  --mobile-export local/jra-four-product-exact-snapshot.json \
  --original-anchor-parameters "$RUNNER_TEMP/original-anchor/prior-model.json" \
  --original-ability-risk local/original-prebet-risk-and-ability.json \
  --limits local/L3-flat100-budget.json \
  --out-dir "$RUNNER_TEMP/paper-one-race"
```

**2026 is prediction-only for truly future prestart races.** No 2026 result/historical OOS evaluation or model selection. The optional as-of adapter and its L2/L3 handoff pass *synthetic full-field tests* only. Real current JRA market and original ability/risk are **not yet proven**. There is no betting API.

## Five races missing one original OOS horse: ROOT CAUSE IS HISTORICAL DNF FILTER

Source years 2025-01-06, original OOS vs actual BACKFILL runners:
`202506010203` / 5 シーフライト, `202506010208` / 1 プラチナドリーム, `202506010209` / 2 テンウォークライ, `202506010212` / 1 コトブキヘスペラス, `202507010203` / 8 タランチュラ.

All five have `entry_status:STARTED`, valid JRA FINAL WIN odds, and entries in the authentic original 7King Council archive. **All five have `result_status:DNF` and no numeric final finishing position.** Official 2025-01-06 JRA announcements separately confirm all five as actual race-abandoned horses.

Original `research/run_l1_market_disagreement_audit_v1.py::read_market_year()` discards any row whose outcome field `target.finish_position` is not a finite positive number **before** market correction is trained/scored. Thus these five horses vanish before risk and ANCHOR_FULL OOS. This is **not** a 7King, BACKFILL or odds missing-data error.

This retrospective outcome filter creates **selection-on-outcome/survivorship bias**: at betting time, these five could not have been identified as nonfinishers and must not be removed from the input universe. Historical ROI 109.18% from the original frozen research must NOT be presented as certified leakage-free forward performance.

Audits:
- Default as-known *actual race starter* scope: 19/24 races, 27 physical original tickets exact; five races BLOCKED due originally omitted DNF.
- Optional `--retrospective-dnf-source-scope` is explicitly **not a forward validation**. It reproduces the original model's after-result 15/16-runner subsets for **historical source-parity only**. No 2026 and no betting; output explicitly records count of outcome-conditioned exclusions.
- Officially preserving L2.1 frozen rules and 2026 holdout takes priority over making this affected study superficially profitable.

## Remaining real blockers

1. Original **L1 ability + FULL Correction Risk** trained model for *true live as-of* all starters. The original 2025 OOS precomputed risk scores are not a deployable model binary and cannot be carried forward to future 2026. Require actual trained, archived original and full prebet-phase feature schema, and all-started DNF-inclusive scoring. If unavailable, refuse to promote to live.
2. Production JRA 4-product snapshot **one coherent capture**, official non-final signal and authoritative cutoff clock, verified on an actual upcoming race with future 2026 result withheld. Mobile PR #18 is separate and CI-validated but unmerged.
3. END-TO-END actual 2026 race readiness on **all horses**. Synthetic as-of + L2.1 + L3 tested; never treat this as a real forward ROI test.

Cost: free GitHub standard CPU, ephemeral runner only. No GPU, paid runner, cloud storage or paid artifacts/cache. Exact source metadata filenames only; 404 is fatal. Frozen L2.1 unchanged and no 2026 evaluation.
