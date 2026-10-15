# L2.2 co-finish dependence: price vs joint outcome

This independent statistical line inherits L2.1 methodology without changing frozen L2.1 or previous L2.2. Base: frozen L2.1 commit 9655c717333636b5cc931f672e12dbc377e396bc. This branch forks from L2.2 result commit 39fb264634089d23cf709fc404cd3a7d8b71906e.

## Hypothesis

Markets may assess each runner accurately in isolation but misprice the probability of **specific pairs/triples placing together**. Competing favorite and longshot effects are not assumed profitable. Harville/Plackett-Luce top-order probabilities from *normalized existing FINAL win odds* are compared with actual exact outcomes for (1) QUINELLA (unordered first two), (2) EXACTA (ordered first two), (3) TRIO (unordered first three), and (4) TRIFECTA (ordered first three).

No actual running styles exist in the confirmed input schema. Do **not** pretend that favorite/AI rank disagreement measures race pace or running-style clashes. Feature cohorts from **pre-result** data: top-3 popularity composition, whether all ticket horses are in frozen AI top-six, and win-market favorite concentration. Each year's cohort report includes observed joint success vs Harville expectation and settled ROI.

## Real trading test of the hypothesis (still closing-odds retrospective)

Prespecified calibration: for each cohort, use ONLY historical observed-to-expected ratio with shrinkage to kind-level parent. Parent has 40 expected-hit pseudocounts, child has 25 expected-hit pseudocounts; ratio clipped to 0.60..1.40. Train 2023 and test 2024; train 2023+2024 and test 2025. The 2025 training scan is **diagnostic only** and cannot inform earlier fold selection.

For each 2024/2025 race, candidate horses are union of first six market + first six frozen AI, truncated at nine, exactly as in initial L2.2. Candidate universe is all physical QUINELLA, EXACTA, TRIO, TRIFECTA tickets from that set. Actual final quoted ticket price is required. Single explicit max accepted quoted price: 2,000. Frozen L2.1 tickets are excluded *before* result scoring, at physical (year,race,kind,ordered/canonical numbers) key. No ticket re-bought within a race.

Prespecified policies:
- **PRIMARY:** RELATION_115_CAP8 (prior-only conditional joint probability times final quote >=1.15; top eight per race).
- BASE_PL_115_CAP8: no historical correction.
- RELATION_130_CAP8: stronger expected edge.
- RELATION_115_CAP20: less restrictive portfolio.

The above are a predefined comparison, **not** a post-hoc search for a profitable threshold. Report all policies and both years regardless of results, raw ROI and biggest-three winners removed; incremental stake/profit vs L2.1 and combined ROI.

## Implementation cost/quality
2023/24/25 diagnostics computed as three independent standard CPU workers. 2024/25 test computed as two standard CPU workers. Download pinned manifest-confirmed official data only once within the runner, reuse locally. Payouts must reconcile to official pack; count quoted candidates per year matches previous L2.2 lab [651539, 665028, 665944]. Missing final quote automatically EXCLUDES the ticket irrespective of success/failure and is audited. Do not guess another remote name on 404. 429/5xx are transient only; Git handles retry if needed. Upload no GitHub paid cache/artifacts and store only compact results in branch. 2026 unopened.

Important: historic **FINAL** odds are not confirmed executable before betting closes. A positive result does not establish practical betting profit; chronological holdout is also not fresh, unexamined race year (repeated research bias remains).
