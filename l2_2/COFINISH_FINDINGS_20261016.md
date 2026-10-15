# L2.2 co-finish dependence — verified research findings (2026-10-16)

Source run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38757930930
Evidence: research-results/l2-2-cofinish-dependence-v1/run-38757930930/

## Decision
**Do not deploy** joint-outcome probability correction as a betting selector. Exact 2024 and 2025 chronological testing was negative. Freeze this research output; do not tune thresholds after seeing the 2025 result as if it remained blind.

Primary pre-registered strategy RELATION_115_CAP8 (prior-year conditional observed/Harville, expected ROI >=115%, max 8 previously unpurchased tickets per race):
- 2024: 23,742 tickets, 170 winning, stake 2,374,200 yen, return 1,945,620 yen, net -428,580 yen, ROI 81.95%.
- 2025: 23,968 tickets, 159 winning, stake 2,396,800 yen, return 1,480,060 yen, net -916,740 yen, ROI 61.75%.
- Pooled: stake 4,771,000 yen, return 3,425,680 yen, net **-1,345,320 yen**, ROI **71.80%**.
- L2.1 frozen overlap excluded at physical ticket identity before selection. L2.1's 2024-25 results were selected with 2023-25 and are *not* independent OOS evidence.

Benchmarks also negative: uncorrected PL expected ROI >=115%, top 8 = 2024 85.05%, 2025 63.65%; corrected threshold >=130%, top 8 = 78.23% and 64.34%; corrected threshold >=115%, top 20 = 95.13% and 73.97%. No cherry-picked deployment of a variant.

## Real finding: important baseline model bias, not yet a market pricing edge
In TRIO candidates with at least two market top-3 horses, all ticket horses in frozen AI top-six, and favorite win-implied probability under 33%:
- 2023: Harville expected 835.749 hits, actual 626 (ratio 0.7490), realized ROI 70.82%.
- 2024: expected 828.089, actual 635 (0.7668), ROI 75.96%.
- 2025: expected 820.012, actual 604 (0.7366), ROI 75.42%.

Therefore the independent win-probability-to-order model overpredicts these joint outcomes consistently. This is *not proof that pari-mutuel TRIO prices are wrong*: actual quoted-pool ROI was roughly the normal takeout range and never profitable in these three samples.

Apparent opposite cell instability: TRIO tickets with <=1 market top-3, at least one ticket horse outside frozen AI top-six, and favorite win-implied chance below 33% had ROI 173.81% (2023, 11 hits), 27.32% (2024, 6 hits), 147.88% (2025, 14 hits). The alternating yearly ROI is insufficient as a stable buying law despite some observed/expected outperformance.

## Confirmed missing layer for an actual pace clash study
This version tests combinations and *rank/market consensus* only. It does NOT identify front-running horses or actual race pace. Existing ML feature catalog lists pre-race, strict-prior-date historical corner features including style.previous_first_ratio, style.previous_last_ratio and style.previous_position_gain. See contracts/l1-feature-catalog-v1.json, source results.corner_raw (HISTORY STRICT_PRIOR_DATE_ONLY). Stage builder research/build-staged-dataset.mjs already derives past corner features. It remains necessary to verify accessible snapshot coverage and ensure no current-race result leakage before using these in a genuinely new pace-conflict study.

Potential next **pre-registered** family: expected pace clash of multiple front runners vs diversified styles, as a group interaction conditional on surface, distance and draw, with separate price comparison. Only pre-race historical corner features allowed. Do not silently repurpose post-race actual running position or current-race pace.
