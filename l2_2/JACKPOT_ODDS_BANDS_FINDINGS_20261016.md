# L2.2 JACKPOT LAB — Final-odds band buying cost vs official refunds

Validated GitHub Actions run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38767922374

- Research branch: research/l2.2-jackpot-odds-bands-v1-20261016
- All 2023–2025 races = 10,368; sealed 2026, frozen L2.1 unchanged.
- Physical hypothetical tickets: market first and second favorite anchoring 3-way permutations × one runner from market ranks 6–9; each of 6 permutations paid for 100 yen. Every available rank/race used, not post-race selected winners.
- Historical market odds are archived FINAL; findings must **not** be claimed as practical/executable before post-close.
- Source: pinned official BACKFILL SHA 0d4af461b77f0060dd9c789dd2da646026fada47, metadata/manifests verified, local reuse. No 404 guessed retry, free standard CPU only.

## Corrected total settlement vs previous representative-payout lower bound

Prior simple payout-only run reported 239,574 potential lines, 935 winning races, 592 jackpot races, 16,891,700 yen in representative settled returns (a lower bound due to dead heats). The exact official individual-ticket payout check found:
- Raw physical tickets 239,574; stake 23,957,400; paid **16,936,420**; **940 winning tickets** (including 5 extra dead-heat tickets), **594 jackpot winning tickets** (versus 592 ordinary jackpot races), profit -7,020,980; ROI 70.69%.
- FINAL quoted lines 239,520; stake **23,952,000**; paid 16,936,420; profit **-7,015,580**; ROI **70.7098%**. Missing FINAL ticket quotes 54 (2024 30, 2025 24), all losers, excluded from every price-filter study.
- Official paid results disagree by >100 yen vs rounded FINAL ticket quote in 10 priced winner tickets (2023=2, 2024=6, 2025=2). Settled returns always use official refund.

## Complete three-year prespecified policy comparison (quote-available tickets only)

| Odds / 100-yen ticket | Lines | Savings vs 239,520 | Jackpot tickets captured (of 594) | ROI | Profit |
|--|--:|--:|--:|--:|--:|
| all quoted | 239,520 | 0% | 594 | 70.71% | -7,015,580 |
| >=100 | 213,122 | 11.02% | 594 | 69.76% | -6,445,280 |
| >=150 | 184,459 | 22.99% | 405 | 67.77% | -5,945,780 |
| >=200 | 156,035 | 34.86% | 297 | 68.04% | -4,987,600 |
| >=300 | 109,214 | 54.40% | 155 | 65.77% | -3,738,050 |
| 100–299.9 | 103,908 | **56.62%** | **439 (73.91%)** | **73.95%** | **-2,707,230** |
| 100–499.9 | 158,050 | 34.01% | 550 (92.59%) | **74.40%** | -4,046,520 |
| 200–999.9 | 142,021 | 40.71% | 293 | 71.34% | -4,070,480 |
| 300–999.9 | 95,200 | 60.25% | 151 | 70.37% | -2,820,930 |

Values of ROI reflect all wins including under-JPY10,000 payouts. Note jackpot-capture definition is official 100-yen payout >=JPY10,000, regardless of whether archived quote was >=100. Exact day/odds filters were predefined before results.

Most importantly, the **historical 30% reduction idea is not sufficient** if removing tickets also loses winning payouts. 100–299.9× removes 56.62% of cost but retains only 45.37% of all real settled returns, yielding ROI 73.95%, not 100%.

## Prespecified chronological choice: year-specific test (no future-year training)

Among 18 fixed floors/ranges with training conditions >=30% cost reduction, >=45% jackpot retention, >=500 sampled lines, >=20 jackpot hits, select **highest historical ROI on training only**.

Train 2023 -> test 2024, selected 100–299.9:
- training ROI 74.24%;
- 2024 test 34,300 tickets, 148 jackpot winning tickets / 189 in quote-available full family, savings 56.93%; ROI **73.94%**, net **-893,930 yen**.

Train 2023–24 -> test 2025, again selected 100–299.9:
- training ROI 74.09%;
- 2025 test 34,106 tickets, 142 jackpot tickets / 207, savings 57.26%; ROI **73.65%**, net **-898,800 yen**.

Pooled 2024–25 forward test under selected rule: stake 6,840,600; returned 5,047,870; profit **-1,792,730**, ROI **73.79%**. Two negative calendar-year heldout tests. Still subject to repeated experiment/multiple-testing and final-odds execution caveat.

## Price-band fundamental pattern

Under 100x there are 26,398 quoted physical lines and 343+? paid winning tickets including legitimate low-paying results, but none is a jackpot winner. Broad 100–500x slices can recover many 100x outcomes, but there are so many losers that the tote-pool margin cannot be overcome. Simply deleting the highest quoted tickets is not an ex-ante winning edge. 10–29.9× appears ROI >100% in a tiny 694-ticket pooled bin, driven by 2023 (153.33%) while 2024=98.13%, 2025=100.91%, with zero jackpot wins; not suitable as a main jackpot-profit claim.

## Decision

**NO-GO** for standalone odds-band filtering as profitable JACKPOT LAB strategy. Yes for reducing capital at an increased average dollar recovery per stake vs baseline; cannot claim profitable or directly bettable system. Do not touch frozen L2.1. Next conceptual stage (not implemented): evaluate structurally distinct pre-race market/AI selection of axes or horse pairs *before* odds-gating rather than post-hoc micro-optimizing band cutpoints. Risks: selection bias, live quote mismatch, 2026 validation remains sealed.

Evidence:
- research-results/l2-2-jackpot-odds-bands-v1/run-38767922374/summary.json
- research-results/l2-2-jackpot-odds-bands-v1/run-38767922374/atomic_odds_bins.csv
- research-results/l2-2-jackpot-odds-bands-v1/run-38767922374/odds_band_policies.csv
