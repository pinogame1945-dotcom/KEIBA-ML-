# Horse WIN-odds cohort experiment: outcome and decision

Validated GitHub Actions standard free CPU run: https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38770623600

Research branch only: research/l2.2-jackpot-opponent-win-odds-v1-20261016. L2.1 frozen untouched. Three complete years 2023–2025 = 10,368 races. 2026 outcomes sealed.

**Question:** Instead of cutting 6 permutations by each trifecta ticket FINAL quoted odds, retain/drop entire third-horse contender based on its **own FINAL single-win odds**. For each contender, pay exactly 6 x JPY100 trifecta permutations with market ranks 1+2 as anchors. Group by target horse market rank and own win odds. The essential denominator includes *all loser tickets in each band*. Historical FINAL win odds are not guaranteed before purchasing.

## Original 1+2-anchor and 6–9-popularity opponent universe reconciled

Physical 239,574 tickets (39,929 individual 6-ticket opponent sets); hits 940, jackpot-winning tickets 594 (including paid dead heats); stake JPY23,957,400; returned JPY16,936,420; profit -JPY7,020,980; ROI 70.69%. Matches prior verified full actual payout universe exactly.

### Third horse's FINAL win-odds bands, *including all loser opponents*

| WIN odds | Ticket lines | Hit tickets | Jackpot-winning tickets | ROI | Profit JPY |
|---|---:|---:|---:|---:|---:|
| 5–7 | 18 | 3 | 1 | 445.00% | +6,210 |
| 7–10 | 4,770 | 19 | 11 | 44.56% | -264,430 |
| 10–15 | 31,704 | 149 | 99 | 75.06% | -790,670 |
| 15–20 | 43,032 | 203 | 135 | 79.11% | -898,770 |
| 20–30 | 61,290 | 246 | 161 | 72.16% | -1,706,340 |
| 30–40 | 34,830 | 148 | 80 | 77.57% | -781,270 |
| 40–50 | 19,986 | 59 | 36 | 66.09% | -677,700 |
| 50–75 | 22,890 | 76 | 51 | 66.48% | -767,350 |
| 75–100 | 9,150 | 14 | 5 | 29.58% | -644,380 |
| 100–150 | 6,924 | 20 | 11 | 76.17% | -164,980 |
| 150–200 | 2,568 | 2 | 2 | 18.36% | -209,640 |
| 200–300 | 1,914 | 3 | 3 | 62.46% | -71,860 |
| 300–500 | 480 | 0 | 0 | 0% | -48,000 |
| 500–1000 | 18 | 0 | 0 | 0% | -1,800 |

Tiny 5–7 example (18 tickets) is NOT a valid edge. No robust individual well-populated band exceeds 100%.

### Prespecified wider own-WIN-odds range scenarios

| Opponent WIN odds selected | Physical lines | Saving | Jackpot wins / original 594 | ROI | Profit JPY |
|---|---:|---:|---:|---:|---:|
| Any | 239,574 | 0% | 594 | 70.69% | -7,020,980 |
| 10–20 | 74,736 | 68.80% | 234 (39.39%) | 77.39% | -1,689,440 |
| 10–30 | 136,026 | 43.22% | 395 (66.50%) | 75.04% | -3,395,780 |
| 10–40 | 170,856 | 28.68% | 475 (79.97%) | 75.55% | -4,177,050 |
| 15–40 | 139,152 | 41.92% | 376 (63.30%) | 75.66% | -3,386,380 |
| 20–40 | 96,120 | 59.88% | 241 (40.57%) | 74.12% | -2,487,610 |

### 2023–2025 ROI for two illustrative prespecified ranges

| Rule | 2023 ROI | 2024 ROI | 2025 ROI |
|---|---:|---:|---:|
| Full unfiltered 1+2,6–9 | 69.14% | 63.40% | 79.53% |
| Opponent WIN odds 10–20 | 81.38% | 74.91% | 75.65% |
| Opponent WIN odds 10–40 | 69.92% | 71.50% | 85.08% |

No consistent profitable rule.

### Strict historical train -> next-year split (no future-outcome learning)

Among 32 prespecified broad floors/windows, with minimum train 1200 tickets, 15 hits, >=25% saved and >=40% jackpot winners retained, take best ROI in train.

- 2023 train => pick WIN 7–20, 2024 holdout: tickets 25,302, ROI **71.09%**, profit **-JPY731,580**, jackpot wins 74/189; 2024 no-filter 63.40%.
- 2023–24 train => pick WIN 10–20, 2025 holdout: tickets 25,176, ROI **75.65%**, profit **-JPY612,950**, jackpot wins 78/207; 2025 no-filter 79.53%. Improvement FAILED in 2025 relative to unfiltered.

These are only retrospective FINAL market prices; still not bona-fide before-race live odds and repeated lab choices may be overfit.

## Anchor pair baseline comparison — third horse rank 4+ and 6–9

| Anchor favorites | Opponent 4–5 rank ROI | Opponent 6–9 rank ROI | Opponent 10+ rank ROI | All opponent rank >=4 ROI |
|---|---:|---:|---:|---:|
| 1+2 | 73.17% | 70.69% | 62.91% | 67.66% |
| 1+3 | 67.97% | 71.54% | 66.07% | 68.41% |
| 2+3 | 75.52% | 68.37% | 67.01% | 69.09% |

Different pairs, independent physical tickets, no unearned postrace routing or selected winning axes.

### Decision
**NO-GO for standalone own-WIN-odds band gates to make fixed market 1+2/6–9 trifecta profitable.** Despite some capital-saving, dropping outsider horses by own WIN odds also drops prize outcomes. We have now tested both ticket trifecta quote gates and third contender WIN quote gates and neither by itself creates an edge. Do not turn isolated thin cells into policy. Research next must add independently demonstrable *pre-race race/pair reliability signal*, not infinite odds cutpoints, or stop and preserve negative result.

Evidence files:
- research-results/l2-2-jackpot-opponent-win-odds-v1/run-38770623600/summary.json
- research-results/l2-2-jackpot-opponent-win-odds-v1/run-38770623600/opponent_win_odds_bins.csv
- research-results/l2-2-jackpot-opponent-win-odds-v1/run-38770623600/opponent_win_odds_windows.csv
