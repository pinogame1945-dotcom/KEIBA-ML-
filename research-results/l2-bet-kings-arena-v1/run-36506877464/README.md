# L2 Bet Kings Arena V1

Upstream is frozen L15_FIXED_V1. This arena does not modify L1/L1.5.

## Leakage guard

P(hit) models explicitly exclude odds, implied probability, popularity, payout,
return, ROI, finish/result labels and blind labels. Final odds enter only after
prediction when edge = P(hit) * final_odds - 1 is calculated.

## Selection discipline

- 2023 and 2024 out-of-sample folds select each bet-type candidate.
- 2025 is a final holdout and is never used to choose template or edge threshold.
- 2026 remains sealed.
- A king candidate must clear ROI >= 100% in both development years and minimum
  ticket/race counts from the contract. Otherwise the bet type is marked
  NO_QUALIFIED_KING even if a research leader is shown.

The next layer, Bet Router, is intentionally NOT built in this run.
