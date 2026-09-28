# KEIBA-ML Research North Star: Profit First

## Primary objective

The project optimizes **realized betting profitability**, not prediction accuracy for its own sake.

L1 and L1.5 are upstream candidate-generation and candidate-routing layers. Their hit rate, capture rate, ranking quality, calibration, and stability are **intermediate metrics**.

A model or routing strategy is not production-final merely because it improves L1/L1.5 accuracy.

## Layer responsibilities

- **L1**: evaluate horses without odds.
- **L1.5**: combine / route / fuse L1 experts and produce stronger candidate sets without odds.
- **L2**: evaluate race shape, tickets, latest odds, hit probability, uncertainty, edge, fit, confidence, scenarios, and buy/skip decisions.
- **L3**: allocate bankroll under budget, risk, minimum hit/edge, ticket-count, and skip constraints.

## Promotion rule

Research candidates should advance in stages:

1. Improve or diversify horse/candidate capture in L1/L1.5.
2. Remain stable on unknown-year walk-forward validation.
3. Demonstrate that the improvement survives L2 ticket construction.
4. Demonstrate positive and robust EV/ROI after realistic odds and betting constraints.
5. Only then consider production adoption.

A small L1/L1.5 accuracy gain that produces worse ticket economics is a failure.
A candidate-selection change that improves downstream ROI can be valuable even when its raw Top-N metric is not the absolute best.

## Guardrails

- Odds are not used to estimate horse ability in L1/L1.5.
- Odds belong to L2/L3 for edge and betting decisions.
- 2026 remains locked for research/tuning unless explicitly opened under a future protocol.
- Avoid selecting strategies on the same future year used to claim validation.
- Prefer walk-forward and untouched confirmation periods.
- Keep successful intermediate datasets immutable and reuse them rather than recomputing.
- Final comparisons must include profit-relevant metrics such as ROI, EV, drawdown/risk, ticket count, and skip behavior—not hit rate alone.

**North star: make money, not pretty accuracy.**
