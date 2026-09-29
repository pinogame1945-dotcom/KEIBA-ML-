# L2 Router Arena V1

Upstream L1/L1.5 is frozen. The 23 Bet Kings templates are unchanged.

P(hit) remains odds-free. Final odds and Market Gap/K2 signals are introduced only after P(hit), at the strategy/Router layer.

Router evaluation is walk-forward: train on 2023 -> test 2024, then train on 2023-2024 -> test 2025. 2025 is development evidence, not an untouched holdout. 2026 remains sealed.

Three routers compete: deterministic SIMPLE_EXPECTED_ROI, action-level STRATEGY_UTILITY, and race-level DIRECT_BET_TYPE. Each race selects at most one strategy or SKIP. Flat stake is 100 yen per selected ticket; L3 bankroll sizing is not part of this run.
