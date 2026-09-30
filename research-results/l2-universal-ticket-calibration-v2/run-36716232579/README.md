# L2 Universal Ticket Calibration V2

V1 showed severe positive-EV hallucination: ticket probabilities were too generous in the tails. V2 keeps the same market-free L1.7 probability world, then calibrates each bet-type ticket distribution with p' = p^gamma / sum(p^gamma) inside each race. Gamma is selected only by strictly-prior-year winning-ticket negative log likelihood. Odds, ROI, payout and the test year are not used to choose gamma. The betting rule remains exactly p' * odds > 1.0 with no human edge floor, Top-K, ticket cap, odds band, or race skip rule. 2026 remains sealed.
