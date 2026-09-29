# L1.7 Seven-King Full-Field Diagnostic V1

L1.7 is an independent diagnostic layer between the seven L1 experts and downstream analysis.

It is intentionally separate from L1.5.

## Flow

Seven L1 experts
→ full horse-level scores for every runner
→ result-order-independent re-ranking inside each expert
→ full-field seven-king aggregation
→ L17_SEVEN_KING_FULLFIELD_OUTPUT_V1

## L1.7 does

For every horse:
- each king's rank and normalized win probability
- mean rank and rank dispersion
- best/worst king rank
- Top1 vote count
- Top3/Top6 support count
- mean probability and probability dispersion
- deterministic full-field consensus rank

## L1.7 does not

- no Consensus World Gate
- no Outsider
- no candidate filtering
- no odds/popularity/payout
- no ROI/EV
- no result labels
- no mutation of L15_FIXED_V1

L1.5 remains the candidate-expansion interface. L1.7 is the independent full-field diagnostic interface.
