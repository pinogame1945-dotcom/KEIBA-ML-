# L2 EXACTA Direction Specialist V1

The unordered-pair candidate generator is held fixed. Only A>B versus B>A is learned. DIR_STANDARD is the ordinary market-aware direction classifier. DIR_VALUE uses the same model but weights historical winning-pair examples by exacta payout with log compression and a training-only 95th-percentile cap, so expensive correct directions matter more without letting one jackpot dominate. Both always buy exactly one orientation for every selected pair; there is no confidence gate, no odds filter, no LAW search, and no test-year tuning. 2026 is sealed.
