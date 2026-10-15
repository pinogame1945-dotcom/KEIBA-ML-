# L2.2: TRIFECTA 万馬券のレース構造 — descriptive first pass

**No predictive AI and NO buying recommendations or selection.** Describe all 2023–2025 (10,368) official JRA race-payout observations and distinguish trifecta-winning tickets with official JPY 10,000+ per JPY 100 from ordinary outcomes. 2026 permanently locked.

1. Reuse pinned BACKFILL race-pack manifest, official actual TRIFECTA refunds, and preexisting frozen per-horse market popularity from archived L2 research. Remote name is always taken from manifest; no guessed file retries.
2. Distinguish the *race* containing any winning trifecta refund above threshold from the number of winning tickets. If a dead heat yields multiple winning tickets, choose the highest official refund as the sole representative outcome and audit the multi-paid races. Do not multiply race counts.
3. Outcome-anatomy axes, ALL/JACKPOT/NOT_JACKPOT/100k+ for each of 2023–25 and pooled: actual first/second/third horses' single-win popularity rank, favorite finished within the first three, how many actual winning horses are among top-three favorites, number of unpopular finishers (rank6+ / rank10+), popularity rank-triplet patterns, and payout band.
4. Separate **pre-race identifiable conditions**: declared/active field size, course length, known course surface. Calculate descriptive jackpot rates within each condition with denominators; never substitute *post-race* popularity ranks into a pre-race classifier. Missing surface/length must remain UNKNOWN, not guessed.
5. Summarize annual jackpot frequency and contrast 100x vs 300x vs 1000x etc, plus representative largest historical races. No cherry-picked profitable subset, no threshold training.

This is retrospective race-structure inspection. It can inform the NEXT research question, but cannot by itself imply a strategy or claim that anyone knew the outcome or final payout at betting time. Prices are actual settled payouts, popularity is archived closing win odds.

Efficiency: historical data fetch once using exactly pinned BACKFILL data/manifest.json names, then reuse on one standard free ubuntu-latest CPU runner with 3 independent year processes. No GPUS, paid runner/cache/artifacts, or dataset upload. Persist only compact aggregate/data-provenance results. All new work on isolated branch; frozen L2.1 untouched.
