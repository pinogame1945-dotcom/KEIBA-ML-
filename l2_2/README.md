# L2.2 Market Distortion Lab

Independent branch built directly from the frozen L2.1 commit 9655c717333636b5cc931f672e12dbc377e396bc. L2.1 is immutable: existing candidate tickets, seven kings, outsider, model rankings and stake rules are not modified.

Scope: four statistical market cohorts (cross-market relative quote discrepancy; favorite placement; multi-horse rank shape; market favorite concentration). QUINELLA, EXACTA, TRIO, TRIFECTA price quotes and actual official 100 JPY payouts, all 2023-2025 race inputs. Candidates use at most nine horses from the market top six plus frozen AI top six. Data source is pinned BACKFILL manifests; no speculative path retries or Kaggle request loops.

Study uses two time-ordered tests: 2023 selects 2024; 2023+2024 select 2025. Conditions must have at least 300 wagers and 12 hits, ROI >= 104 percent, largest three payouts <= 80 percent of total, and ROI without largest three >= 75 percent. Per-year top three sum is used as a conservative bound for combined-years concentration. Only previous-year settlement chooses each test cohort. Physical duplicate bets and all existing frozen L2.1 bets excluded from incremental L2.2. Both all selected fresh tickets and max eight fresh tickets per race are evaluated; combined portfolio totals are calculated.

Implementation reuses previously archived predictions, FINAL prices, frozen L2.1 physical ledger and official payout/parser, with first-pass parallel per year and second-pass parallel held-out years. Download history exactly once onto free standard GitHub Actions CPU runner and locally reuse it. Output is compact aggregate evidence only (strict size <=4 MiB). No artifact or cache upload, GPU, paid compute/storage, or 2026 labels.

CAUTION: FINAL historical odds are not observable before betting cutoff; these retrospective results would not establish executable live edge. Four-arm search is exploratory and still at risk of multiple testing. Keep L2.1 original in-sample ROI clearly distinct from the chronological L2.2 folds. Do not cite 2023 training as out-of-sample performance.
