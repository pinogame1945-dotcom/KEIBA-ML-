# KODOKU-003 — memory-safe ACTOR decomposition

Purpose:
- Re-test ALL and elite5 after the memory-reduction trainer changes.
- Compare elite5 with/without PEDIGREE.
- Keep the two strongest KODOKU-002 combinations as reference candidates.
- Decompose ACTOR into jockey, trainer, and horse×jockey subfamilies and pairwise combinations.
- Record peak RSS and elapsed time by default.
- Keep 2026 sealed.

Data/model contract:
- Snapshot generation: e75cd10b7837f676
- Train: 2023-01-01 .. 2024-12-31
- Holdout: 2025-01-01 .. 2025-12-31
- Model: WIN_BINARY_LIGHTGBM
- Odds in L1: NO
- Feature selection: none
- Standard GitHub-hosted CPU runners only
- No GitHub Artifact/Cache
- Existing Kaggle snapshot is download-only

Candidates:
1. all_retest
2. elite5_retest
3. core4_no_pedigree
4. strong_opp
5. strong_time
6. actor_jockey
7. actor_trainer
8. actor_horse_jockey
9. actor_jockey_trainer
10. actor_jockey_horse_jockey
11. actor_trainer_horse_jockey
12. actor_full

No composite winner is assigned automatically.
