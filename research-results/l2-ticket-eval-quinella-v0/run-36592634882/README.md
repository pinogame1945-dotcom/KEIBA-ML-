# L2 Ticket Evaluator — QUINELLA V0

Rebuild from scratch at ticket level. Every priced quinella pair in the L1.7 full field is scored independently. The model predicts hit probability from pre-race race/L1.7 features only. Final market odds are not model inputs; they are applied after prediction as edge = p * odds - 1. Development policy is selected on 2023-2024 only, then frozen for the 2025 holdout. No candidate-horse truncation, no template classifier, no Outsider, and 2026 remains sealed.
