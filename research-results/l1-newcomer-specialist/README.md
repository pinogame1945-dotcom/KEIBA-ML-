# L1 NEWCOMER Specialist Research Ledger

新馬戦専用L1研究の永久台帳。

- 本線 `research-results/l1-feature-arena/` とは完全分離する。
- 実験IDは `NEWCOMER-001`, `NEWCOMER-002`, ... を使う。
- 各Actions実行は `<experiment>/attempts/run-<run_id>/` に不変保存する。
- 成功・失敗の両方を残す。
- 大容量モデル/データセットは保存しない。
- L1能力評価にオッズは使わない。
