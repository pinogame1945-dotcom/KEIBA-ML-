# KODOKU-005 — 死亡組再戦 + AUTO解剖 + 人要素軽量化

目的:
- KODOKU-004でメモリ上限に達した候補を、最新メモリ対策（LightGBM histogram cache制限）後に再戦する。
- KODOKU-004で有力だった「AUTOなし」を基準に、人要素をさらに軽量化する。
- AUTOを ROLLING / CONDITION / FIELD / PAIR に分解し、必要な部分だけ戻した時の性能とメモリを比較する。
- 2026は封印したまま。

条件:
- Snapshot: e75cd10b7837f676
- 学習: 2023-01-01 .. 2024-12-31
- 検証: 2025-01-01 .. 2025-12-31
- L1でオッズ不使用
- Feature selection: none
- 標準GitHub CPU runnerのみ
- GPUなし
- GitHub Artifact/Cacheなし
- 既存Kaggle Snapshotは読み込みのみ

候補:
1. ALL再戦3
2. 精鋭5再戦3
3. 騎手だけ+血統 再戦
4. 騎手+馬×騎手+血統 再戦
5. AUTOなし・人要素全部
6. AUTOなし・騎手だけ
7. AUTOなし・騎手+調教師
8. AUTOなし・騎手+馬×騎手
9. AUTO過去N走統計だけ戻す
10. AUTO出走馬内相対値だけ戻す
11. AUTO条件別成績だけ戻す
12. AUTO特徴組合せだけ戻す

AUTO slice:
- ROLLING: 過去N走の集計・トレンド・直近差など
- CONDITION: 同競馬場/芝ダ/距離/馬場/クラス等の条件別成績
- FIELD: 今回の出走馬内での順位・平均との差・上位との差など
- PAIR: 特徴同士の差・比率・正規化差

総合優勝は自動決定しない。指標・特徴数・メモリを永久台帳へ保存する。
