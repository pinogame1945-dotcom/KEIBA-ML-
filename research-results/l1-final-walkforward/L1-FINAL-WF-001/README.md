# L1-FINAL-WF-001 — L1決勝リーグ

目的:
- 2025単年での特徴構成探索をいったん終了し、年を跨いでも強いL1構成を選ぶ。
- 代表5体に、過去のメモリ死亡/死亡扱い4体を対策4-1ベンチマークとして混ぜる。
- 2026は完全ロックしたまま使用しない。

## Walk-forward

学習窓は2年固定、翌年1年を未知年として評価する。

- 2019-2020 -> 2021
- 2020-2021 -> 2022
- 2021-2022 -> 2023
- 2022-2023 -> 2024
- 2023-2024 -> 2025

この方式により各foldの学習量を概ね揃え、候補間・年代間の安定性を比較する。
長期の学習窓そのものの比較は別実験とする。

## 9 candidates

### 決勝代表
1. 旧・純性能王者: core4_no_pedigree
2. 血統型: no_auto_full_pedigree
3. 条件型: jockey_trainer_condition
4. Top1型: no_auto_full
5. 軽量型: no_auto_jockey

### 対策4-1 メモリベンチ
6. ALL死亡組: all_retest_v5
7. 精鋭5死亡組: elite5_retest_v5
8. 騎手+血統+フルAUTO死亡組: jockey_pedigree_fullauto_v3
9. 騎手+馬×騎手+血統+フルAUTO死亡組: jockey_horse_pedigree_fullauto_v3

## Metrics

- Top1 / Top3 / Top6 winner capture
- mean winner rank
- MRR
- race normalized NLL
- raw logloss / Brier / ROC-AUC
- peak RSS
- completion status

自動の総合点や総合優勝は作らない。
各年の安定性、性能、メモリ、完走可否を保存し、2026投入候補を後段で決める。

## Cost / storage

- public repo standard GitHub-hosted CPU runner only
- GPUなし
- GitHub Artifact/Cacheなし
- 既存の承認済みKaggle Snapshotはread-only
- 結果は小さいJSON/text台帳のみGitへ保存
