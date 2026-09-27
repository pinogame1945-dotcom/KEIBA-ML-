# NEXT TASKS — L1 Final Walk-Forward / Handoff

更新時点: 2026-09-28 JST

## 旧Run

旧Run:
- Run ID: `36330952531`
- Launch SHA: `447f405af4a537e2421a2a74b75b9a1f792a6b4f`

問題:
- 45試合自体は正しいが、matrix展開順が候補偏重。
- 同一の重い候補が複数年ぶん連続でrunner枠へ入り、重い系が枠を占領する可能性がある。

対応:
- 旧Runはユーザーが手動停止する。
- 旧Run停止後にラウンドロビン修正版を再発火する。
- 旧Runの途中結果は最終比較の正本にしない。

## 修正版Workflow

Workflow:
`L1 Final Walk-Forward League V1`

Round-robin修正commit:
`88532badb03f044290e64ea153c16cff70ca09e4`

Branch:
`feature/l1-feature-arena-trainer-v1-20260927`

Snapshot generation:
`bf811fa2eab73db0`

2026:
LOCKED

### ラウンド制

各年で9候補を1本ずつ走らせる。
次年は前ラウンド9本が全て終了してから開始する。
失敗候補があっても次年へ進めるよう後続roundは `if: always()`。

- Round 2021: 2019-2020 -> 2021
- Round 2022: 2020-2021 -> 2022
- Round 2023: 2021-2022 -> 2023
- Round 2024: 2022-2023 -> 2024
- Round 2025: 2023-2024 -> 2025

各round:
- 9 candidates exactly once
- max-parallel: 9
- fail-fast: false

これにより、同一候補の年違いが同時にrunnerを占領しない。

## 決勝候補 5体

1. `core4_no_pedigree` — 旧・純性能王者
2. `no_auto_full_pedigree` — 血統型
3. `jockey_trainer_condition` — 条件型
4. `no_auto_full` — Top1型
5. `no_auto_jockey` — 軽量型

## 対策4-1メモリベンチ 4体

6. `all_retest_v5` — ALL死亡組
7. `elite5_retest_v5` — 精鋭5死亡組
8. `jockey_pedigree_fullauto_v3` — 騎手+血統+フルAUTO死亡組
9. `jockey_horse_pedigree_fullauto_v3` — 騎手+馬x騎手+血統+フルAUTO死亡組

## 次にやること

1. ユーザーが旧Run `36330952531` を停止する。
2. 停止確認後、修正版workflowを再発火する。
3. 5ラウンド終了まで確認する。
4. 永久台帳が正常commitされたか確認する。
5. 各候補の5年分について以下を比較する。
   - Top1 / Top3 / Top6
   - 勝ち馬平均順位
   - MRR
   - race-normalized NLL
   - LogLoss / Brier / AUC
   - peak RSS
   - 完走年数
6. 単年トップではなく、5年間の平均・最悪年・ばらつきで安定性を見る。
7. 対策4-1ベンチ4体:
   - 生還 → 性能候補として比較対象へ昇格
   - 死亡 → 「4-1でも標準runner不可」と記録
8. Walk-forward後、2026へ投入する候補を2-5体に絞る。
9. ユーザー明示判断後に2026をL1決勝戦として使用する。
10. 2026で暫定L1上位を決めたらL2研究へ進む。

## 対策4-1

- Commit: `d709ac00981e566124038feba95e572a4227c10a`
- Message: `perf: stream training JSONL rows into flatten`
- training JSONLを全件list保持せず逐次読み込みする。
- 現在の修正版Walk-forwardはこの対策を含む。

## データ正本

Snapshot generation:
`bf811fa2eab73db0`

2019-2025:
- total rows: 332,419
- compressed total: 2,431,615,951 bytes
- 全年 model_ready=true
- Kaggle dataset: `pino1945/keiba-ml-snapshot-bf811fa2eab73db0`

## 研究ルール

- L1能力評価にオッズ・人気・払戻を入れない。
- 2026は現在ロック。
- 自動の総合点や合成winnerは作らない。
- 単一指標だけで決めず、複数年の安定性を比較する。
- mainへfeature branchを盲目的にmergeしない。

## COST / RUNNER RULE

事前確認なしで可:
- GitHub Actions無料枠
- 標準CPU runner
- 無料枠内の通常CI / test / build

必ず実行前に停止しユーザー許可:
- 有料runner
- GPU runner / GPU instance
- 課金クラウド計算資源
- 課金Artifact / Cache / Dataset / Model保存
- 無料枠超過の可能性があるstorage
- その他明示的課金

曖昧なら停止側に倒す。
