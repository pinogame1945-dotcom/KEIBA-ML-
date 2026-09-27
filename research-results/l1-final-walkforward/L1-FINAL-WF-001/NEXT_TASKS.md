# NEXT TASKS — L1 Final Walk-Forward / Handoff

更新時点: 2026-09-28 JST

## 現在進行中

- Workflow: `L1 Final Walk-Forward League V1`
- Run ID: `36330952531`
- Branch: `feature/l1-feature-arena-trainer-v1-20260927`
- Launch SHA: `447f405af4a537e2421a2a74b75b9a1f792a6b4f`
- Snapshot generation: `bf811fa2eab73db0`
- 2019-2025: 全年 model-ready / Kaggle保存済み
- 2026: LOCKED。まだ学習・調整・検証に使わない。
- Jobs: 9 candidates x 5 validation years = 45
- max-parallel: 9
- 学習窓:
  - 2019-2020 -> 2021
  - 2020-2021 -> 2022
  - 2021-2022 -> 2023
  - 2022-2023 -> 2024
  - 2023-2024 -> 2025

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

1. Run `36330952531` を最後まで確認する。
2. 自動生成される永久台帳
   `research-results/l1-final-walkforward/L1-FINAL-WF-001/attempts/run-36330952531/`
   が正常にcommitされたか確認する。
3. 各候補について5年分の以下を比較する。
   - Top1 / Top3 / Top6
   - 勝ち馬平均順位
   - MRR
   - race-normalized NLL
   - LogLoss / Brier / AUC
   - peak RSS
   - 完走年数
4. 単年トップではなく、5年間の平均・最悪年・ばらつきで安定性を見る。
5. 対策4-1ベンチ4体について:
   - 生還したら性能候補としても比較対象へ昇格。
   - まだメモリ死亡なら「4-1でも標準runner不可」と記録し、同じ条件の再試行を無限に繰り返さない。
6. Walk-forward後、2026へ投入する候補を2-5体に絞る。
7. ユーザーの明示判断後に2026を「L1決勝戦」として使用する。
   - 2026を見た後は2026を完全未知データとして再利用しない。
8. 2026で暫定L1上位を決めたら、L1を完全完成待ちにせずL2研究へ進む。
   - L2は複数L1候補を入力にして、買い方・見送り・オッズ乖離・券種を比較する。
9. L2へ進んでもL1 walk-forward結果と2026結果は永久台帳に保持し、後から差し替え可能にする。

## KODOKU-006の扱い

Run: `36329288628`

確認済み:
- AUTOなし・騎手+血統: success
- 人全部+条件別AUTO: success
- AUTOなし・人全部: success
- 騎手だけ+条件別AUTO: success
- 騎手だけ+特徴組合せAUTO: success
- 騎手+調教師+条件別AUTO: success
- AUTOなし・人全部+血統: success
- AUTOなし・騎手だけ: success
- 精鋭5再戦4: failure
- 騎手だけ+血統 再戦2: failure
- ALL再戦4 / 騎手+馬x騎手+血統 再戦2 は引き継ぎ時点では in_progress だが、研究判断上は死亡扱い済み。

KODOKU-006を待って次へ進む必要はない。現在のL1決勝リーグを正とする。

## 重要な基準値

2025 holdout 3,456Rでの主な結果:

- 旧・純性能王者 `core4_no_pedigree`
  - Top1 28.07%
  - Top3 60.16%
  - Top6 82.18%
  - mean winner rank 3.802
  - 旧peak RAM 約14.95GB

- `no_auto_full`
  - Top1 28.79%
  - Top3 59.46%
  - Top6 81.57%
  - mean 3.829
  - RAM 約7GB

- `no_auto_jockey`
  - Top1 28.10%
  - Top3 59.61%
  - Top6 82.03%
  - mean 3.820
  - RAM 約4.42GB

- `no_auto_full_pedigree`
  - Top1 28.30%
  - Top3 59.38%
  - Top6 82.18%
  - mean 3.795
  - RAM 約12.55GB

- `jockey_trainer_condition`
  - Top1 28.21%
  - Top3 59.87%
  - Top6 81.66%
  - mean 3.795
  - RAM 約11.77GB

## 対策4-1

- Commit: `d709ac00981e566124038feba95e572a4227c10a`
- Message: `perf: stream training JSONL rows into flatten`
- 内容: training JSONLを全件list保持せず逐次読み込みする。
- 目的: 入力JSONの二重保持を減らし、標準runnerのpeak RAMを下げる。
- 現在のWalk-forward Runはこの対策を含む。

## データ正本

Snapshot generation: `bf811fa2eab73db0`

2019-2025:
- total rows: 332,419
- compressed total: 2,431,615,951 bytes
- 全年 model_ready=true
- Kaggle dataset: `pino1945/keiba-ml-snapshot-bf811fa2eab73db0`

## 研究ルール

- L1能力評価にオッズ・人気・払戻を入れない。
- 2026は現在ロック。
- 自動の「総合点」や合成winnerは作らない。
- 単一指標だけで決めず、複数年の安定性を比較する。
- mainへfeature branchを盲目的にmergeしない。mainとの差分監査後に判断する。

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
- その他明示的課金が発生するもの

曖昧なら停止側に倒す。
