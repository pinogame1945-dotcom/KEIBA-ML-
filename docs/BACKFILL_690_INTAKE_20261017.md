# BACKFILL 690 -> ML input receiver (研究用・入力整合性確認のみ)

このブランチでは特徴量生成器・学習モデル・凍結済み L2.1 / L3 V3 を変更しない。

BACKFILL 側の `ml_features/research/build_690_any_race_v1.py` または
`build_live_690_batch_36_v1.py` が生成する各レース `<race_id>.json` を受け入れる。

入力が存在するときだけ以下を実行する（RUNNER_TEMP など一時領域を推奨）。

```bash
python research/receive_backfill_690_v1.py \
  --input-dir "$RUNNER_TEMP/backfill-690" \
  --output-dir "$RUNNER_TEMP/ml-690-input" \
  --expected-races 36 --expected-horses 471
```

構造確認：レースID・出走馬キー・649+19+22=690 列・固定 CORE4 SHA・
結果/払戻/確定オッズ混入フラグ・欠損値/数値の形式・重複を確認。
**元研究との全数値一致は検査しない（利用者の指示）。**

この入口は元の `readiness: ASOF_UNVERIFIED_DO_NOT_BET` を
`STRUCTURE_ACCEPTED_RESEARCH_ONLY_NO_BETS` として記録し、
発走前の時点保証や、モデルの実行成功を主張しない。

## 2026-10-17 時点の事実
- BACKFILL側最新Run `38923244743` は完走せず。
- 10月10日24レース分の690特徴量はRun一時領域で作成されたが、
  Artifactも公開Gitファイルもない。
- 10月17日36レース分の690特徴量はRun中の最終工程がスキップされた。
- そのため**現時点で実データの受信を完了したわけではない**。
- 有料Artifact/Cache/保存や学習モデルの改変はしない。
