# KEIBA-ML

KEIBA の機械学習専用リポジトリです。

## Responsibility

- **KEIBA-BACKFILL**: 過去データの正本。収集・補修・正規化を担当する。
- **KEIBA-ML**: BACKFILL の正規化データを読み、L1/L2 の学習済みモデルを作る。
- **Android / app**: 学習は行わず、正式モデルと最新状態を使って推論する。

このリポジトリには巨大な BACKFILL データを複製してコミットしません。

## L1 contract

L1 は「馬を見る」能力モデルです。

- target: `target.is_win`
- odds / popularity / payout: **予測特徴量には使用しない**
- leakage policy: `STRICT_PRIOR_DATE_ONLY`
- validation: 時系列 holdout / walk-forward
- experimental outputs: GitHub Actions artifact 等の一時成果物
- promoted models: 検証を通過したものだけを永続化する

## Initial foundation

1. BACKFILL のデータから L1 dataset を生成
2. 学習・holdout 期間を引数化
3. walk-forward fold を作成
4. holdout の全馬予測（OOF）を保存
5. LightGBM model / metadata / exact feature schema を出力
6. Android 向けモデル化の前に PC/GitHub 側で parity の基準を固定

## Safety

GitHub Actions は原則 `workflow_dispatch` の手動起動のみとします。
コード push だけで重い学習を自動実行しません。
