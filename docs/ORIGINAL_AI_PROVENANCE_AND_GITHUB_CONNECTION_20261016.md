# 原版AI推論の復旧・未達境界（2026-10-16）

## このフェーズでGitHub側に実装したもの

- 凍結L2.1の既存 `ANCHOR_FULL` 19特徴量の正本ジオメトリを、復旧コミットから原文で移植：`research/anchor-original-feature-geometry.py`。
- 元2025対象foldの学習済み係数・2022〜2024年だけから作った標準化を、保存済み正本から**再推定ではなく忠実に復元**：`research/replay-original-anchor-coefficients-2025.py`。
- GitHub Contents metadataで正本4ファイルの正式パス・blob SHAを照合し、runner内sessionに一度取得。404に別名を試さない。標準無料CPUのみ：`scripts/build_original_anchor_2025_params_github.sh`。
- 上記元の係数を用いて、GitHub収集のPRE_RACE JRA単勝価格と、元のAI能力・Riskの全頭値から元の `P__ANCHOR_FULL` を計算：`research/score_github_original_anchor_asof.py`。
- 既存 `research/run_github_race_ids_v1.py` に、`original_l21_features` 外部完成JSONの代わりに `original_anchor_parameters`＋`original_ability_risk` を受け取る経路を追加。**同時入力禁止**。元のL2.1 STANDARD 50条件とL3 V3正式 `REVERSE_P3_RACE_MAX300` は変更していない。
- 追加の全馬ID厳密照合：`original_l175_horse_p3` の馬番だけでなく、公式出馬表の `canonicalHorseId` との全頭一致を要求。

## 実行方法：元2025用係数の再構築（GitHub無料CPU内の一時領域）

```bash
bash scripts/build_original_anchor_2025_params_github.sh "$RUNNER_TEMP/original-anchor"
# 生成するのは既存研究の係数と標準化。新規学習・新規最適化は行わない。
# 一時ファイル: $RUNNER_TEMP/original-anchor/original-prior-anchor-asof-params.json
```

生成モデルは **2025を未知年とした元のモデル** で、`trained_only_on_years=[2022,2023,2024]`。学習年に2025も2026も混ぜない。元の `research-cache/market-fixed-anchor-residual-v2/fair-base-bf811fa2eab73db0.csv.gz`、保存済みOOS予測、`residual-coefficients.csv`、`lambda-selection.csv` からのみ算出。正本の2015年以降等の全モデル適用を証明するものではない。

### レースID実行マニフェストの代替入力

各レースで `original_l21_features` の代わりに下記2つを指定する。

```json
{
  "original_anchor_parameters": "sources/original-prior-anchor-asof-params.json",
  "original_ability_risk": "2026XXXXXXXX/original-ability-risk-asof.json"
}
```

他に `race_card`, `jra_prebet_quotes`, `original_l175_horse_p3` と予算マップは必須。ここで提示する `2026XXXXXXXX` はパスの説明用プレースホルダー。日時が過去のものである場合、**CLIの現時刻検査に失敗**する。2025年の確定オッズを現在の市場データと偽装しない。

## 検証の区別

1. **元モデル係数の実データ再現成功**：2025-01-06 `202507010212` 16頭、19特徴量と元のP値を保存済み2025研究と照合。ただし入力オッズはFINAL。ここでの成功は元研究との数値一致のみ。Actions `38827799165`。
2. **配線の合成テスト**：真正な正本特徴量関数と検証用の合成係数・合成原版companionを使用。単数/複数レースで元の `ANCHOR_FULL → 凍結L2.1 → L3 V3` まで流し、馬券集合変更なし・リスク欠落・結果列禁止・年境界を確認。テスト専用の時刻注入は関数内部のテストだけで許される。実運用CLIでは時刻偽装できない。
3. **未達：実際の発走前収集・元の馬別AI能力/Risk/p3に対する再推論成功**。GitHubに保存済みの7王OOS・L1.7 p3・2025 FINAL市場予測だけで補完しない。これらは別モデル・別時点の値。

## 発見した本当の依存ギャップ

- 元の `distill_ai_rank`, `distill_score_z`, `distill_score_gap_to_best` を全出走馬に対して再推論する**配備済み本体**：現行ブランチと旧ハンドオフのGitソースツリーでは確認できない。学習コードと過去研究予測はある。7王7体のKaggle学習済み本体の存在は**このdistillモデルの代替を正当化しない**。
- `Correction Risk` の元の FULL_PRERACE ヘッド：元研究 `run_l1_market_correction_deepdive_v1.py` ではLightGBMをfoldごとに学習して予測値を出力し、保存済みモデル本体にはしていない。既存foldで使った全時点適格の特徴量・対象年以前の教師値・モデル選択仕様を特定して忠実復元する必要がある。
- 原版 `L1.75` の全出走馬 `p3`：L3 V3掛け金優先順の `min_p3` に必要。保存済み別系統L1.7の上位10頭p3やCouncil票を本物のL1.75全頭p3として無断昇格させない。
- 過去2025の元研究には、事後DNF馬を欠落させた選択バイアスのある出走母集団が存在する。全出走馬の発走前推論ではDNFを事前に落としてはならない。
- GitHub-side本物の当日JRA 4券種の観測時刻/締切/FINAL判定は開催日まで未検証。スマホ側を収集器として利用しない。

## 今回触っていないもの

`main`、凍結 `freeze/l2.1-standard50-to-l3-20261015`、MOBILE本線、元研究のL3 V3／26資金方式、2026年レース結果・払戻。GitHub Actions無料標準CPUの一時領域で照合し、追加の有料保存・GPUを使っていない。実馬券の購入機能もない。
