# 元L1の新規レース推論 — 保存済み予想に依存しない入口（2026-10-16）

## 目的

従来の「過去に保存済みのAI結果・凍結馬券をレースIDで再生する」のとは別に、**まだ予測していない馬群に対して凍結L2.1の元AIを新規に学習・推論し、L2.1→正式L3 V3に渡すこと**を目指す。GitHub側がデータ収集・モデル学習/推論・出力を担う。アプリは完成JSONの表示だけ。

このブランチ：`feature/original-ai-cold-inference-20261016`。既存の `main`、凍結L2.1 `9655c717333636b5cc931f672e12dbc377e396bc`、正式L3 V3のブランチ、MOBILEを変更しない。

## 確認した正本

`freeze/l2.1-standard50-to-l3-20261015` の元コードから移植ではなく**直接再利用**する。

- `research/run_l1_objective_rebuild_v1.py` の `prepare()`, `model_common()`：特徴量の数値化・カテゴリ語彙・LightGBMハイパーパラメータを元研究のまま使用。
- `research/run_l1_residual_regime_full_pairwise_v1.py` の `fit_direct_rank()`：純粋AIの正本LambdaRank。対象年Yの前2年で学習、`LGBMRanker(objective=lambdarank, metric=ndcg, label_gain=[0,1,3,7,15,31,63])`、seed `5100000+Y*100`。元の関数は未知年にも着順列を要求するので、その関数そのものの呼び出しはしない。**同じ演算・設定を、未知の着順列を使わない推論関数に移した**。これはAPI適合のための移設で、別モデル化ではない。
- `research/run_l1_market_correction_distill_v1.py` の `add_ai_context()`：能力順位・馬群内z値・先頭との差を元の同点規則で計算。
- `research/run_l1_market_correction_deepdive_v1.py` の `fit_risk_full()`：Correction Riskの `FULL_PRERACE` 分類器を**対象年より前の過去年OOS教師行**で学習。seed `8200000+Y*100`。FINAL人気順位は過去年の教師ラベルにのみ許容、対象レースでは**禁止**。
- `research/run_l1_ceiling_audit_v1.py` の `FULL_FORBIDDEN`：過去研究と同じ危険列リストで、未知レースへの結果/確定オッズ/賞金等の入力を拒否。

`models/promoted/README.md` は、**元モデルが正式promoted配備されておらず**、研究foldの学習済みモデルは一時成果物扱いと明示している。新規レース推論は「保存済みモデルがすでにある」とは仮定しない。

## 今回追加した実行入口

`research/run_original_l1_cold_inference_v1.py` は次の **3つの元データ**を受け取る。

1. `prior-ability.json`：対象年Yの**Y-2、Y-1年のみ**の元L1学習用特徴量＋過去レース着順ラベル。
2. `prior-oos-risk-teacher.json`：2021〜Y-1年の本来の**未知年OOSモデル出力から作ったCorrection Risk教師値**と元のFULL_PRERACE特徴量。これは元の研究仕様に準じて再生成する必要がある。未確認の代理信号を混ぜない。
3. `unscored-target.json`：新しいレースIDの**全出走馬の発走前特徴量のみ**。着順、確定オッズ、払戻、過去に保存したその対象レースのAI順位、Correction Risk、ANCHOR_FULL確率は1列でも存在すれば停止する。全頭IDの重複・年ズレ・能力特徴量のschema driftは停止。

いずれも `{"rows":[{"_race_id":"...","_horse_id":"...","_race_date":"...","feature":...},...]}` 形式のJSON（.gzも可）である。実運用では元の年度supersetスナップショットを `prediction_phase=PRE_RACE` で投影し、同日bias、残差regime、state処理を**元の研究コードと同じ方法**で追加しなければならない。既存OOS予測ファイルを「新規ターゲットの入力」として渡すことは禁止する。

### 単独レース推論コマンド

```bash
export L1_THREADS=2
python3 research/run_original_l1_cold_inference_v1.py \
  --target-year 2025 \
  --prior-ability out/prior-ability-2023-2024.json \
  --prior-oos-risk-teacher out/original-oos-risk-teacher-2021-2024.json \
  --unscored-target out/2025-pre-race-full-field.json \
  --output out/cold-original-l1.json
```

実行結果 `KEIBA_ORIGINAL_L1_COLD_INFERENCE_V1` には全頭の `ai_raw_rank`、`distill_score_z`、`distill_score_gap_to_best`、`correction_risk` を出す。**元の能力ランキングをオッズで補正しない**。市場のオッズから勝率 `P__ANCHOR_FULL` を計算するのは次の元ANCHOR_FULLモデルの役割であり、この処理では行わない。

## 成功した検証と、未完了を区別

- `tests/test_original_l1_cold_inference_v1.py`：結果ラベルを持たない**完全に未知の合成10頭**で、元設定のLambdaRankとFULL_PRERACE Riskを学習・推論し全馬へ順位/リスクを付与するCPU smoke。過去年以外の教師混入、結果列/FINALオッズ、既存モデル順位、予測を持たない全頭・重複・schema driftは拒否。
- CIは `ubuntu-latest` の無料標準CPU、元の `ml/requirements.txt` LightGBM等バージョン固定。Artifact/Cache/有料保存/GPU使用なし。
- **未達：真の実レース発走前特徴量の全頭入力、過去年完全OOS Risk教師の再構築、元2025foldとの学習一致、元ANCHOR_FULL/当時オッズ/7王＋Outsider L1.7 p3の接続、凍結L2.1→正式V3までの実レース一気通貫。** 本テスト成功をもってレースIDだけで予想できるとは言わない。
- 2026年のレース結果・払戻を使った学習・ROI検証は禁止。2026年実レースを事前予測することと、結果を研究評価に使用することは区別する。

## 次の開発手順（順番を入れ替えない）

1. 凍結研究と同じKaggle snapshot manifest+metadataを1回解決、必要な年度の実在名を1回取得して同じ標準CPU runner内で再利用。404はメタデータ不一致として停止、429/5xxのみ再試行。課金不明ストレージには保存しない。
2. **一度も予測結果を読み込まない検証用レース**を指定し、当該レースの全出走馬が欠けない `PRE_RACE` 特徴量を元の `build_state_features`、`build_same_day_bias`、`build_regime_features` の過去日基準で生成。過去DNF等の着順なし馬を勝手に除外しない。
3. 前2年で元純粋AIを学習し、その対象レースへの能力順位を新規生成。既存OOSファイルは**後から一致検証用だけ**に使う。
4. 元の過去年未知年OOS教師を元コードで復元してFULL_PRERACE Correction Riskを新規学習。特徴量と全頭集合・モデルハイパーパラメータの一致を検査。
5. GitHubが取得する発走前オッズと元2025fold ANCHOR_FULL係数で期待値AIを算出。凍結L2.1 STANDARDの50条件をそのまま適用。保存済み7王＋Outsider L1.7 p3が適正な時点に得られるかを確認し、正式L3 V3の `REVERSE_P3_RACE_MAX300` を実行して完成JSONをGitHubに書き出す。

この文書は方針・テスト範囲の記録であり、「未予測の実レースまで現在完成」とは認定しない。
