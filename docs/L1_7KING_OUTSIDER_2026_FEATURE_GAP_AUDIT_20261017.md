# 元の7王・Outsider 復旧調査（2026-10-17、実物検証済み）

## 検証結果
- GitHub Actions #38998425018: 成功。凍結7王の候補構成とKaggle 2025年版モデル本体・schema.json・順位出力の存在を確認（**7/7**）。
- GitHub Actions #38998680837: 成功。Kaggleに保存された **実際の2025年版7王 schema.json** を取得し、BACKFILL 2026-10-10 `202605040301.json` の実際の690特徴量の**列名**と照合。
- 検証対象BACKFILL: `handoff/690-feature-data-v1`。調査時点のHEAD `db54e887bfa09e8dd473e85d79c96e1973a7b77a`。元の納品commit `113d6ab73bf840c95a761cf1d8df2b4a0281adf5` 以降の追加コミットはオッズ配送関連で、比較API上 `ml_features/delivery/690` は変更なし。
- 照合は列名に限定。**数値同一性、学習年適格性、2026発走前時点の保証、予測精度は調べていない。**

## 2025年版モデルの必要列と、BACKFILL納品690特徴量の列名一致数

| alias | 元の王 | 必須列 | 690内に存在 | 欠落 | 欠落した列の種別 |
| --- | --- | ---: | ---: | ---: | --- |
| core4 | 純性能型 | 649 | 649 | 0 | — |
| pedlegacy | 旧血統型 | 470 | 266 | 204 | ped_ 204 |
| condition | 条件型 | 423 | 423 | 0 | — |
| full | FULL型 | 266 | 266 | 0 | — |
| light | 軽量騎手型 | 164 | 164 | 0 | — |
| pedv1 | 新血統型 | 504 | 266 | 238 | ped_ 238 |
| condrc | 条件・レースクラス血統型 | 457 | 423 | 34 | ped_ 34 |

**4系統は列名が揃うが、それだけで2026の推論モデルとして利用可能とは認めない。** 2025年版のモデルは過去の2年学習foldで作成されたもので、2026年用の正本更新・学習・適格性は別問題。

2026納品690列のfeature family: OPPONENT 26、AUTO 383、ACTOR 179、TIME_PACE 23、BASE/その他79。PEDIGREE、NETWORK、LAP、STYLE、DISTANCE、BACKFILL は0。合計690。純粋AIの690列には19+22の追加特徴量があり、保存された7王の学習特徴量セットとは同一ではない。

## 確認した確定設定
- 7王の正本設定: `research/l1-raceclass-absorption-walkforward-v1.json`。snapshot生成ID `eb13d4096519167b`、2年rolling、2026年評価封印。
- 各王の保存済み2025年版: Kaggle `pino1945/keiba-l15-7k-<alias>-y2025-v1` 内 `model.txt`, `schema.json`, `score.jsonl(.gz)`。
- 保存モデル本体は存在するが、元2025年foldの重みを無断で2026年本番モデルに昇格させない。
- Outsider系の設定: `research/l1-outsider-nonhorse-v1.json`（6候補）、`research/l1-outsider-transition-v1.json`（7候補）、`research/l1-outsider-arena-v1.json`（5候補）。正本Council保存票の13候補は前2系列の6+7。5つのarena研究候補を追加採用したと解釈しない。
- 研究の過去順位・投票アーカイブは2026年の新規予測値ではない。

## 復旧に必要な残作業（未実行）
1. 元の方式の `ped_` 特徴量をBACKFILL生成経路へ接続し、2026年の列名・定義・時点適格性を確認する。0やNaNによる仮埋めは禁止。
2. 元の7王それぞれについて、正本の2年rolling fold（2026を未知年として2024/25で学習）を再構築し、真正な7つの出走馬順位を生成する。2025年版の保存モデルは検証用・復旧用正本資産であり、2026用の学習済みモデルとは区別。
3. 凍結した13 Outsiderの独立した特徴量・候補投票を実行し、Council集計と照合する。7王の不足出力を純粋AIで代用しない。
4. 2026年の着順・払戻・確定オッズで学習・評価しない。結果は当面研究・動作確認用。L1.75、L2.1、L3は変更しない。

## 調査の変更範囲
新規 `audit/l1-7king-outsider-2026-asset-preflight-20261017` ブランチのみ。
元学習器・元7王・Council・Outsider・main・凍結モデルのロジック・データは無変更。標準無料CPUで読み取りのみ。モデル/大規模データのダウンロード・有料リソース利用・課金保存は実施していない。

## BACKFILL側で復旧に再利用できる実装（追加確認）
- BACKFILL `ml_features/src/pedigree-feature-builder.mjs` に、父・母父の履歴を持つ特徴量生成器 `createPedigreeFeatureState` が既存実装として存在する。
- BACKFILL `ml_features/research/build-yearly-superset-snapshots.mjs` は `--feature-sets` に `PEDIGREE` を含めると既存生成器を使う構成。馬プロフィールの血統情報を参照する。
- したがって `ped_` の不足列を**新規ロジックで作り直す前に、既存のBACKFILL生成経路を再利用する**のが正しい。ただし2026年の全出走馬に関する血統データの有無・34/204/238列の実際の生成可否・発走前の適格性は**未実行、未検証**。
- 元の7王が使っていた snapshot生成ID `eb13d4096519167b` とBACKFILL受け渡し690特徴量の生成履歴の相違に注意。2025版schemaの列名一致は、年度横断の数値・カテゴリ整合性を保証しない。
