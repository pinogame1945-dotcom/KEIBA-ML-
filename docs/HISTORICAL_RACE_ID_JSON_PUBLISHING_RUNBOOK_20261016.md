# 過去レースID → 純粋AI・期待値AI・L2.1買い目・L3 V3金額のJSONを保存する

## 使い方（mainに一切変更を加えない）

1. 開発ブランチ `feature/l3-v3-official-github-producer-20261016` の [config/historical_race_ids.txt](../config/historical_race_ids.txt) を開く。
2. GitHub画面の鉛筆アイコンから、希望の2023〜2025年の12桁netkeibaレースIDをカンマ区切りで入力する（1〜24件）。
3. **同じ開発ブランチ** にコミットする。標準無料CPUの `Publish frozen L2.1 original AI and L3 V3 historical race JSON - Free CPU` が `push` で自動起動する。
4. CIが成功すると各レースが**公開データブランチ** `data/historical-l21-v3-v1` の `historical/YYYY/RACE_ID.json` に保存される。後から再取得可能。workflowのログにも `HISTORICAL_RESULT_OPEN_URL` が表示される。

GitHubのワークフロー手動実行（`workflow_dispatch`）が、開発ブランチだけではGitHub UIで表示されない場合がある。その場合でも**設定ファイルを編集・コミットする方法**で実行できる。 `main` にワークフローを移すことはこの段階で行わない。

### 確認済みの実データ結果

| レースID | L2.1から変更していない組合せ | 正式L3 V3紙上金額 | GitHubで開く |
|---|---|---:|---|
| `202507010212` | 三連複 2-4-7 | 200円 | [JSON](https://github.com/pinogame1945-dotcom/KEIBA-ML-/blob/data/historical-l21-v3-v1/historical/2025/202507010212.json) |
| `202506010102` | 三連複 4-12-13・三連単 12→6→4 | 100円・300円＝400円 | [JSON](https://github.com/pinogame1945-dotcom/KEIBA-ML-/blob/data/historical-l21-v3-v1/historical/2025/202506010102.json) |

アプリ・プログラムから直接読む場合は次の公開URLを使う。

```text
https://raw.githubusercontent.com/pinogame1945-dotcom/KEIBA-ML-/data/historical-l21-v3-v1/historical/2025/202507010212.json
```

URL中の `2025` と12桁IDだけを切り替えれば、**保存済みの別のレース**も取得できる。未保存レースは404となるので、勝手に予測が存在すると解釈しない。JSONに `scope=HISTORICAL_RESEARCH_REPLAY_ONLY`, `purchase_ready=false` を明記する。

### 保存データ

`KEIBA_FROZEN_L21_V3_HISTORICAL_DISPLAY_V1` 契約で、元の `ai_raw` 順位、`ai_prob` 順位、`value_ev` 順位、元ANCHOR_FULLの軸馬ID、Correction Risk、元の能力スコア、凍結STANDARD購入券、正式L3 V3 `REVERSE_P3_RACE_MAX300` の各券100〜300円・合計額・元L1.7 `p3` 不明数を保存する。

- 予算は**現在の紙上検証設定**：1レース上限20,000円、同日リプレイ対象合算の原資100,000円。これは実資金やユーザーの常用予算ではない。資金不足の際はV3の全件停止が優先。購入処理は実行しない。
- 元のL1/OOSスコア、元ANCHOR_FULL選択、凍結L2.1物理馬券13,479点を正本として読み込み、**再学習や馬券選別のやり直しをしない**。
- `p3` は元の7王＋Outsider校正L1.7が対象にした馬だけ。215券分のUNKNOWNをゼロ扱い/除外しない。
- 過去研究の**レース後FINALオッズ**と2023〜2025年の後知恵選択の再生。発走前の予想保証はなく、2026未知年の結果は開封しない。
- 現段階では馬番→馬IDの出馬表対照が未接続なので、**ランキング表示の馬名・馬番はまだない**。馬券の選択馬番は保存済みの元データから復元している。スマホのJSON受信処理も未接続。

### 無料コスト・ドリフト保護

- `ubuntu-latest` の標準無料CPUのみ。無料と確認できないArtifact/Cache/追加クラウド保存は使用しない。
- GitHubの既存**公開通常Gitブランチ**にJSONを直接コミット。最大24レース/実行、1ファイル32KiB、1回256KiB、新規保存の合計上限4MiB。上限で**書き込み前停止**。有料ストレージに切り替えない。
- 同じレースの保存済みファイルが同一SHAなら再利用する。**内容が違えば上書き禁止で停止**。新仕様・新資金配分は別契約/別保存キーで管理する。
- 凍結版ソース5ファイルは正式GitHub Contentsメタデータ・Git Blob SHAを確認して一度だけ取得。同一runnerで使い回す。404を別名推測でリトライしない。
- 別ブランチへのコミットが必要なためワークフローは限定ブランチで `contents:write` を使う。公開先も固定であり、`main`、凍結L2.1、MOBILEには変更しない。

### テストとコード

- JSON公開：[scripts/publish_frozen_historical_races.py](../scripts/publish_frozen_historical_races.py)
- 公開時の検証：[tests/test_publish_frozen_historical_races.py](../tests/test_publish_frozen_historical_races.py)
- 実レース研究再生：[research/replay_frozen_l21_to_l3_v3_historical.py](../research/replay_frozen_l21_to_l3_v3_historical.py)
- GitHub Actions：[frozen-l21-original-ai-to-v3-real-archive-proof.yml](../.github/workflows/frozen-l21-original-ai-to-v3-real-archive-proof.yml)
- 初回の公開実証成功：[GitHub Actions run 38831322481](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38831322481)

本フェーズは**過去研究結果をレースIDで取得できる状態**まで。新しいレースの実時点予測や、スマホアプリ画面の実装までは含めない。
