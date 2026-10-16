# 2026-10-16 富士S 発走前4フェーズ検証

対象：2026-10-17 東京11R 富士ステークス、race_id 202605040611。main/既存BACKFILL/L1/L2.1/L3/MOBILEは無変更。監査専用ブランチでread-onlyの標準無料runnerを使用。

## フェーズ① コース内外・周回数・年齢上限（完了）

[Action run 38850456180](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38850456180)

既存BACKFILLの `src/race-meta.mjs` commit 03950cfe132abeca15ef17dbc7c6885ceb1591b7 を読み、2026富士Sの実ページの原文から変更なしのnormalizerを実行した。

- course_layout = NORMAL（元HTMLに「内」「外」の指定なし。既存normalizerの定義）
- course_laps = null（「周」の明記なし）
- age_max = null（「3歳以上」に上限なし）、age_min = 3、condition_type = THREE_YEAR_PLUS。

別途、指定競走表示の `(指)` と従来parserの `[指]` は意味的に不一致。**未修正**、要差分評価。

## フェーズ② 実際の14頭×134項目の結合（結合検証完了、推論互換未検証）

[Action run 38850791166](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38850791166)

出馬表31項目＋BACKFILL歴史・静的103項目を14頭で結合し、全1,876セルを確認。horse_id 14/14、履歴の既知null 37/37一致、騎手/調教師IDの文字列5桁を保持。モデル候補874列、列名・型不整合0（**raw pedigree IDsなど静的情報を数値列に混ぜない**）。ただし複合特徴量の内部にも41〜65のnull/頭あり、AIへの推論器接続・モデル入力契約の完全互換は未テスト。従来BACKFILLの(指)扱いとは不一致が残る。

## フェーズ③ 天候・芝馬場・馬体重・増減（取得準備完了、実値未公開）

[Action run 38851134947](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38851134947)

netkeiba富士S出馬表の実ページ、JRA東京馬場ページを取得。4件テスト成功。10/16 13:32 JST時点、weather, turf goingが未表示、馬体重14/14頭は未掲載。JRAの馬場ページは対象当日の値を確認できないので、前回開催の情報を混ぜないfail-closed guardを追加。10/17公開後の実値検証は未実施。

参考JRA公式FAQ: https://www.jra.go.jp/faq/pop02/2_3.html

## フェーズ④ JRA発走前オッズ（実取得にブロッカーあり）

[公式出馬表からのオッズ遷移と馬番検証](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38851494541)
[8券種・7テストの検証器](https://github.com/pinogame1945-dotcom/KEIBA-ML-/actions/runs/38851631330)

JRA富士S公式出馬表が取得でき、14頭の馬名・馬番・枠番がnetkeiba監査記録と完全一致。公式カードの「オッズ」リンクは `doAction('/JRADB/accessO.html','pw15oli00/6D')`。そのURLへの直接アクセスは **エラー013：パラメータエラー**。HTTP 200でもオッズ取得成功に数えない。JRA公式の実データ形式・8券種のHTMLパーサーは未完成。

券種別入力データに対する受け入れ検証器のみ整備。WIN, PLACE, BRACKET_QUINELLA, QUINELLA, WIDE, EXACTA, TRIO, TRIFECTA、順序/組合せ数・馬番・範囲オッズ・レースID・取得元・取得時刻・ページ更新時刻を確認する7テストが全成功（**模擬データのみ**）。実オッズ・実際のページ更新時刻は一切取得していない。

JRA公式FAQ: https://www.jra.go.jp/faq/pop02/1_1.html / https://www.jra.go.jp/faq/pop02/1_12.html

## 残課題の厳密な境界

1. コース正規化の `(指)` と `[指]` の不一致（未修正）。
2. 14頭×874のモデル入力を実際の推論器に渡す互換性テスト（未実施）。
3. 10/17当日発表の天候・馬場・馬体重の実値取得テスト（情報未公開）。
4. 10/16 18:30以降のJRAでのオッズ一覧/式別への**正しい遷移**の確立、実8券種・全組み合わせ/更新時刻の取得検証（未実施）。JRA出馬表の単純な `accessO.html` URLはエラー013になるため、直接URLアクセスを成功と誤認しない。

本監査はデータソースや凍結モデルの仕様を変更していない。GitHub Actionsの有料保存（cache/artifact）やGPU、課金runnerは使用していない。
