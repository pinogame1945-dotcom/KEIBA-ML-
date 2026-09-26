# BACKFILL READINESS CHECK V1

## 目的

BACKFILLのレース条件拡張が「本当に研究へ使える状態か」を数字で確認する。

目視ではなく、過去データ全体を同じ基準で見る。

## 出すもの

全体と年ごとに以下を出す。

- レース数
- 各項目の埋まり率
- `UNKNOWN` の割合
- 不正値の割合
- 欠損率
- parser契約に合わないwarning
- 着差の正規化がどれだけ埋まったか

## 完成判定の初期値

主な正規化項目について、

- 既知値 98%以上
- 不正値 0%
- 年代による埋まり率の差 10ポイント以内

を初期の目安にする。

これは絶対的な真理ではない。
実データを見て「その項目は仕様上UNKNOWNがあり得る」と分かった場合は、
理由を残した上で閾値を変える。

## UNKNOWNの扱い

`UNKNOWN` は「フィールド自体が無い」とは分ける。

- coverage: 値またはUNKNOWNが入っている割合
- known_coverage: 実際の既知値が入っている割合
- unknown_rate: UNKNOWNの割合
- missing_rate: フィールド自体が空の割合

完成判定では `known_coverage` を使う。

## 年代差

新しい年だけ100%、古い年は20%のような状態を見逃さないため、
年別coverageの最大差も見る。

## 着差

勝ち馬以外のFINISHED結果を母数として、

- margin_raw
- normalized_margin
- margin_seconds
- margin_length_equivalent
- margin_kind
- margin_normalization_version

のcoverageを出す。

ただし着差は現時点ではL1学習へ入れない。
この検査は「材料が揃ったか」を見るだけ。

## 実行例

`npm run audit:backfill-readiness -- --source-root ../KEIBA-BACKFILL --start 2000-01-01 --end 2025-12-31`

結果は初期値では:

`out/backfill-readiness.json`

へ保存される。

## コスト

この処理はNode.jsでgzipを順番に読むだけ。
GPU不要。

GitHub Actionsで実行する場合も標準CPU runnerで足りる設計だが、
このスクリプト追加時点ではActionsは実行しない。
