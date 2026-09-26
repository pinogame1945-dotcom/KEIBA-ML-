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
- margin_type
- margin_lengths

を検査する。

`margin_type` は98%以上を要求する。
`margin_type=LENGTHS` の行だけ `margin_lengths` を必須にする。
ハナ・アタマ・クビなどのカテゴリ着差で `margin_lengths=null` は正常。

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


## 学習前の門番

walk-forwardの実行モードでは、この検査を**データ作成・LightGBM学習より前**に必ず通す。

失敗時:

- `out/walk-forward/backfill-readiness.json` に理由を保存
- 学習データを作らない
- LightGBMを起動しない
- モデルを保存しない
- OOFを作らない

plan-onlyは日付計画を表示するだけなので、この門番を走らせない。

GitHub Actions側でも同じ検査をPython環境やML依存の準備より先に行う。
これによりBACKFILL未完成時の無駄な計算を早い段階で止める。

門番を無視するオプションはV1では用意しない。


## 追加の意味チェック

単に値が埋まっているだけでは合格にしない。

- `course_laps` は全レース必須ではない。元のコース表記に「N周」がある場合だけ、その数値との一致を必須にする。
- `新馬` → `NEWCOMER`
- `未勝利` → `MAIDEN`
- `500万下` → `ONE_WIN`
- `900万下 / 1000万下` → `TWO_WIN`
- `1600万下` → `THREE_WIN`
- `オープン` → `OPEN`

明確な元表記と正規化結果が食い違う場合は不正値として止める。

## 二重読み込み防止

GitHub Actionsでは最初に一度だけBACKFILL全体を検査し、結果JSONにBACKFILLのcommit SHAを記録する。
walk-forward本体は、そのSHA・対象期間・判定基準が完全一致した検査結果だけ再利用できる。

一致しなければ再検査または停止となり、古い検査結果を使い回すことはできない。
