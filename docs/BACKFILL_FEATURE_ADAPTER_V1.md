# BACKFILL FEATURE ADAPTER V1

## 目的

KEIBA-BACKFILLで追加中のレース条件を、KEIBA-MLが受け取れるようにする受け口。

BACKFILLがまだ値を持っていない時は `null` のままにする。
ML側で文字列から推測して埋め直さない。

## 学習候補へ流す項目

接頭辞 `backfill_` を付けて既存特徴と分離する。

- field_size
- course_layout
- course_laps
- race_class_normalized
- grade
- age_min
- age_max
- sex_condition
- weight_rule
- mixed
- international
- special_designated
- designated

この接頭辞により、従来の `style` 学習には自動で混ざらない。

## RAW文字列

以下はadapterで読めるが、L1の各馬データへ複製しない。

- race_class_raw
- age_condition_raw
- course_meta_raw
- race_condition_raw

正本はBACKFILLに残す。
同じ長文を全出走馬へ複製するとデータ容量が無駄に増えるため。

## 着差

以下を受け取れる。

- margin_raw
- margin_type
- margin_lengths

BACKFILLの正規化契約は以下。

- 数値着差 → `margin_type=LENGTHS` + `margin_lengths`
- ハナ / アタマ / クビ / 大差 / 同着など → `margin_type`のみ
- カテゴリ着差へ勝手な馬身換算はしない

現時点では受け口と完成検査まで。L1特徴への投入は別工程で行う。

## 不正値

たとえば `course_layout=MYSTERY` のように契約外の値が来た場合、

- 勝手にUNKNOWNへ丸めない
- 学習値はnull
- warningを出せる

という扱いにする。

## 学習段階

- `style`: 従来どおり。BACKFILL新条件を使わない
- `backfill_v1`: 新しいBACKFILL条件を追加
- `auto_v1`: AUTO特徴を追加
- `auto_backfill_v1`: AUTO + BACKFILL新条件

これで、何が効いたのかを分けて比較できる。
