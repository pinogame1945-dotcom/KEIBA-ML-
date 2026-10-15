# L2.1: 1レースの発走前組合せオッズ→凍結買い目の紙上価格付け

**段階**: SOURCE_CONTRACT + SYNTHETIC_CI_VALIDATED / REAL_LIVE_QUOTE_NOT_YET_VERIFIED

L2.1正本（読み取り専用）: `contracts/KEIBA_L2_1_FIXED_PURCHASE_V1.json`。
元の軸 `EXACT_EXISTING_ANCHOR_FULL`・9種類の元テンプレート・STANDARD 50個の`line×gate`を**変更しない**。STRICT 26個はSTANDARDの部分集合であり、追加購入の根拠にしない。

## 入力（すべてユーザーまたは元システムの実測値）

1. `--card`：KEIBA-MOBILEの`{race:JraRace,entries:JraEntry[]}`（SCHEDULED/ACTIVE）。レースIDは12桁`canonicalRaceId`、馬番と発走予定時刻が必要。既存年度の紙上研究は2022〜2025のみ、2026結果の未知年封印を守る。
2. `--original-features`：`KEIBA_L21_ORIGINAL_ANCHOR_FEATURES_PREBET_V1`。当該レースの元`ANCHOR_FULL`が**確定している**`anchor_horse_id`、全出走馬の`horse_id/horse_number/P__ANCHOR_FULL/risk/distill_score_z/distill_ai_rank/market_rank/market_prob_norm/win_odds_at_cutoff`、その取得時刻を含む。入力を生成する元のL1.75 / anchor処理は別途本番接続を監査する。2025保存済み`FINAL`、7王順位、L1.7の`p3`、保存済みOOSをこのファイルの代用品にしてはいけない。
3. `--combination-quotes`：`KEIBA_JRA_COMBINATION_ODDS_ASOF_V1`。同一レースのJRA馬連/三連複/三連単`OddsRow`、全部共通の`observed_at`、取引締切`bet_cutoff_at`、`scheduled_start_at`、`is_final:false`、`source_status:LIVE`、`source_confirmed_prestart:true`を含む。各行には`raceKey,betType,selection1,selection2,selection3,odds,oddsMin,oddsMax,observedAt,sourceUrl`。馬番順は馬連/三連複で**昇順正規化**、三連単では**着順どおり順序保持**。
   - `research/bridge-mobile-combination-odds.py` は、元のMOBILEの`{race,entries,odds_rows,capture}`をこの契約へ変換する。ただし`capture.final_probe_checked:true`、`capture.is_final:false`、`final_probe_checked_at`、`bet_cutoff_at`の証跡が必要で、**存在しない証跡を生成しない**。記録されたJSONはサーバー署名付き証明ではないため、真正性の外部検証は別に必要。

```bash
python3 research/bridge-mobile-combination-odds.py \
  --mobile-export local/original-mobile-race-and-odds.json \
  --output local/quote-book.json

python3 research/l21-frozen-prebet-price.py \
  --card local/jra-card.json \
  --original-features local/original-l21-horse-features.json \
  --combination-quotes local/quote-book.json \
  --policy STANDARD --out local/l21-ticket-price.json

python3 research/l21-format-price-preview.py \
  --tickets local/l21-ticket-price.json \
  --out local/l21-price-preview.md
```

### 凍結契約との実装対応

- `AI_PROB`：元`P__ANCHOR_FULL`降順、`market_rank`昇順、`horse_id`昇順。
- `AI_RAW`：元`distill_ai_rank`、`market_rank`、`horse_id`の昇順。
- `MARKET`：元`market_rank`、観測単勝オッズ、`horse_id`の昇順。事後`final_win_odds`は使わない。
- 軸は常に元`ANCHOR_FULL`。`k=2,3,5`の相手だけ使用し、`QUINELLA` / `TRIO` / `TRIFECTA_FIRST` / `TRIFECTA_THIRD`の元テンプレートを適用。
- オッズ帯は半開区間：`15-20 = [15,20)`、`20-30 = [20,30)`、`30-50 = [30,50)`、`200-300 = [200,300)`。等号の扱いを固定。
- 元`gate_flags`相当の条件は**券の軸以外の相手馬**について評価する。ALL、RAW/CALIBRATED/MARKET TOP6/TOP10、RISK_LT040/LT060、ABILITY_Z_GE_0/GE_M1、MODEL_WIN_P_GE_003、MARKET_UNDERVALUE_GE_2、AGREEMENT_TOP6。**50組中のルールのいずれも書き換えない**。
- `(race_id,bet_type,canonical_selection)`で1枚へ重複排除。重複ゲートは`matched_line_gates`に全て記録、掛け金は**券1点100円**のみ。L3への入力形は買い目を隠して増減させず同一キーを維持。

### 安全停止と未達部分

必要な組み合わせオッズが**1つでも欠けた場合**は`BLOCKED_INCOMPLETE_QUOTE_BOOK`を返し、漏れた組合せを報告し、購入候補を空にする。単勝や三連単のオッズを他券種から推定しない。部分約定に見えるリストを出さない。捕捉時刻が締切以降、事後`FINAL`、異なる観測時刻、同一物理券の重複オッズ、未出走馬の組合せ、未登録ソースURL、2026歴史後知恵の混入は停止。

`PAPER_ONLY_PREBET_QUOTE_VALIDATED`でも、これは入力の内部整合チェックであり**実売買承認ではない**。真の締切時刻・元JRAサーバー時刻・元の`ANCHOR_FULL`モデルの公開前特徴量時刻は独立検証されていない。

**現状、元のL1.75 `ANCHOR_FULL`正本のライブ推論出力と、JRA組合せオッズの完全な当日スナップショットを、同じレースIDで確保した実データ結合検証は未達。** 今回のPASSは正本規則50本からの合成データ紙上生成/価格付けの完全性を証明するものであり、正式L2.1本番購入を意味しない。凍結ROI109.18%は2023〜25の同年選択・同年評価であり、将来利益の証明ではない。

無料標準GitHub Actions CPUで`python3 scripts/l21-frozen-prebet-price-smoke.py`、`python3 scripts/mobile-combination-bridge-smoke.py`、`python3 scripts/verify_keiba_l2_1_freeze.py`のテストを実行。GPU・有料runner・有料Artifact/Cache・外部課金保存なし。
