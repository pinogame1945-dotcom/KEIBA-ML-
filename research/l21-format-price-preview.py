#!/usr/bin/env python3
"""Readable L2.1 price preview; never implies live execution or profit.
Sorts by bet type / canonical numbers. On quote coverage failure, displays
missing quote keys instead of inventing partial buying recommendations.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

FAMILY={"QUINELLA":"馬連","TRIO":"三連複","TRIFECTA":"三連単"}
def render(payload):
    if (payload.get("contract")!="KEIBA_L21_PREBET_QUOTED_TICKETS_V1" or
        payload.get("frozen_policy")!="KEIBA_L2_1_FIXED_PURCHASE_V1"):
        raise ValueError("UNRECOGNIZED_L21_OUTPUT")
    status=payload.get("status")
    if status not in ("PAPER_ONLY_PREBET_QUOTE_VALIDATED","BLOCKED_INCOMPLETE_QUOTE_BOOK"):
        raise ValueError("UNRECOGNIZED_L21_STATUS")
    lines=["# L2.1 凍結買い目・発走前オッズ照合（紙上プレビュー）","",
      f"レースID：`{payload['race_id']}`",
      f"使用方針：`{payload['policy']}`（変更なし）",
      f"元軸馬：馬番 **{payload['axis_horse_number']}**、ID `{payload['axis_horse_id']}`",
      f"オッズ観測時刻：`{payload['quote_observed_at']}`",
      f"馬券締切時刻：`{payload['bet_cutoff_at']}`",
      f"状態：**{status}**",
      "",]
    if status=="BLOCKED_INCOMPLETE_QUOTE_BOOK":
        lines+=["必要な組み合わせオッズが不足したため、**購入候補は出力しない**。",
            f"不足した物理オッズ：**{payload['ticket_quote_missing_unique_count']}通り**",
            "","| 券種 | 不足組合せ |","|---|---|"]
        for item in payload["missing_quote_tickets"][:40]:
            lines.append(f"| {FAMILY[item['bet_type']]} | {item['canonical_selection']} |")
        if len(payload["missing_quote_tickets"])>40:lines.append("| … | 40件まで表示 |")
    else:
        n=payload["physical_ticket_count"]
        lines+=["| 券種 | 組合せ | 観測オッズ | 100円基準 | ルール一致数 | 厳格版にも存在 |",
            "|---|---|---:|---:|---:|---|"]
        for t in payload["tickets"]:
            lines.append(f"| {FAMILY[t['bet_type']]} | {t['canonical_selection']} | "+
                f"{t['decimal_odds']:.1f} | ¥100 | {len(t['matched_line_gates'])} | "+
                ("○" if t["strict_subset"] else "—")+" |")
        lines += ["",f"**物理的な購入候補：{n}点**、100円基準の合計：**¥{n*100:,}**",
            f"重複排除前のルール一致数：{payload['matching_line_gate_count']}（買い増しなし）"]
    lines+=["",
      "## 判定に関する制限","",
      "- 買い目の軸・順位・条件は凍結L2.1正本を前提。7王の代理値に置き換えない。",
      "- 入力JSONの時刻・JRA出典やモデル由来の真正性は独立認証していない。",
      "- この表は自動投票指示ではない。L3の掛け金最適化・購入APIは未接続。",
      "- 2023〜2025年の後知恵選択ROI109.18%は未来の利益を保証しない。",
      "- 2026年の未知年履歴・確定オッズ・払戻の読み込みは行わない。"]
    return "\n".join(lines)+"\n"
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tickets",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    result=render(json.loads(Path(a.tickets).read_text(encoding="utf-8")))
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(result,encoding="utf-8")
    print(result)
if __name__=="__main__":main()
