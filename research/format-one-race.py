#!/usr/bin/env python3
"""User-readable one-race diagnosis, preserving strict prediction provenance.

No outcomes or payout columns; never blend live market with FINAL research.
Never promote uncalibrated gaps into 'edge' or bet suggestions.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def percent(value):
    try:n=float(value)
    except (TypeError,ValueError):return "—"
    if not math.isfinite(n):return "—"
    return f"{n*100:.1f}%"

def fnum(value):
    try:n=float(value)
    except (TypeError,ValueError):return "—"
    if not math.isfinite(n):return "—"
    return f"{n:.1f}"

def mark(value):
    if value is None:return "—"
    return str(value).replace("|","\\|").replace("\n"," ")

def build(report):
    if report.get("contract")!="KEIBA_ONE_RACE_COMBINED_ANALYSIS_V1":
        raise ValueError("UNKNOWN_ANALYSIS_CONTRACT")
    rid=str(report.get("race_id") or "")
    if len(rid)!=12 or not rid.isascii() or not rid.isdigit() or rid[:4]=="2026":
        raise ValueError("INVALID_OR_SEALED_RACE")
    horses=report.get("horses") or []
    if len(horses)<2 or len({h["horse_id"] for h in horses})!=len(horses):
        raise ValueError("HORSE_COVERAGE_INVALID")
    prior=(report.get("prior_year_new_model_inference") or {}).get("status")
    saved=(report.get("saved_seven_kings") or {}).get("status")
    l17=(report.get("saved_l17_podium") or {}).get("status")
    market=(report.get("market_context") or {}).get("status")
    final=(report.get("archived_final_market_fair_value") or {}).get("status")
    if market!="NOT_PROVIDED" and final=="HISTORICAL_FINAL_ODDS_OOS_RESEARCH_REPLAY":
        raise ValueError("DO_NOT_SHOW_FINAL_RESEARCH_ALONGSIDE_LIVE_QUOTES")
    has_prior=prior=="ORIGINAL_MODELS_SCORED_NOT_VERIFIED_PREBET"
    has_final=final=="HISTORICAL_FINAL_ODDS_OOS_RESEARCH_REPLAY"
    has_market=market=="TIMESTAMP_CHECKED_SOURCE_AUTHENTICITY_NOT_INDEPENDENTLY_ATTESTED"
    if has_prior:ordered=sorted(horses,key=lambda x:(x.get("prior_year_king_rank_sum",999),int(x.get("horse_number") or 99)))
    elif saved=="GITHUB_ARCHIVED_7KING_REPLAY":ordered=sorted(horses,key=lambda x:(x.get("archived_rank_sum_comparator_rank",999),int(x.get("horse_number") or 99)))
    else:ordered=sorted(horses,key=lambda x:int(x.get("horse_number") or 99))
    lines=[
        "# 1レース・全頭比較",
        "",
        f"レースID：`{rid}`　出走頭数：**{len(horses)}頭**",
        "",
        "## データ出所",
        "",
        f"- 過去年学習済み7王の新規推論：**{mark(prior)}**（前日までの実公開時刻は未立証）",
        f"- 保存済み7王＋Outsider：**{mark(saved)}**（研究時のFINAL入力）",
        f"- L1.7 3着内確率：**{mark(l17)}**（年外OOS保存済み）",
        f"- 発走前単勝オッズ：**{mark(market)}**（取得時刻と発走予定時刻の整合のみ検査）",
        f"- 最終オッズによる過去勝率研究：**{mark(final)}**（正式L1.75ではない）",
        "",
        "## 全頭の比較",
        "",
    ]
    columns=["馬番","馬名","過去年7王参考順","保存7王参考順","Out投票点","L1.7 3着内率"]
    if has_market:columns+=["当時単勝","市場勝率","市場人気","7王粗勝率"]
    if has_final:columns+=["研究時の最終市場勝率","研究時の補正勝率"]
    lines+=["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for h in ordered:
        row=[mark(h.get("horse_number")),mark(h.get("horse_name")),
             mark(h.get("prior_year_king_rank_sum")) if has_prior else "—",
             mark(h.get("archived_rank_sum_comparator_rank")) if saved=="GITHUB_ARCHIVED_7KING_REPLAY" else "—",
             mark(h.get("saved_outsider_points")),
             percent(h.get("l17_saved_p3_oos"))]
        if has_market:
            comp=h.get("market_comparison") or {}
            row += [fnum(h.get("market_win_odds")),percent(h.get("market_normalized_win_probability")),
                    mark(h.get("market_win_rank")),
                    percent(comp.get("uncalibrated_prior_year_king_mean_win_probability"))]
        if has_final:
            fair=(h.get("archived_final_market_oos_research") or {}).get("oos_win_probabilities") or {}
            row += [percent(fair.get("MARKET_DIRECT")),percent(fair.get("ANCHOR_FULL"))]
        lines.append("| "+" | ".join(row)+" |")
    lines += ["","## 注意事項","",
        "この表は独立した出所のモデル比較。過去研究に含まれる最終オッズは発走前に購入可能だった価格ではない。",
        "7王の平均raw勝率は校正済みの馬券期待値ではない。7王参考順位は凍結L2.1の軸ではない。",
        "**L1.75正式モデル・L2.1馬券生成・L3資金配分は、この表で実行していない。**",
        "2026年未知年の着順を使った評価は行っていない。",
    ]
    return "\n".join(lines)+"\n"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--analysis",required=True)
    p.add_argument("--out")
    a=p.parse_args()
    text=build(json.loads(Path(a.analysis).read_text(encoding="utf-8")))
    if a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding="utf-8")
    print(text)

if __name__=="__main__":main()
