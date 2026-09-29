#!/usr/bin/env python3
import argparse,csv,gzip,json
from collections import defaultdict
from pathlib import Path

from build_l2_bet_kings_dataset_v1 import (
    YEARS, load_fixed_ledgers, load_router, seven_stats,
    load_day, payout_map, horse_number_map
)

ANALYSIS_YEARS=(2023,2024,2025)
BET_TYPES=("QUINELLA","EXACTA","TRIO","TRIFECTA")

TEMPLATES={
    "QUINELLA_CORE12":"QUINELLA",
    "QUINELLA_AXIS1_KING":"QUINELLA",
    "QUINELLA_KING_TOP4_BOX":"QUINELLA",
    "EXACTA_CORE12_MULTI":"EXACTA",
    "EXACTA_AXIS1_FORWARD":"EXACTA",
    "EXACTA_AXIS1_REVERSE":"EXACTA",
    "EXACTA_AXIS1_MULTI":"EXACTA",
    "TRIO_AXIS12_ALL":"TRIO",
    "TRIO_AXIS1_ALL":"TRIO",
    "TRIO_KING_TOP6_BOX":"TRIO",
    "TRIFECTA_ANCHOR12_MULTI":"TRIFECTA",
    "TRIFECTA_A1_FIRST_ANCHOR2_MATE":"TRIFECTA",
    "TRIFECTA_ANCHORS_TOP2_MATE":"TRIFECTA",
    "TRIFECTA_KING_TOP4_BOX":"TRIFECTA",
}

CATEGORY_ORDER=(
    "PAYOUT_INCOMPLETE",
    "L1_WINNER_OUTSIDE",
    "L1_TOP2_PARTIAL",
    "L1_TOP3_PARTIAL",
    "L2_ROLE_MISS",
    "L2_HIT_BUT_PRICE_POINTS_LOSS",
    "L2_ROUTING_OPPORTUNITY",
)


def parse_args():
    p=argparse.ArgumentParser(description="PASS_SEVEN_ONLY loss anatomy V1.")
    p.add_argument("--fixed-ledger-dir",required=True)
    p.add_argument("--router-year",action="append",required=True)
    p.add_argument("--dataset-dir",required=True)
    p.add_argument("--backfill-root",required=True)
    p.add_argument("--out-dir",required=True)
    return p.parse_args()


def parse_paths(items):
    out={}
    for spec in items:
        y,p=spec.split(":",1)
        out[int(y)]=p
    return out


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def role_for_pos(pos):
    if pos==1: return "A1"
    if pos==2: return "A2"
    if pos in (3,4): return "KING3_4"
    if pos in (5,6): return "KING5_6"
    if pos and pos>=7: return "KING7_PLUS"
    return "OUTSIDE"


def load_ticket_aggregates(dataset_dir):
    root=Path(dataset_dir)
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract")!="L2_BET_KINGS_DATASET_V1":
        raise SystemExit("wrong dataset contract")
    if manifest.get("source_l15")!="L15_FIXED_V1":
        raise SystemExit("wrong L1.5 source")
    if manifest.get("locked_years")!=[2026]:
        raise SystemExit("2026 lock drift")

    agg={}
    for template,bet in TEMPLATES.items():
        info=manifest["templates"].get(template)
        if not info or info.get("bet_type")!=bet:
            raise SystemExit(f"template manifest drift {template}")
        with gzip.open(root/info["file"],"rt",encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r=json.loads(line)
                y=int(r["year"])
                if y not in ANALYSIS_YEARS: continue
                if int(r.get("gate_alert") or 0)!=0: continue
                if int(r.get("ticket_novel_count") or 0)!=0:
                    raise SystemExit(f"novel leakage template={template} race={r['race_id']}")
                rid=str(r["race_id"])
                k=(y,rid,template)
                z=agg.setdefault(k,{"tickets":0,"stake":0.0,"ret":0.0})
                z["tickets"]+=1
                z["stake"]+=100.0
                z["ret"]+=float(r.get("return_yen_per100") or 0.0)
    return manifest,agg


def winning_combos(payouts,bet):
    return [nums for (bt,nums),pay in payouts.items() if bt==bet and float(pay)>0]


def any_fully_covered(combos,candidate_numbers):
    c=set(candidate_numbers)
    return any(all(int(n) in c for n in nums) for nums in combos)


def main():
    a=parse_args()
    paths=parse_paths(a.router_year)
    if set(paths)!=set(YEARS):
        raise SystemExit(f"router years mismatch {sorted(paths)}")

    fixed=load_fixed_ledgers(a.fixed_ledger_dir)
    routers={y:load_router(paths[y],y) for y in YEARS}
    manifest,ticket_agg=load_ticket_aggregates(a.dataset_dir)

    date_to_races=defaultdict(list)
    race_owner={}
    expected_normal=defaultdict(int)
    for y in ANALYSIS_YEARS:
        alerts=set(fixed[y])
        for rid,row in routers[y].items():
            if rid in alerts:
                continue
            d=str(row.get("race_date") or "")[:10]
            date_to_races[d].append(rid)
            race_owner[rid]=y
            expected_normal[y]+=1

    anatomy=[]
    signature_counts=defaultdict(int)
    signature_den=defaultdict(int)
    bet_type_stats=defaultdict(lambda:{"races":0,"hit_races":0,"profit_races":0,"oracle_profit_sum":0.0})
    root=Path(a.backfill_root)

    for di,date in enumerate(sorted(date_to_races),1):
        wanted=set(date_to_races[date])
        day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
        for rid in sorted(wanted):
            y=race_owner[rid]
            pack=day.get(rid)
            if pack is None:
                raise SystemExit(f"missing race pack {rid}")
            router=routers[y][rid]
            seven_order,_=seven_stats(router)
            horse_no=horse_number_map(pack)
            if any(h not in horse_no for h in seven_order):
                raise SystemExit(f"seven horse number missing race={rid}")
            nums=[horse_no[h] for h in seven_order]
            num_to_pos={horse_no[h]:i+1 for i,h in enumerate(seven_order)}
            a1_no=nums[0] if nums else None
            a2_no=nums[1] if len(nums)>1 else None

            payouts,present=payout_map(pack)
            wins={bet:winning_combos(payouts,bet) for bet in ("WIN","QUINELLA","TRIO","TRIFECTA")}
            payout_complete=all(wins[b] for b in ("WIN","QUINELLA","TRIO"))

            win_cov=any_fully_covered(wins["WIN"],nums) if wins["WIN"] else None
            top2_cov=any_fully_covered(wins["QUINELLA"],nums) if wins["QUINELLA"] else None
            top3_cov=any_fully_covered(wins["TRIO"],nums) if wins["TRIO"] else None

            template_rows=[]
            for template,bet in TEMPLATES.items():
                z=ticket_agg.get((y,rid,template))
                if not z: continue
                profit=z["ret"]-z["stake"]
                template_rows.append((template,bet,z["tickets"],z["stake"],z["ret"],profit))

            hit_rows=[x for x in template_rows if x[4]>0]
            profit_rows=[x for x in template_rows if x[5]>0]
            any_hit=bool(hit_rows)
            any_profit=bool(profit_rows)
            if template_rows:
                best=max(template_rows,key=lambda x:(x[5],x[4],-x[3],x[0]))
                best_template,best_bet,best_tickets,best_stake,best_ret,best_profit=best
            else:
                best_template=best_bet=None
                best_tickets=0; best_stake=best_ret=best_profit=0.0

            if not payout_complete:
                category="PAYOUT_INCOMPLETE"
            elif not win_cov:
                category="L1_WINNER_OUTSIDE"
            elif not top2_cov:
                category="L1_TOP2_PARTIAL"
            elif not top3_cov:
                category="L1_TOP3_PARTIAL"
            elif not any_hit:
                category="L2_ROLE_MISS"
            elif not any_profit:
                category="L2_HIT_BUT_PRICE_POINTS_LOSS"
            else:
                category="L2_ROUTING_OPPORTUNITY"

            for bet in BET_TYPES:
                rows=[x for x in template_rows if x[1]==bet]
                if not rows: continue
                s=bet_type_stats[(y,bet)]
                s["races"]+=1
                s["hit_races"]+=int(any(x[4]>0 for x in rows))
                profitable=[x for x in rows if x[5]>0]
                s["profit_races"]+=int(bool(profitable))
                s["oracle_profit_sum"]+=max((x[5] for x in rows),default=0.0)

            # Exact trifecta role signature is diagnostic only; skip dead-heat/multi-payout races.
            trifecta_role_signature=""
            trifecta_rank_signature=""
            if len(wins["TRIFECTA"])==1:
                tri=wins["TRIFECTA"][0]
                roles=[]
                ranks=[]
                for no in tri:
                    pos=num_to_pos.get(int(no))
                    roles.append(role_for_pos(pos))
                    ranks.append(str(pos) if pos is not None else "OUT")
                trifecta_role_signature=">".join(roles)
                trifecta_rank_signature=">".join(ranks)
                signature_counts[(y,trifecta_role_signature)]+=1
                signature_den[y]+=1

            anatomy.append({
                "year":y,
                "race_id":rid,
                "race_date":date,
                "seven_pool_size":len(nums),
                "a1_no":a1_no,
                "a2_no":a2_no,
                "winner_in_seven":int(bool(win_cov)) if win_cov is not None else None,
                "top2_all_in_seven":int(bool(top2_cov)) if top2_cov is not None else None,
                "top3_all_in_seven":int(bool(top3_cov)) if top3_cov is not None else None,
                "any_template_hit":int(any_hit),
                "any_template_profitable":int(any_profit),
                "best_hindsight_template":best_template or "",
                "best_hindsight_bet_type":best_bet or "",
                "best_hindsight_tickets":best_tickets,
                "best_hindsight_stake_yen":best_stake,
                "best_hindsight_return_yen":best_ret,
                "best_hindsight_profit_yen":best_profit,
                "loss_anatomy_category":category,
                "trifecta_role_signature":trifecta_role_signature,
                "trifecta_consensus_rank_signature":trifecta_rank_signature,
            })

        if di%50==0:
            print(f"NORMAL_LOSS_ANATOMY_PROGRESS dates={di}/{len(date_to_races)} races={len(anatomy)}",flush=True)

    # Guards.
    actual=defaultdict(int)
    for r in anatomy: actual[int(r["year"])]+=1
    for y in ANALYSIS_YEARS:
        if actual[y]!=expected_normal[y]:
            raise SystemExit(f"normal race count drift y={y} actual={actual[y]} expected={expected_normal[y]}")

    category_rows=[]
    year_rows=[]
    for y in ANALYSIS_YEARS:
        yy=[r for r in anatomy if int(r["year"])==y]
        n=len(yy)
        for cat in CATEGORY_ORDER:
            c=sum(r["loss_anatomy_category"]==cat for r in yy)
            category_rows.append({
                "year":y,"category":cat,"races":c,"share_pct":100*c/n if n else None
            })
        year_rows.append({
            "year":y,
            "normal_races":n,
            "winner_in_seven_pct":100*sum(r["winner_in_seven"]==1 for r in yy)/n if n else None,
            "top2_all_in_seven_pct":100*sum(r["top2_all_in_seven"]==1 for r in yy)/n if n else None,
            "top3_all_in_seven_pct":100*sum(r["top3_all_in_seven"]==1 for r in yy)/n if n else None,
            "any_template_hit_pct":100*sum(r["any_template_hit"]==1 for r in yy)/n if n else None,
            "any_template_profitable_pct":100*sum(r["any_template_profitable"]==1 for r in yy)/n if n else None,
            "routing_opportunity_races":sum(r["loss_anatomy_category"]=="L2_ROUTING_OPPORTUNITY" for r in yy),
            "role_miss_races":sum(r["loss_anatomy_category"]=="L2_ROLE_MISS" for r in yy),
            "hit_but_price_points_loss_races":sum(r["loss_anatomy_category"]=="L2_HIT_BUT_PRICE_POINTS_LOSS" for r in yy),
        })

    bet_rows=[]
    for y in ANALYSIS_YEARS:
        for bet in BET_TYPES:
            s=bet_type_stats[(y,bet)]
            n=s["races"]
            bet_rows.append({
                "year":y,"bet_type":bet,"races":n,
                "any_hit_races":s["hit_races"],
                "any_hit_rate_pct":100*s["hit_races"]/n if n else None,
                "any_profitable_template_races":s["profit_races"],
                "any_profitable_template_rate_pct":100*s["profit_races"]/n if n else None,
                "hindsight_best_profit_sum_yen":s["oracle_profit_sum"],
            })

    sig_rows=[]
    for (y,sig),count in sorted(signature_counts.items(),key=lambda x:(x[0][0],-x[1],x[0][1])):
        sig_rows.append({
            "year":y,"trifecta_role_signature":sig,
            "races":count,
            "share_of_unique_trifecta_pct":100*count/signature_den[y] if signature_den[y] else None,
        })

    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"year-summary.csv",year_rows)
    write_csv(out/"category-summary.csv",category_rows)
    write_csv(out/"bet-type-opportunity.csv",bet_rows)
    write_csv(out/"trifecta-role-signatures.csv",sig_rows)
    with gzip.open(out/"race-anatomy.csv.gz","wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(anatomy[0]))
        w.writeheader(); w.writerows(anatomy)

    summary={
        "contract":"L2_NORMAL_LOSS_ANATOMY_V1",
        "source_structure_run":36524917751,
        "scope":"PASS_SEVEN_ONLY only",
        "analysis_years":list(ANALYSIS_YEARS),
        "normal_races":{str(y):actual[y] for y in ANALYSIS_YEARS},
        "race_filtering":False,
        "all_normal_races_used":True,
        "categories":{
            "L1_WINNER_OUTSIDE":"Winning horse is outside Seven-King candidate pool.",
            "L1_TOP2_PARTIAL":"Winner is visible, but the winning top-two pair is not fully inside Seven-King pool.",
            "L1_TOP3_PARTIAL":"Top two are visible, but the podium trio is not fully inside Seven-King pool.",
            "L2_ROLE_MISS":"Podium trio is fully visible, but none of the coarse role templates hits.",
            "L2_HIT_BUT_PRICE_POINTS_LOSS":"At least one coarse template hits, but every hitting template still loses money after ticket count/price.",
            "L2_ROUTING_OPPORTUNITY":"At least one existing coarse template would have been profitable on the race in hindsight.",
        },
        "hindsight_guard":"Best-template and routing-opportunity fields are diagnostic upper bounds only, never a deployable selector.",
        "2026_locked":True,
        "production_promotion":False,
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"README.md").write_text(
        "# L2 Normal Loss Anatomy V1\n\n"
        "PASS_SEVEN_ONLY races only. No race filtering. This diagnostic separates candidate-pool misses "
        "from L2 role misses, ticket-spread/price losses, and hindsight routing opportunities across the "
        "existing QUINELLA/EXACTA/TRIO/TRIFECTA coarse templates. "
        "Hindsight best-template fields are diagnostic upper bounds, not production rules. 2026 remains sealed.\n",
        encoding="utf-8",
    )
    print("L2_NORMAL_LOSS_ANATOMY_V1_READY",flush=True)
    print(json.dumps(summary,ensure_ascii=False,separators=(",",":")),flush=True)


if __name__=="__main__":
    main()
