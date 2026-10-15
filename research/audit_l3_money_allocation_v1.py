#!/usr/bin/env python3
"""Audit funded fixed-ticket L3: contribution attribution + random rank null.
Descriptive only: baseline L2.1 was selected with 2023-2025 hindsight.
"""
import argparse
import csv
import gzip
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fp:
        yield from csv.DictReader(fp)


def key(r):
    return (r["year"], r["race_id"], r["bet_type"], r["selection"])


def interval(x):
    v = sorted(x)
    return [v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    dest=Path(args.out_dir)
    inp=dest/"ticket_stakes.csv.gz"
    allocations=defaultdict(dict)
    races=defaultdict(list)
    for r in rows(inp):
        pol=r["policy"]; k=key(r)
        if k in allocations[pol]:
            raise ValueError(("duplicate ticket",pol,k))
        allocations[pol][k]=r
        if pol=="FLAT200":
            races[(r["year"],r["race_id"])].append(k)
    expected=set(allocations["FLAT200"])
    for pol,part in allocations.items():
        if set(part)!=expected: raise ValueError(("ticket key drift",pol))
    counts=Counter(len(v) for v in races.values())
    gains={}
    for pol in allocations:
        if pol in ("FLAT100","FLAT200"):continue
        gain=sum((int(r["stake_yen"])-200)*int(r["return_yen"])//int(r["stake_yen"])
                 for r in allocations[pol].values())
        # Return difference from FLAT200: (stake-200)*(per100_return/100).
        # Integer exact: all stakes are positive multiples of 100.
        gains[pol]=gain
    # Attribution by year and bet type, no re-selection or optimization.
    details=[]
    for pol in gains:
        for yr in ("2023","2024","2025"):
            for bet in ("QUINELLA","TRIO","TRIFECTA"):
                part=[r for k,r in allocations[pol].items()
                      if k[0]==yr and k[2]==bet]
                diff=sum((int(r["stake_yen"])-200)*int(r["return_yen"])//
                         int(r["stake_yen"]) for r in part)
                details.append(dict(policy=pol,year=yr,bet_type=bet,
                                    tickets=len(part),delta_profit_vs_flat200_yen=diff))
    with (dest/"attribution_bet_type_year.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]))
        writer.writeheader();writer.writerows(details)

    policy="REVERSE_P3"
    rev=allocations[policy]
    ticket_contributions=[]
    for k,r in rev.items():
        stake=int(r["stake_yen"])
        actual_return=int(r["return_yen"])
        per100=actual_return*100//stake
        if actual_return*100%stake:raise ValueError("non-integral per100")
        diff=(stake-200)*per100//100
        if diff and per100>0:
            ticket_contributions.append({
                "year":k[0],"race_id":k[1],"bet_type":k[2],
                "selection":k[3],"stake_yen":stake,
                "payout_per100":per100,"delta_payout_yen":diff})
    ticket_contributions.sort(key=lambda x:x["delta_payout_yen"],reverse=True)
    winners=ticket_contributions[:10]
    positives=sum(max(0,x["delta_payout_yen"]) for x in ticket_contributions)
    negatives=sum(min(0,x["delta_payout_yen"]) for x in ticket_contributions)
    # Empirical null: rank randomly within each race but keep original tickets,
    # same stake cap [100,300] and the same fixed 200*N race budget.
    packs=[]
    for k_list in races.values():
        n=len(k_list)
        if n<=1:continue
        pay=[int(allocations["FLAT200"][k]["return_yen"])//2 for k in k_list]
        packs.append((n,pay))
    rng=random.Random(2026101501)
    null=[]
    for _ in range(1200):
        diff=0
        for n,pay in packs:
            order=list(range(n))
            rng.shuffle(order)
            extra=[0]*n; left=n
            for i in order:
                add=min(2,left);extra[i]=add;left-=add
                if left==0: break
            diff+=sum((extra[i]-1)*pay[i] for i in range(n))
        null.append(diff)
    observed=gains[policy]
    # Fraction of random rank allocations beating the chosen reversed-p3 rule.
    frac=(sum(x>=observed for x in null)+1)/(len(null)+1)
    summary=dict(
        caveat="L2.1 rule selected on same 2023-25 outcomes; randomisation is not a future-outcome test.",
        all_ticket_membership_unchanged=True,
        number_of_races_by_ticket_count=dict(sorted(counts.items())),
        races_where_stake_choice_possible=sum(n for size,n in counts.items() if size>1),
        winning_stake_adjustments=counts.get(1,0),
        extra_profit_vs_flat200_by_policy_yen=gains,
        reverse_p3_excess_profit_yen=observed,
        reverse_p3_top_10_positive_ticket_deltas=winners,
        reverse_p3_top_3_positive_ticket_delta_yen=sum(x["delta_payout_yen"] for x in winners[:3]),
        reverse_p3_winner_contribution_positive_yen=positives,
        reverse_p3_winner_contribution_negative_yen=negatives,
        random_rank_null=dict(
            iterations=len(null),mean_extra_profit_yen=round(statistics.mean(null)),
            median_extra_profit_yen=round(statistics.median(null)),
            stdev_yen=round(statistics.pstdev(null)),
            percentile95_yen=interval(null),maximum_yen=max(null),
            observed_reverse_p3_profit_greater_than_null_fraction=round(1-frac,5),
            tail_probability_random_exceeds_reverse_p3=round(frac,5)),
    )
    (dest/"allocation_audit.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("L3_ATTRIBUTION_AUDIT_DONE",json.dumps({
        "reverse_p3_delta":observed,"random_null":summary["random_rank_null"],
        "races_with_choices":summary["races_where_stake_choice_possible"]}),flush=True)

if __name__=="__main__":
    main()
