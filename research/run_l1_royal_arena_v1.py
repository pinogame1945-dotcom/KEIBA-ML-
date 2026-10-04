#!/usr/bin/env python3
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

EXPECTED_RACES=3456
LOCKED_YEAR=2026
EXPERT_ALIASES=("core4","pedlegacy","condition","full","light","pedv1","condrc")

def open_text(path,mode="rt"):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def finite(v):
    try:
        if isinstance(v,str): v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError):
        return None

def horse_no(v):
    try: return int(float(v))
    except (TypeError,ValueError): return 10**9

def load_expert(path):
    races=defaultdict(list)
    with open_text(path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("contract")!="L1_TO_L2_OUTPUT_CONTRACT_V1":
                raise SystemExit(f"bad expert contract: {path}")
            rid=str(r.get("race_id") or ""); hid=str(r.get("horse_id") or "")
            date=str(r.get("race_date") or "")[:10]
            rank=int(r.get("predicted_rank") or 0)
            if not rid or not hid or len(date)!=10 or rank<1:
                raise SystemExit(f"bad expert row path={path} race={rid} horse={hid}")
            races[rid].append({
                "horse_id":hid,"horse_number":horse_no(r.get("horse_number")),
                "rank":rank,"probability":finite(r.get("race_normalized_win_probability")),
                "race_date":date,
            })
    if len(races)!=EXPECTED_RACES:
        raise SystemExit(f"expert race coverage {path}: {len(races)} != {EXPECTED_RACES}")
    for rid,rows in races.items():
        rows.sort(key=lambda x:x["rank"])
        if [x["rank"] for x in rows] != list(range(1,len(rows)+1)):
            raise SystemExit(f"non-contiguous safe ranks path={path} race={rid}")
        if len({x["horse_id"] for x in rows})!=len(rows):
            raise SystemExit(f"duplicate horse path={path} race={rid}")
    return races

def load_ballots(path):
    out=defaultdict(dict)
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            rid=str(r.get("race_id") or ""); cand=str(r.get("candidate") or "")
            ids=[str(r.get(f"top{i}_horse_id") or "") for i in (1,2,3)]
            ids=[x for x in ids if x]
            if not rid or not cand or not ids: continue
            out[cand][rid]=ids
    if len(out)<10:
        raise SystemExit(f"too few outsider candidates: {sorted(out)}")
    for cand,rows in out.items():
        if len(rows)!=EXPECTED_RACES:
            raise SystemExit(f"outsider coverage {cand}: {len(rows)} != {EXPECTED_RACES}")
    return out

def load_day(path,wanted):
    rows={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); rid=str((r.get("race") or {}).get("race_id") or "")
            if rid in wanted: rows[rid]=r
    return rows

def load_odds_day(path,wanted):
    rows={}
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); rid=str(r.get("race_id") or "")
            if rid in wanted: rows[rid]=r
    return rows

def final_odds_tuple(raw):
    if not isinstance(raw,list) or len(raw)<3: return None
    return raw[3:6] if len(raw)>=6 else raw[:3]

def race_outcome(pack):
    finish={}
    winners=set(); podium=set()
    for r in pack.get("results") or []:
        if r.get("result_status")!="FINISHED": continue
        hid=str(r.get("horse_id") or "")
        try: pos=int(r.get("official_finish_position"))
        except (TypeError,ValueError): continue
        if not hid: continue
        finish[hid]=pos
        if pos==1: winners.add(hid)
        if pos<=3: podium.add(hid)
    return finish,winners,podium

def market_order(orec,horse_to_no):
    raw=((orec.get("odds") or {}).get("1") or {})
    odds={}
    for k,v in raw.items():
        ks=str(k)
        if not ks.isascii() or not ks.isdigit(): continue
        tup=final_odds_tuple(v)
        if not tup: continue
        odd=finite(tup[0])
        if odd is not None and odd>0: odds[int(ks)]=odd
    needed=set(horse_to_no.values())
    if set(odds)!=needed:
        return None,None,{"got":len(odds),"expected":len(needed)}
    inv={no:hid for hid,no in horse_to_no.items()}
    order=[inv[no] for no in sorted(odds,key=lambda no:(odds[no],no,inv[no]))]
    comp={}
    vals=list(odds.values())
    for hid,no in horse_to_no.items():
        comp[hid]=1+sum(1 for x in vals if x<odds[no])
    return order,comp,odds

def expert_maps(expert_races,rid):
    rows=expert_races[rid]
    return {x["horse_id"]:x for x in rows}

def consensus_order(per_alias,aliases,market_ranks=None):
    horses=set(next(iter(per_alias.values())))
    if any(set(m)!=horses for m in per_alias.values()):
        raise SystemExit("cross-expert horse coverage drift")
    scored=[]
    for hid in horses:
        ranks=[int(per_alias[a][hid]["rank"]) for a in aliases]
        probs=[per_alias[a][hid].get("probability") for a in aliases]
        probs=[x for x in probs if x is not None]
        if market_ranks is not None: ranks=ranks+[float(market_ranks[hid])]
        scored.append({
            "hid":hid,
            "mean_rank":sum(ranks)/len(ranks),
            "top6":sum(r<=6 for r in ranks),
            "top3":sum(r<=3 for r in ranks),
            "top1":sum(r==1 for r in ranks),
            "mean_prob":sum(probs)/len(probs) if probs else 0.0,
            "horse_number":per_alias[aliases[0]][hid]["horse_number"],
        })
    scored.sort(key=lambda x:(x["mean_rank"],-x["top6"],-x["top3"],-x["top1"],-x["mean_prob"],x["horse_number"],x["hid"]))
    return [x["hid"] for x in scored]

def empty_acc():
    return defaultdict(float)

def add_metrics(acc,order,winners,podium,market_comp=None,full_rank=True):
    if not order: return
    top1=order[:1]; top3=order[:3]; top6=order[:6]
    acc["races"]+=1
    acc["top1_win_hit"]+=int(bool(set(top1)&winners))
    acc["top1_top3_hit"]+=int(bool(set(top1)&podium))
    acc["winner_top3_hit"]+=int(bool(set(top3)&winners))
    acc["podium_hits_top3"]+=len(set(top3)&podium)
    acc["top3_selected"]+=len(top3)
    if market_comp is not None:
        market_top3={h for h,r in market_comp.items() if r<=3}
        nonmarket=[h for h in top3 if market_comp.get(h,999)>3]
        acc["top3_nonmarket_selected"]+=len(nonmarket)
        acc["top3_nonmarket_top3_hit"]+=sum(h in podium for h in nonmarket)
        acc["top3_nonmarket_win_hit"]+=sum(h in winners for h in nonmarket)
        union=set(top3)|market_top3
        acc["top3_jaccard_market_sum"]+=(len(set(top3)&market_top3)/len(union) if union else 1.0)
        acc["top3_jaccard_market_n"]+=1
    if full_rank:
        acc["full_rank_races"]+=1
        acc["winner_top6_hit"]+=int(bool(set(top6)&winners))
        acc["podium_hits_top6"]+=len(set(top6)&podium)
        acc["top6_selected"]+=len(top6)

def corr(xs,ys):
    n=len(xs)
    if n<2:return 1.0
    mx=sum(xs)/n; my=sum(ys)/n
    dx=[x-mx for x in xs]; dy=[y-my for y in ys]
    vx=sum(x*x for x in dx); vy=sum(y*y for y in dy)
    if vx<=0 or vy<=0:return 1.0
    return sum(x*y for x,y in zip(dx,dy))/math.sqrt(vx*vy)

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def cmd_year(a):
    year=int(a.year)
    if year==LOCKED_YEAR: raise SystemExit("2026 sealed")
    specs={}
    for s in a.expert:
        name,path=s.split("=",1); specs[name]=path
    if set(specs)!=set(EXPERT_ALIASES):
        raise SystemExit(f"expert set drift {sorted(specs)}")
    experts={name:load_expert(path) for name,path in specs.items()}
    race_sets=[set(x) for x in experts.values()]
    if any(s!=race_sets[0] for s in race_sets[1:]): raise SystemExit("expert race coverage mismatch")
    ballots=load_ballots(a.ballots)

    dates=defaultdict(list)
    date_by_race={}
    base=experts[EXPERT_ALIASES[0]]
    for rid,rows in base.items():
        date=rows[0]["race_date"]
        if not date.startswith(str(year)+"-"): raise SystemExit(f"year/date drift race={rid} date={date}")
        dates[date].append(rid); date_by_race[rid]=date

    acc=defaultdict(empty_acc)
    market_scope_acc=defaultdict(empty_acc)
    loo=defaultdict(empty_acc)
    pair=defaultdict(lambda:defaultdict(float))
    rescue=defaultdict(lambda:defaultdict(float))
    root=Path(a.backfill_root)
    processed=0
    market_covered=0
    market_missing=[]

    for date,rids in sorted(dates.items()):
        wanted=set(rids)
        dpath=root/"data"/"daily"/f"{date}.jsonl.gz"
        opath=root/"data"/"odds"/"daily"/f"{date}.jsonl.gz"
        if not dpath.exists() or not opath.exists(): raise SystemExit(f"missing backfill files date={date}")
        day=load_day(dpath,wanted); oddsday=load_odds_day(opath,wanted)
        for rid in rids:
            pack=day.get(rid); orec=oddsday.get(rid)
            if pack is None or orec is None: raise SystemExit(f"missing backfill row race={rid}")
            finish,winners,podium=race_outcome(pack)
            if not winners or not podium: raise SystemExit(f"outcome missing race={rid}")
            per={alias:expert_maps(experts[alias],rid) for alias in EXPERT_ALIASES}
            horses=set(next(iter(per.values())))
            if any(set(x)!=horses for x in per.values()): raise SystemExit(f"expert horse drift race={rid}")
            horse_to_no={hid:per[EXPERT_ALIASES[0]][hid]["horse_number"] for hid in horses}
            market,market_comp,market_meta=market_order(orec,horse_to_no)
            market_ok=market is not None
            if market_ok:
                market_covered+=1
            else:
                market_missing.append({"year":year,"race_id":rid,"race_date":date,**market_meta})
            council=consensus_order(per,EXPERT_ALIASES)

            orders={alias:[x["horse_id"] for x in experts[alias][rid]] for alias in EXPERT_ALIASES}
            orders["COUNCIL7"]=council
            if market_ok:
                orders["MARKET"]=market
                orders["COUNCIL7_PLUS_MARKET"]=consensus_order(per,EXPERT_ALIASES,market_comp)
            for name,order in orders.items():
                add_metrics(acc[name],order,winners,podium,market_comp if market_ok else None,True)
                if market_ok:
                    add_metrics(market_scope_acc[name],order,winners,podium,market_comp,True)

            for cand,byrace in ballots.items():
                order=byrace[rid]
                add_metrics(acc["OUTSIDER:"+cand],order,winners,podium,market_comp if market_ok else None,False)
                if market_ok:
                    add_metrics(market_scope_acc["OUTSIDER:"+cand],order,winners,podium,market_comp,False)
                r=rescue[cand]
                miss3=not bool(set(council[:3])&winners)
                miss6=not bool(set(council[:6])&winners)
                mmiss3=(not bool(set(market[:3])&winners)) if market_ok else None
                r["races"]+=1
                r["council_top3_miss_n"]+=int(miss3)
                r["council_top3_miss_rescued"]+=int(miss3 and bool(set(order[:3])&winners))
                r["council_top6_miss_n"]+=int(miss6)
                r["council_top6_miss_rescued"]+=int(miss6 and bool(set(order[:3])&winners))
                if market_ok:
                    both=miss3 and mmiss3
                    r["council_market_top3_both_miss_n"]+=int(both)
                    r["council_market_both_miss_rescued"]+=int(both and bool(set(order[:3])&winners))
                    novel=[h for h in order[:3] if h not in set(council[:3]) and market_comp.get(h,999)>3]
                    r["novel_selected"]+=len(novel)
                    r["novel_actual_top3"]+=sum(h in podium for h in novel)
                    r["novel_winner"]+=sum(h in winners for h in novel)

            for omit in EXPERT_ALIASES:
                aliases=tuple(x for x in EXPERT_ALIASES if x!=omit)
                order=consensus_order(per,aliases)
                add_metrics(loo[omit],order,winners,podium,market_comp if market_ok else None,True)

            for i,a1 in enumerate(EXPERT_ALIASES):
                r1={h:int(per[a1][h]["rank"]) for h in horses}
                s1=set(orders[a1][:3])
                for a2 in EXPERT_ALIASES[i+1:]:
                    r2={h:int(per[a2][h]["rank"]) for h in horses}
                    ids=sorted(horses)
                    key=f"{a1}|{a2}"
                    pair[key]["races"]+=1
                    pair[key]["rank_corr_sum"]+=corr([r1[h] for h in ids],[r2[h] for h in ids])
                    s2=set(orders[a2][:3]); u=s1|s2
                    pair[key]["top3_jaccard_sum"]+=(len(s1&s2)/len(u) if u else 1.0)
            processed+=1

    if processed!=EXPECTED_RACES: raise SystemExit(f"processed {processed} != {EXPECTED_RACES}")
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"participant-counts.csv",[{"year":year,"participant":k,**dict(v)} for k,v in sorted(acc.items())])
    write_csv(out/"participant-market-covered-counts.csv",[{"year":year,"participant":k,**dict(v)} for k,v in sorted(market_scope_acc.items())])
    write_csv(out/"market-missing-races.csv",market_missing)
    write_csv(out/"leave-one-out-counts.csv",[{"year":year,"omitted":k,**dict(v)} for k,v in sorted(loo.items())])
    write_csv(out/"pairwise-counts.csv",[{"year":year,"pair":k,**dict(v)} for k,v in sorted(pair.items())])
    write_csv(out/"outsider-rescue-counts.csv",[{"year":year,"candidate":k,**dict(v)} for k,v in sorted(rescue.items())])
    summary={
        "contract":"L1_ROYAL_ARENA_YEAR_V1","year":year,"races":processed,
        "experts":list(EXPERT_ALIASES),"outsiders":sorted(ballots),
        "market":"final WIN odds; deterministic horse-number tie break, competition rank retained for independence metrics",
        "market_covered_races":market_covered,
        "market_missing_races":len(market_missing),
        "market_missing_policy":"market-only metrics excluded; Seven-King, Council, Outsider and leave-one-out metrics continue on all races",
        "council":"tie-safe ranks only; no result/order leakage",
        "2026_locked":True,"paid_compute":False,"artifact_cache":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L1_ROYAL_ARENA_YEAR_V1_READY",json.dumps(summary,ensure_ascii=False,separators=(",",":")))

def read_rows(path):
    with open(path,newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def num(v):
    x=finite(v); return 0.0 if x is None else x

def derive(r):
    races=num(r.get("races"))
    full=num(r.get("full_rank_races"))
    def pct(n,d): return 100*n/d if d else None
    return {
        "races":int(races),
        "top1_win_pct":pct(num(r.get("top1_win_hit")),races),
        "top1_top3_pct":pct(num(r.get("top1_top3_hit")),races),
        "winner_top3_capture_pct":pct(num(r.get("winner_top3_hit")),races),
        "winner_top6_capture_pct":pct(num(r.get("winner_top6_hit")),full),
        "podium_precision_top3_pct":pct(num(r.get("podium_hits_top3")),num(r.get("top3_selected"))),
        "podium_precision_top6_pct":pct(num(r.get("podium_hits_top6")),num(r.get("top6_selected"))),
        "nonmarket_top3_actual_top3_pct":pct(num(r.get("top3_nonmarket_top3_hit")),num(r.get("top3_nonmarket_selected"))),
        "nonmarket_top3_winner_pct":pct(num(r.get("top3_nonmarket_win_hit")),num(r.get("top3_nonmarket_selected"))),
        "mean_top3_jaccard_vs_market":num(r.get("top3_jaccard_market_sum"))/num(r.get("top3_jaccard_market_n")) if num(r.get("top3_jaccard_market_n")) else None,
    }

def sum_group(rows,key):
    out={}
    for r in rows:
        k=r[key]
        if k not in out: out[k]=defaultdict(float)
        for c,v in r.items():
            if c in (key,"year"): continue
            x=finite(v)
            if x is not None: out[k][c]+=x
    return out

def cmd_aggregate(a):
    years=[int(x) for x in a.years.split(",")]
    if LOCKED_YEAR in years: raise SystemExit("2026 sealed")
    root=Path(a.root); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    parts=[]; market_parts=[]; loos=[]; pairs=[]; rescues=[]
    per_year={}; per_year_market={}
    for y in years:
        d=root/f"y{y}"
        for fn in ("participant-counts.csv","participant-market-covered-counts.csv","leave-one-out-counts.csv","pairwise-counts.csv","outsider-rescue-counts.csv","summary.json"):
            if not (d/fn).exists(): raise SystemExit(f"missing year result {d/fn}")
        yp=read_rows(d/"participant-counts.csv"); parts+=yp
        ymp=read_rows(d/"participant-market-covered-counts.csv"); market_parts+=ymp
        loos+=read_rows(d/"leave-one-out-counts.csv"); pairs+=read_rows(d/"pairwise-counts.csv"); rescues+=read_rows(d/"outsider-rescue-counts.csv")
        per_year[y]={r["participant"]:derive(r) for r in yp}
        per_year_market[y]={r["participant"]:derive(r) for r in ymp}

    pg=sum_group(parts,"participant")
    mpg=sum_group(market_parts,"participant")
    leaderboard=[]
    market_by_year={y:per_year_market[y]["MARKET"] for y in years}
    for name,c in sorted(pg.items()):
        d=derive(c)
        md=derive(mpg[name]) if name in mpg else {}
        d.update({
            "participant":name,
            "market_covered_races":md.get("races"),
            "market_covered_winner_top3_capture_pct":md.get("winner_top3_capture_pct"),
            "market_covered_top1_top3_pct":md.get("top1_top3_pct"),
        })
        d["years_beat_market_winner_top3"]=sum(
            1 for y in years
            if name in per_year_market[y] and per_year_market[y][name]["winner_top3_capture_pct"] is not None
            and per_year_market[y][name]["winner_top3_capture_pct"]>market_by_year[y]["winner_top3_capture_pct"]
        )
        d["years_beat_market_top1_top3"]=sum(
            1 for y in years
            if name in per_year_market[y] and per_year_market[y][name]["top1_top3_pct"] is not None
            and per_year_market[y][name]["top1_top3_pct"]>market_by_year[y]["top1_top3_pct"]
        )
        leaderboard.append(d)
    leaderboard.sort(key=lambda r:(-(r["winner_top3_capture_pct"] or -1),-(r["top1_top3_pct"] or -1),r["participant"]))
    write_csv(out/"leaderboard.csv",leaderboard)

    loo_g=sum_group(loos,"omitted")
    full=derive(pg["COUNCIL7"])
    contrib=[]
    for omit,c in sorted(loo_g.items()):
        d=derive(c)
        delta=(full["winner_top3_capture_pct"]-d["winner_top3_capture_pct"]) if d["winner_top3_capture_pct"] is not None else None
        hurts=0; helps=0
        for y in years:
            full_y=per_year[y]["COUNCIL7"]["winner_top3_capture_pct"]
            row=next(r for r in loos if int(r["year"])==y and r["omitted"]==omit)
            minus_y=derive(row)["winner_top3_capture_pct"]
            hurts+=int(full_y>minus_y); helps+=int(full_y<minus_y)
        contrib.append({
            "expert":omit,
            "council7_winner_top3_pct":full["winner_top3_capture_pct"],
            "council_minus_expert_winner_top3_pct":d["winner_top3_capture_pct"],
            "removal_cost_pp":delta,
            "years_removal_hurts":hurts,
            "years_removal_improves":helps,
            "status_hint":"CONTRIBUTES" if delta>0 and hurts>=3 else ("HURTS_COUNCIL" if delta<0 and helps>=3 else "MIXED_OR_REDUNDANT")
        })
    write_csv(out/"council-contribution.csv",contrib)

    pair_g=sum_group(pairs,"pair")
    pair_rows=[]
    corr_by=defaultdict(list); jac_by=defaultdict(list)
    for pair,c in sorted(pair_g.items()):
        n=num(c.get("races"))
        cr=num(c.get("rank_corr_sum"))/n if n else None
        ja=num(c.get("top3_jaccard_sum"))/n if n else None
        pair_rows.append({"pair":pair,"races":int(n),"mean_rank_correlation":cr,"mean_top3_jaccard":ja})
        a1,a2=pair.split("|",1)
        corr_by[a1].append(cr); corr_by[a2].append(cr); jac_by[a1].append(ja); jac_by[a2].append(ja)
    write_csv(out/"pairwise-redundancy.csv",pair_rows)

    leader={r["participant"]:r for r in leaderboard}
    contrib_map={r["expert"]:r for r in contrib}
    king=[]
    for e in EXPERT_ALIASES:
        r=leader[e]; c=contrib_map[e]
        king.append({
            "expert":e,
            "winner_top3_capture_pct":r["winner_top3_capture_pct"],
            "top1_top3_pct":r["top1_top3_pct"],
            "winner_top6_capture_pct":r["winner_top6_capture_pct"],
            "nonmarket_top3_actual_top3_pct":r["nonmarket_top3_actual_top3_pct"],
            "years_beat_market_winner_top3":r["years_beat_market_winner_top3"],
            "mean_rank_corr_with_other_kings":statistics.mean(corr_by[e]) if corr_by[e] else None,
            "mean_top3_jaccard_with_other_kings":statistics.mean(jac_by[e]) if jac_by[e] else None,
            "removal_cost_pp":c["removal_cost_pp"],
            "years_removal_hurts":c["years_removal_hurts"],
            "years_removal_improves":c["years_removal_improves"],
            "status_hint":c["status_hint"],
        })
    king.sort(key=lambda r:(-(r["removal_cost_pp"] or -999),-(r["nonmarket_top3_actual_top3_pct"] or -1)))
    write_csv(out/"king-audit.csv",king)

    rg=sum_group(rescues,"candidate")
    outsider=[]
    for cand,c in sorted(rg.items()):
        def pctn(n,d): return 100*n/d if d else None
        participant=leader.get("OUTSIDER:"+cand,{})
        outsider.append({
            "candidate":cand,
            "winner_top3_capture_pct":participant.get("winner_top3_capture_pct"),
            "top1_top3_pct":participant.get("top1_top3_pct"),
            "nonmarket_top3_actual_top3_pct":participant.get("nonmarket_top3_actual_top3_pct"),
            "council_top3_miss_races":int(num(c.get("council_top3_miss_n"))),
            "council_top3_rescue_pct":pctn(num(c.get("council_top3_miss_rescued")),num(c.get("council_top3_miss_n"))),
            "council_top6_miss_races":int(num(c.get("council_top6_miss_n"))),
            "council_top6_rescue_pct":pctn(num(c.get("council_top6_miss_rescued")),num(c.get("council_top6_miss_n"))),
            "council_market_both_top3_miss_races":int(num(c.get("council_market_top3_both_miss_n"))),
            "both_miss_rescue_pct":pctn(num(c.get("council_market_both_miss_rescued")),num(c.get("council_market_top3_both_miss_n"))),
            "novel_actual_top3_pct":pctn(num(c.get("novel_actual_top3")),num(c.get("novel_selected"))),
            "novel_winner_pct":pctn(num(c.get("novel_winner")),num(c.get("novel_selected"))),
        })
    outsider.sort(key=lambda r:(-(r["both_miss_rescue_pct"] or -1),-(r["council_top6_rescue_pct"] or -1),-(r["nonmarket_top3_actual_top3_pct"] or -1)))
    write_csv(out/"outsider-qualification.csv",outsider)

    summary={
        "contract":"L1_ROYAL_ARENA_V1","years":years,
        "purpose":"Re-audit all Seven Kings against final WIN market and saved tie-safe Outsider ballots.",
        "rules":{
            "seven_kings":"full-field tie-safe ranks; one-at-a-time removal test",
            "market":"final WIN market baseline and eighth-member council test; market-missing races are excluded only from market comparisons",
            "outsiders":"saved tie-safe Top3 ballots only in stage 1; no fake full-field rank",
            "next_stage":"recompute full-field ranks only for Outsider finalists after qualification"
        },
        "files":["leaderboard.csv","king-audit.csv","council-contribution.csv","pairwise-redundancy.csv","outsider-qualification.csv"],
        "2026_locked":True,"paid_compute":False,"artifact_cache":False,"promotion":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== KING AUDIT ====="); print((out/"king-audit.csv").read_text(encoding="utf-8"))
    print("===== OUTSIDER QUALIFICATION TOP ====="); print("\n".join((out/"outsider-qualification.csv").read_text(encoding="utf-8").splitlines()[:15]))
    print("===== LEADERBOARD TOP ====="); print("\n".join((out/"leaderboard.csv").read_text(encoding="utf-8").splitlines()[:20]))
    print("L1_ROYAL_ARENA_V1_READY")

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    y=sub.add_parser("year")
    y.add_argument("--year",required=True,type=int); y.add_argument("--expert",action="append",required=True)
    y.add_argument("--ballots",required=True); y.add_argument("--backfill-root",required=True); y.add_argument("--out-dir",required=True)
    g=sub.add_parser("aggregate")
    g.add_argument("--years",default="2021,2022,2023,2024,2025"); g.add_argument("--root",required=True); g.add_argument("--out-dir",required=True)
    a=p.parse_args()
    cmd_year(a) if a.cmd=="year" else cmd_aggregate(a)

if __name__=="__main__":
    main()
