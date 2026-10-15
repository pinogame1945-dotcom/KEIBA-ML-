#!/usr/bin/env python3
"""Fixed-152-template, exact 20-band ticket-pruning experiment, OOS 2024/2025.

This script NEVER chooses new templates or new odds bands. For every physical
ticket of ALL 152 original EV-anchor template generators, it calculates
ex-ante feature-only score and retains/discards the ticket. The generated
tickets and original per-year hits and refunds must exactly match the archived
3040-line baseline or the run FAILS. No training on same-year returns.

Frozen comparison groups are OBSERVATION ONLY: 245 pooled-2023-25 profitable
odds lines vs the remaining 2795 (nonpositive); model NEVER gets that label.
Past FINAL odds are only known after betting has closed. Consequently OOS refers
to labels only; live bet-time deployability is NOT established.
"""
import argparse,csv,gzip,itertools,json,math
from bisect import bisect_right
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
import pandas as pd
import run_l2_all_bet_types_ticket_odds_gate_v1 as lab
import run_l2_value_anchor_bet_frontier_v1 as prior
import run_l2_ai_market_rank_betfrontier_v1 as prior_rank
from build_l2_market_helpers_v1 import load_day,load_odds_day,horse_number_map
from run_all_race_flow_ticket_audit_v1 import YEARS,RACES_YEAR,TOTAL_RACES,assert_ok

BINS=lab.FLOORS
MODES=("PROB","MARKET","LEARNED","HYBRID")
KEEP=(90,75,50,25,10)
POLICIES=("BASE",)+tuple(m+"_"+str(k) for m in MODES for k in KEEP)
GROUPS=("ALL_3040","ORIGINAL_PLUS_245","ORIGINAL_NONPLUS_2795")
YEAR_TEXT=("2023","2024","2025")
FAMILIES=lab.FAMILIES
FIELDS=("races_bought","tickets","hit_tickets","hit_races","return_yen")
ZERO=(0,0,0,0,0)

def bin_name(lo):
    i=BINS.index(lo)
    return f"{int(lo) if float(lo).is_integer() else lo}-{int(BINS[i+1]) if float(BINS[i+1]).is_integer() else BINS[i+1]}" if i<len(BINS)-1 else "5000+"

def load_lines(path):
    with open(path,encoding="utf-8",newline="") as f:rr=list(csv.DictReader(f))
    assert len(rr)==3040
    by={};is_positive={}
    for r in rr:
        _,order,k,fam=r["買い方ID"].split("|")
        key=(order,int(k),fam,r["馬券オッズ帯"])
        assert key not in by,key
        by[key]=r
        is_positive[key]=int(r["2023-2025_損益円"])>0
    assert sum(is_positive.values())==245
    return by,is_positive

def partner_p(candidates,n2horse,probs):
    pp=[max(.000001,probs[n2horse[n]]) for n in candidates]
    return sum(pp)/len(pp) if pp else 0.0

def create_cards(g,pack,market,rid,axis,race):
    no=horse_number_map(pack)
    n2horse={v:k for k,v in no.items()}
    ranked={"AI_PROB":lab.sort_horses(g,"AI_PROB"),
       "MARKET":lab.sort_horses(g,"MARKET"),
       "AI_RAW":g.sort_values(["distill_ai_rank","market_rank","_horse_id"],
               ascending=[True,True,True],kind="mergesort")["_horse_id"].astype(str).tolist()}
    assert_ok(ranked["AI_RAW"][0]==str(race["AI_RAW_TOP_horse"]),"raw model rank drift")
    assert_ok(axis==str(race["VALUE_EV_horse"]),"EV horse changed")
    an=no[axis]
    pos={kind:{no[h]:i+1 for i,h in enumerate(seq)} for kind,seq in ranked.items()}
    probs=dict(zip(g["_horse_id"].astype(str),g["P__ANCHOR_FULL"].astype(float)))
    pays=prior.payouts(pack,rid)
    price_cache={}; card_cache={}
    # An original recipe. No odds threshold, no rank swap, no added bets.
    for order,k,fam in [("SINGLE",0,"WIN"),("SINGLE",0,"PLACE")]+[
        (order,k,fam) for order in lab.ORDERS for k in (2,3,5,7,10) for fam in FAMILIES]:
        if order=="SINGLE":tickets={(fam,(an,))}
        else:
            others=[no[h] for h in ranked[order] if h!=axis][:k]
            tickets=prior.pair_templates(an,others)[fam]
        for ticket in tickets:
            if ticket not in price_cache:
                price=lab.ticket_price(ticket,market)
                ret=round(float(pays.get(ticket,0)))
                price_cache[ticket]=(price,ret)
            price,ret=price_cache[ticket]
            if price is None:continue
            assert price>0 and math.isfinite(price)
            bz=bisect_right(BINS,price)-1
            band=bin_name(BINS[bz])
            if ticket not in card_cache:
                kind,seq=ticket
                partners=[n for n in seq if n!=an]
                assert len(partners)==len(seq)-1
                orientation=(seq.index(an)+1 if kind in ("EXACTA","TRIFECTA") else 0)
                mkt=mean(pos["MARKET"][n] for n in partners) if partners else pos["MARKET"][an]
                ai=mean(pos["AI_PROB"][n] for n in partners) if partners else pos["AI_PROB"][an]
                raw=mean(pos["AI_RAW"][n] for n in partners) if partners else pos["AI_RAW"][an]
                p=partner_p(partners,n2horse,probs) if partners else probs[axis]
                p=max(.00001,min(.99999,float(p)))
                # Discrete historical features for smoothing plus continuous
                # pre-race horse assessment used to prune individual tickets.
                b_m=min(5,int((mkt-1)//3))
                b_a=min(5,int((ai-1)//3))
                feat=(kind,bz,orientation,b_m,b_a)
                prob_score=math.log(p)+.08*math.log(max(.00001,probs[axis]))
                market_score=-math.log1p(mkt)-.15*math.log1p(pos["MARKET"][an])
                card_cache[ticket]=(float(price),int(ret),feat,float(prob_score),
                                    float(market_score),float((mkt-raw)/(mkt+raw+1)))
            # A physical ticket may appear in several lines, by design.
            yield (order,k,fam,band),ticket,card_cache[ticket]

def learning_keys(feat):
    kind,bz,o,m,a=feat
    return (("M",kind,bz,o,m,a),
            ("A",kind,bz,o,a),
            ("K",kind,bz,o,m),
            ("P",kind,bz,o),
            ("F",kind,bz))

def add_learning(stat,card):
    price,ret,feat,pr,market,dis=card
    expected=min(.95,.80/price)
    for k in learning_keys(feat):
        s=stat[k];s[0]+=1;s[1]+=int(ret>0);s[2]+=expected

def calc_scores(card,stat):
    price,ret,feat,p,m,d=card
    keys=learning_keys(feat)
    def lr(i,pseudo):
        n,hit,ex=stat.get(keys[i],ZERO[:3])
        return math.log((hit+pseudo)/(ex+pseudo))
    # Hierarchical shrinkage, retaining meaningful rare-bin uncertainty.
    empirical=.25*lr(0,8)+.25*lr(1,12)+.25*lr(2,12)+.15*lr(3,25)+.10*lr(4,40)
    return {"PROB":p,"MARKET":m,
            "LEARNED":empirical+.12*p+.04*d,
            "HYBRID":empirical*.7+p*.20+m*.10}

def threshold(scores,keep):
    if not scores:return float("-inf")
    vals=sorted(scores)
    z=int(len(vals)*(100-keep)/100)
    return vals[min(len(vals)-1,z)]

def create_thresholds(history,stat):
    # Thresholds are computed from PREVIOUS training-year tickets only.
    # Score feature distributions of test years are never inspected.
    dist=defaultdict(list);whole=defaultdict(list)
    for card in history:
        group=(card[2][0],card[2][1])
        scores=calc_scores(card,stat)
        for mode,v in scores.items():
            dist[mode,group].append(v)
            whole[mode,group[0]].append(v)
    limits={}
    for (mode,group),arr in dist.items():
        src=arr if len(arr)>=100 else whole[mode,group[0]]
        for keep in KEEP:
            limits[mode,keep,group]=threshold(src,keep)
    for (mode,kind),arr in whole.items():
        for keep in KEEP:
            limits[mode,keep,kind]=threshold(arr,keep)
    return limits

def mask(card,stat,thresholds):
    s=calc_scores(card,stat)
    group=(card[2][0],card[2][1])
    flags={"BASE":True}
    for m in MODES:
        for k in KEEP:
            th=thresholds.get((m,k,group),thresholds.get((m,k,group[0]),float("-inf")))
            flags[m+"_"+str(k)]=(s[m]>=th)
    return flags

def bump(c,values):
    n,h,ret,hit_r=values
    c["tickets"]+=n;c["hit_tickets"]+=h;c["return_yen"]+=ret
    c["races_bought"]+=int(n>0);c["hit_races"]+=int(hit_r>0)

def run_year(year,pred,picks,races,root,stat,limits,history):
    # At year boundary 'stat' and 'limits' are snapshots exclusively learned
    # from prior years. This function never mutates either.
    gyear=pred[pred.year.eq(year)].groupby("_race_id",sort=False)
    ry=races[races.year.eq(year)]
    assert len(ry)==RACES_YEAR
    rows=ry.set_index("race_id")
    dates=ry.groupby("race_date")["race_id"].apply(list)
    line=defaultdict(Counter);union=defaultdict(Counter)
    learn=defaultdict(lambda:[0,0,0.0])
    future=[]
    seen_races=0;scored_card_count=0
    for day,ids in dates.items():
        packs=load_day(root/"data"/"daily"/f"{day}.jsonl.gz",set(ids))
        odds=load_odds_day(root/"data"/"odds"/"daily"/f"{day}.jsonl.gz",set(ids))
        for rid in sorted(ids):
            seen_races+=1
            pack=packs.get(rid);market=odds.get(rid)
            assert pack is not None and market is not None,("race input missing",rid)
            assert market.get("odds_kind")=="FINAL" and market.get("source_status")=="result",rid
            info=rows.loc[rid]
            allcards={}
            each_line=defaultdict(dict)
            for tid,ticket,card in create_cards(gyear.get_group(rid),
                    pack,market,rid,picks["ANCHOR_FULL"][rid],info):
                assert tid in history,("new line",tid)
                allcards[ticket]=card
                if ticket in each_line[tid]:
                    raise ValueError("duplicate physical ticket within original template "+str(tid))
                each_line[tid][ticket]=card
            scored_card_count+=len(allcards)
            votes={ticket:(mask(c,stat,limits) if year>2023 else {"BASE":True})
                   for ticket,c in allcards.items()}
            all_policies=("BASE",) if year==2023 else POLICIES
            for tid,items in each_line.items():
                for policy in all_policies:
                    cards=[c for ticket,c in items.items() if votes[ticket].get(policy,False)]
                    n=len(cards);h=sum(c[1]>0 for c in cards);ret=sum(c[1] for c in cards)
                    bump(line[(policy,tid)],(n,h,ret,h))
            for group in GROUPS:
                tickets=set()
                for tid,items in each_line.items():
                    if group=="ALL_3040" or (group=="ORIGINAL_PLUS_245")==history[tid]["plus"]:
                        tickets.update(items)
                for policy in all_policies:
                    bought=[allcards[t] for t in tickets if votes[t].get(policy,False)]
                    h=sum(c[1]>0 for c in bought)
                    bump(union[(policy,group)],(len(bought),h,sum(c[1] for c in bought),h))
            # No lookahead: training data is returned and merged only AFTER
            # the entire year has finished scoring with prior-year statistics.
            for card in allcards.values():
                add_learning(learn,card)
                future.append(card)
            if seen_races%1200==0:print("PRUNE_PROGRESS",year,seen_races,
                                        "unique_priced",scored_card_count,flush=True)
    assert seen_races==RACES_YEAR
    print("YEAR_DONE",year,"unique_physical",scored_card_count,
          "original_lines_bought",len({tid for p,tid in line if p=="BASE"}),flush=True)
    return line,union,learn,future

def merge(a,b):
    for k,v in b.items():
        s=a[k]
        for j in range(3):s[j]+=v[j]

def roi(c):
    n=c["tickets"];ret=c["return_yen"]
    return (100*ret/n/100) if n else None

def fields(prefix,c):
    n=c["tickets"];hit=c["hit_tickets"];races=c["races_bought"]
    return {prefix+"races":races,prefix+"tickets":n,prefix+"hits":hit,
            prefix+"misses":n-hit,prefix+"hit_races":c["hit_races"],
            prefix+"stake_yen":n*100,prefix+"return_yen":c["return_yen"],
            prefix+"profit_yen":c["return_yen"]-n*100,
            prefix+"roi_pct":(100*c["return_yen"]/(100*n) if n else None)}

def write_csv(path,records):
    assert records
    with (gzip.open(path,"wt",encoding="utf-8",newline="") if str(path).endswith(".gz")
          else open(path,"w",encoding="utf-8",newline="")) as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)

def main():
    ap=argparse.ArgumentParser()
    for p in ("predictions","picks","racewise","backfill-root","bands","old-union","out-dir"):
        ap.add_argument("--"+p,required=True)
    args=ap.parse_args()
    dest=Path(args.out_dir);dest.mkdir(parents=True,exist_ok=True)
    originals,plus={},{}
    with open(args.bands,encoding="utf-8",newline="") as f:
        old=list(csv.DictReader(f))
    assert len(old)==3040
    for r in old:
        _,order,k,fam=r["買い方ID"].split("|")
        tid=(order,int(k),fam,r["馬券オッズ帯"])
        assert tid not in originals
        originals[tid]=r
    assert sum(int(r["2023-2025_損益円"])>0 for r in originals.values())==245
    # It is crucial to keep all 3,040 original lines: even those with no
    # previous purchase/old red and without hits must remain in ledger.
    history={tid:{"plus":int(row["2023-2025_損益円"])>0} for tid,row in originals.items()}
    pred,picks=prior_rank.load_predictions(args.predictions,args.picks)
    races=pd.read_csv(args.racewise,compression="infer",dtype={
       "race_id":str,"VALUE_EV_horse":str,"AI_RAW_TOP_horse":str})
    races["year"]=pd.to_numeric(races.year,errors="raise").astype(int)
    assert len(races)==TOTAL_RACES and races.race_id.is_unique and set(races.year)==set(YEARS)
    stats=defaultdict(lambda:[0,0,0.0]);feature_history=[]
    yearly={}
    for y in YEARS:
        lim=create_thresholds(feature_history,stats) if y!=2023 else {}
        line,union,learn,future=run_year(y,pred,picks,races,
                    Path(args.backfill_root),stats,lim,history)
        yearly[str(y)]=(line,union)
        # Validate ALL 3040 original template/odds-band lines before filtering.
        for tid,old in originals.items():
            c=line["BASE",tid]
            for orig,observed in (("購入点数",c["tickets"]),
                                  ("的中点数",c["hit_tickets"]),
                                  ("払戻円",c["return_yen"])):
                assert observed==int(old[f"{y}_{orig}"]),(
                    "BASELINE_DRIFT",y,tid,orig,observed,old[f"{y}_{orig}"])
        merge(stats,learn)
        feature_history.extend(future)
        print("ALL_3040_ORIGINAL_BASELINES_EXACTLY_REPRODUCED",y,flush=True)
    # Assert preexisting profitable 245 union: exact race, tickets, hits, refund.
    with open(args.old_union,encoding="utf-8",newline="") as f:
        prior_u={r["year"]:r for r in csv.DictReader(f) if r["group"]=="THREE_YEAR_TOTAL_PLUS"}
    for y in YEAR_TEXT:
        c=yearly[y][1]["BASE","ORIGINAL_PLUS_245"]
        o=prior_u[y]
        for name,metric in (("races_bought","races_bought"),("unique_tickets","tickets"),
                             ("hit_tickets","hit_tickets"),("return_yen","return_yen")):
            assert int(o[name])==c[metric],("OLD_245_UNION_DRIFT",y,metric,int(o[name]),c[metric])
    print("OLD_BLACK_245_UNION_EXACTLY_REPRODUCED",flush=True)
    detail=[];union_rows=[]
    years=("2024","2025","2024-2025")
    for yr in years:
        for tid,orig in originals.items():
            if yr=="2024-2025":
                base=yearly["2024"][0]["BASE",tid]+yearly["2025"][0]["BASE",tid]
            else:base=yearly[yr][0]["BASE",tid]
            for policy in POLICIES:
                if yr=="2024-2025":post=yearly["2024"][0][policy,tid]+yearly["2025"][0][policy,tid]
                else:post=yearly[yr][0][policy,tid]
                assert post["tickets"]<=base["tickets"]
                assert post["hit_tickets"]<=base["hit_tickets"]
                row={"year":yr,"original_3yr_black":history[tid]["plus"],
                    "order":tid[0],"k":tid[1],"family":tid[2],"odds_band":tid[3],
                    "policy":policy,
                    **fields("before_",base),**fields("after_",post),
                    "saved_losing_tickets":(base["tickets"]-base["hit_tickets"])
                                         -(post["tickets"]-post["hit_tickets"]),
                    "lost_winning_tickets":base["hit_tickets"]-post["hit_tickets"],
                    "saved_stake_yen":100*(base["tickets"]-post["tickets"]),
                    "lost_winnings_yen":base["return_yen"]-post["return_yen"],
                    "profit_delta_yen":(post["return_yen"]-100*post["tickets"])
                                       -(base["return_yen"]-100*base["tickets"])}
                row["red_to_black"]=row["before_profit_yen"]<0 and row["after_profit_yen"]>0
                row["black_to_red"]=row["before_profit_yen"]>0 and row["after_profit_yen"]<0
                detail.append(row)
        for group in GROUPS:
            for policy in POLICIES:
                if yr=="2024-2025":
                    base=yearly["2024"][1]["BASE",group]+yearly["2025"][1]["BASE",group]
                    post=yearly["2024"][1][policy,group]+yearly["2025"][1][policy,group]
                    denom=6912
                else:
                    base=yearly[yr][1]["BASE",group];post=yearly[yr][1][policy,group]
                    denom=3456
                rr={"year":yr,"segment":group,"policy":policy,"available_races":denom,
                    **fields("before_",base),**fields("after_",post)}
                rr.update(before_coverage_pct=100*base["races_bought"]/denom,
                          after_coverage_pct=100*post["races_bought"]/denom,
                          saved_losing_tickets=(base["tickets"]-base["hit_tickets"])
                                           -(post["tickets"]-post["hit_tickets"]),
                          lost_winning_tickets=base["hit_tickets"]-post["hit_tickets"],
                          saved_stake_yen=(base["tickets"]-post["tickets"])*100,
                          lost_winnings_yen=base["return_yen"]-post["return_yen"],
                          profit_delta_yen=(post["return_yen"]-100*post["tickets"])
                                                -(base["return_yen"]-100*base["tickets"]))
                union_rows.append(rr)
    write_csv(dest/"all_3040_fixed_lines_before_after_2024_2025.csv.gz",detail)
    write_csv(dest/"deduped_original_245_and_all_tickets_union_before_after.csv",union_rows)
    focus=[r for r in detail if r["original_3yr_black"] and r["year"]=="2024-2025"]
    write_csv(dest/"all_245_original_black_lines_pruning_preservation.csv",focus)
    rest=[r for r in detail if not r["original_3yr_black"] and
          r["year"]=="2024-2025" and r["red_to_black"]]
    write_csv(dest/"red_lines_changed_to_black_2024_25.csv.gz",rest if rest else [
      {k:"" for k in detail[0]}])
    audit=[]
    for policy in POLICIES:
        x=[r for r in detail if r["year"]=="2024-2025" and r["policy"]==policy]
        y24=[r for r in detail if r["year"]=="2024" and r["policy"]==policy]
        y25=[r for r in detail if r["year"]=="2025" and r["policy"]==policy]
        audit.append({"policy":policy,"fixed_lines":len(x),
          "original_black_lines":sum(t["original_3yr_black"] for t in x),
          "original_black_went_red":sum(t["original_3yr_black"] and t["black_to_red"] for t in x),
          "red_went_black":sum((not t["original_3yr_black"]) and t["red_to_black"] for t in x),
          "red_went_black_both_individual_test_years":sum(
              (not t["original_3yr_black"]) and t["red_to_black"] and
              y24[i]["after_profit_yen"]>0 and y25[i]["after_profit_yen"]>0
              for i,t in enumerate(x)),
          "black_profit_delta_sum_not_dedup":sum(t["profit_delta_yen"] for t in x if t["original_3yr_black"]),
          "red_profit_delta_sum_not_dedup":sum(t["profit_delta_yen"] for t in x if not t["original_3yr_black"])})
    write_csv(dest/"red_to_black_preservation_audit_all_policies.csv",audit)
    chosen=sorted([u for u in union_rows if u["year"]=="2024-2025"
           and u["segment"]=="ORIGINAL_PLUS_245"],
           key=lambda x:x["after_profit_yen"],reverse=True)
    print("ALL_245_HISTORICALLY_BLACK_UNION_TEST",json.dumps(chosen,
                               ensure_ascii=False),flush=True)
    print("FULL_RESULT_COUNTS",json.dumps(audit,ensure_ascii=False),flush=True)
    (dest/"summary.json").write_text(json.dumps({
        "contract":"L2_FIXED_152_WAYS_20_BANDS_TICKET_PRUNING_WALK_FORWARD_V1",
        "scope":"3040 original line definitions unchanged; 245 hindsight-black and 2795 non-positive line segments unchanged; no new tickets",
        "train2023_test2024":True,"train2023_2024_test2025":True,
        "each_year_exact_reconstruction":True,"old245_union_exact_reconstruction":True,
        "years_eval":[2024,2025],"years_2023":"training only, left unpruned",
        "original_templates":152,"original_bands":20,"all_lines":3040,
        "observed_3yr_positive":245,
        "variants":list(POLICIES),"races_eval":6912,
        "hindsight_group_warning":"Original 245 plus label based on full 2023-2025 hindsight; only a descriptive group, NOT fed to model",
        "final_odds_warning":"Uses historical FINAL prices observed after betting closed; not live-tradeable proof",
        "model_selection_warning":"20 pruning variants all evaluated, no final winner chosen based on test set; apparent best is exploratory",
        "year2026_sealed":True,"cost":"free standard CPU, pinned BACKFILL, no purchased storage"
      },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FIXED_TICKET_PRUNING_COMPLETE",len(detail),len(union_rows),flush=True)

if __name__=="__main__":main()
