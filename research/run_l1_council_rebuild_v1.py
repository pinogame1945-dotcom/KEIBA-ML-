#!/usr/bin/env python3
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

EXPERTS=("core4","pedlegacy","condition","full","light","pedv1","condrc")
YEARS=(2021,2022,2023,2024,2025)
MARKET_SHARES=(0.0,0.125,0.25,0.5,1.0,2.0,4.0)
EXPECTED_RACES=3456

def opent(path,mode="rt"):
    return gzip.open(path,mode,encoding="utf-8") if str(path).endswith(".gz") else open(path,mode,encoding="utf-8")

def finite(v):
    try:
        if isinstance(v,str): v=v.replace(",","").strip()
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError,ValueError): return None

def hno(v):
    try: return int(float(v))
    except (TypeError,ValueError): return 10**9

def load_expert(path):
    races=defaultdict(list)
    with opent(path) as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("contract")!="L1_TO_L2_OUTPUT_CONTRACT_V1":
                raise SystemExit(f"bad score contract: {path}")
            rid=str(r.get("race_id") or ""); hid=str(r.get("horse_id") or "")
            date=str(r.get("race_date") or "")[:10]
            rank=int(r.get("predicted_rank") or 0)
            if not rid or not hid or len(date)!=10 or rank<1:
                raise SystemExit(f"bad score row: {path} {rid} {hid}")
            races[rid].append((rank,hid,hno(r.get("horse_number")),date))
    if len(races)!=EXPECTED_RACES:
        raise SystemExit(f"score race coverage {path}: {len(races)} != {EXPECTED_RACES}")
    for rid,rows in races.items():
        rows.sort()
        if [x[0] for x in rows]!=list(range(1,len(rows)+1)):
            raise SystemExit(f"noncontiguous safe ranks: {path} {rid}")
        if len({x[1] for x in rows})!=len(rows):
            raise SystemExit(f"duplicate horse: {path} {rid}")
    return races

def load_ballots(path):
    out=defaultdict(lambda:defaultdict(lambda:{"points":0,"votes":0,"best":99}))
    candidates=set(); races=set()
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            rid=str(r.get("race_id") or ""); cand=str(r.get("candidate") or "")
            if not rid or not cand: continue
            candidates.add(cand); races.add(rid)
            for pos,pts in ((1,3),(2,2),(3,1)):
                hid=str(r.get(f"top{pos}_horse_id") or "")
                if not hid: continue
                z=out[rid][hid]
                z["points"]+=pts; z["votes"]+=1; z["best"]=min(z["best"],pos)
    if len(candidates)<10 or len(races)!=EXPECTED_RACES:
        raise SystemExit(f"ballot coverage drift candidates={len(candidates)} races={len(races)}")
    return out,sorted(candidates)

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

def market(orec,horse_to_no):
    raw=((orec.get("odds") or {}).get("1") or {})
    odds={}
    for k,v in raw.items():
        ks=str(k)
        if not ks.isascii() or not ks.isdigit(): continue
        tup=final_odds_tuple(v)
        if not tup: continue
        x=finite(tup[0])
        if x is not None and x>0: odds[int(ks)]=x
    if set(odds)!=set(horse_to_no.values()): return None,None
    inv={no:hid for hid,no in horse_to_no.items()}
    order=[inv[no] for no in sorted(odds,key=lambda no:(odds[no],no,inv[no]))]
    vals=list(odds.values())
    rank={hid:1+sum(1 for x in vals if x<odds[no]) for hid,no in horse_to_no.items()}
    return order,rank

def outcome(pack):
    winners=[]; podium=[]
    for r in pack.get("results") or []:
        if r.get("result_status")!="FINISHED": continue
        hid=str(r.get("horse_id") or "")
        try: pos=int(r.get("official_finish_position"))
        except (TypeError,ValueError): continue
        if not hid: continue
        if pos==1: winners.append(hid)
        if pos<=3: podium.append(hid)
    return sorted(set(winners)),sorted(set(podium))

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
    if year==2026: raise SystemExit("2026 sealed")
    specs={}
    for s in a.expert:
        name,path=s.split("=",1); specs[name]=path
    if set(specs)!=set(EXPERTS): raise SystemExit(f"expert set drift {sorted(specs)}")
    data={e:load_expert(specs[e]) for e in EXPERTS}
    sets=[set(x) for x in data.values()]
    if any(s!=sets[0] for s in sets[1:]): raise SystemExit("cross-expert race drift")
    ballots,candidates=load_ballots(a.ballots)

    base=data[EXPERTS[0]]
    bydate=defaultdict(list)
    for rid,rows in base.items():
        date=rows[0][3]
        if not date.startswith(str(year)+"-"): raise SystemExit(f"date drift {rid} {date}")
        bydate[date].append(rid)

    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    root=Path(a.backfill_root)
    market_missing=0; written=0
    with gzip.open(out,"wt",encoding="utf-8") as fo:
        for date,rids in sorted(bydate.items()):
            wanted=set(rids)
            day=load_day(root/"data"/"daily"/f"{date}.jsonl.gz",wanted)
            odds=load_odds_day(root/"data"/"odds"/"daily"/f"{date}.jsonl.gz",wanted)
            for rid in sorted(rids):
                pack=day.get(rid); orec=odds.get(rid)
                if pack is None or orec is None: raise SystemExit(f"missing backfill row {rid}")
                winners,podium=outcome(pack)
                if not winners or not podium: raise SystemExit(f"missing result {rid}")
                orders=[]
                horse_sets=[]
                no_map={}
                for e in EXPERTS:
                    rows=data[e][rid]
                    orders.append([x[1] for x in rows])
                    horse_sets.append({x[1] for x in rows})
                    if e==EXPERTS[0]: no_map={x[1]:x[2] for x in rows}
                if any(s!=horse_sets[0] for s in horse_sets[1:]):
                    raise SystemExit(f"horse coverage drift {rid}")
                horses=sorted(horse_sets[0])
                morder,mrank=market(orec,no_map)
                if morder is None:
                    market_missing+=1
                    mranks=None
                else:
                    mranks=[int(mrank[h]) for h in horses]
                outs=[]
                for hid,z in ballots[rid].items():
                    if hid in horse_sets[0]:
                        outs.append([hid,int(z["points"]),int(z["votes"]),int(z["best"])])
                outs.sort(key=lambda x:(-x[1],-x[2],x[3],x[0]))
                rec={
                    "year":year,"race_id":rid,"race_date":date,
                    "horses":horses,"orders":orders,
                    "market_order":morder,"market_ranks":mranks,
                    "winner":winners,"podium":podium,"outsider_support":outs,
                }
                fo.write(json.dumps(rec,ensure_ascii=False,separators=(",",":"))+"\n")
                written+=1
    if written!=EXPECTED_RACES: raise SystemExit(f"written {written} != {EXPECTED_RACES}")
    summary={
        "contract":"L1_COUNCIL_REBUILD_YEAR_V1","year":year,"races":written,
        "market_missing_races":market_missing,"experts":list(EXPERTS),
        "outsider_candidates":candidates,"2026_locked":True
    }
    Path(a.summary).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("L1_COUNCIL_REBUILD_YEAR_READY",json.dumps(summary,ensure_ascii=False,separators=(",",":")))

def read_races(path):
    out=[]
    with gzip.open(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def popcount(mask): return bin(mask).count("1")

def config_id(mask,share):
    if mask==0: return "MARKET_ONLY"
    return f"M{mask:03d}_S{share:g}"

def members(mask):
    return [EXPERTS[i] for i in range(7) if mask&(1<<i)]

def order_for(rec,mask,share):
    horses=rec["horses"]; n=len(horses)
    if mask==0:
        return list(rec["market_order"])
    idx={h:i for i,h in enumerate(horses)}
    rankvec=[]
    for order in rec["orders"]:
        m={h:i+1 for i,h in enumerate(order)}
        rankvec.append([m[h] for h in horses])
    mr=rec["market_ranks"]
    k=popcount(mask); mw=share*k
    scored=[]
    for j,h in enumerate(horses):
        rs=[rankvec[i][j] for i in range(7) if mask&(1<<i)]
        s=sum(rs)+(mw*mr[j] if mw else 0.0)
        s6=sum(r<=6 for r in rs)+(mw if mw and mr[j]<=6 else 0.0)
        s3=sum(r<=3 for r in rs)+(mw if mw and mr[j]<=3 else 0.0)
        s1=sum(r==1 for r in rs)+(mw if mw and mr[j]==1 else 0.0)
        scored.append((s,-s6,-s3,-s1,h))
    scored.sort()
    return [x[-1] for x in scored]

def metric_counts(rec,order):
    winners=set(rec["winner"]); podium=set(rec["podium"])
    t1=order[:1]; t3=order[:3]; t6=order[:6]
    return {
        "races":1,
        "top1_top3_hit":int(bool(set(t1)&podium)),
        "winner_top3_hit":int(bool(set(t3)&winners)),
        "winner_top6_hit":int(bool(set(t6)&winners)),
        "podium_hits_top3":len(set(t3)&podium),
        "podium_hits_top6":len(set(t6)&podium),
        "top3_selected":len(t3),
        "top6_selected":len(t6),
    }

def addc(dst,src):
    for k,v in src.items(): dst[k]+=v

def cmd_search(a):
    shard=int(a.shard); shards=int(a.shards)
    root=Path(a.root)
    yr={y:read_races(root/f"y{y}.jsonl.gz") for y in YEARS}
    for y,rows in yr.items():
        if len(rows)!=EXPECTED_RACES: raise SystemExit(f"year file drift {y} {len(rows)}")
    configs=[]
    if shard==0: configs.append((0,1.0))
    for mask in range(1,128):
        if mask%shards!=shard: continue
        for share in MARKET_SHARES: configs.append((mask,share))
    rows=[]
    for ci,(mask,share) in enumerate(configs,1):
        per={y:defaultdict(float) for y in YEARS}
        for y in YEARS:
            for rec in yr[y]:
                if rec["market_order"] is None: continue
                order=order_for(rec,mask,share)
                addc(per[y],metric_counts(rec,order))
        for y in YEARS:
            r=per[y]
            rows.append({
                "config_id":config_id(mask,share),"mask":mask,"market_share":share,
                "members":"|".join(members(mask)) if mask else "",
                "member_count":popcount(mask),"year":y,**dict(r)
            })
        if ci%25==0: print(f"SEARCH_PROGRESS shard={shard} {ci}/{len(configs)}",flush=True)
    write_csv(a.output,rows)
    print("L1_COUNCIL_SEARCH_SHARD_READY",json.dumps({"shard":shard,"configs":len(configs),"rows":len(rows)},separators=(",",":")))

def read_csv(path):
    with open(path,newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def nv(v): return float(v) if v not in (None,"") else 0.0

def sum_rows(rows):
    out=defaultdict(float)
    for r in rows:
        for k in ("races","top1_top3_hit","winner_top3_hit","winner_top6_hit","podium_hits_top3","podium_hits_top6","top3_selected","top6_selected"):
            out[k]+=nv(r.get(k))
    return out

def metrics(c):
    races=c["races"]
    return {
        "races":int(races),
        "top1_top3_pct":100*c["top1_top3_hit"]/races if races else None,
        "winner_top3_capture_pct":100*c["winner_top3_hit"]/races if races else None,
        "winner_top6_capture_pct":100*c["winner_top6_hit"]/races if races else None,
        "top3_podium_precision_pct":100*c["podium_hits_top3"]/c["top3_selected"] if c["top3_selected"] else None,
        "top6_podium_precision_pct":100*c["podium_hits_top6"]/c["top6_selected"] if c["top6_selected"] else None,
    }

def key_primary(m,meta):
    return (
        m["top3_podium_precision_pct"],
        m["winner_top3_capture_pct"],
        m["top1_top3_pct"],
        m["winner_top6_capture_pct"],
        -int(meta["member_count"]),
        -float(meta["market_share"]),
    )

def choose(table,train_years,mode):
    best=None
    bycfg=defaultdict(list)
    meta={}
    for r in table:
        cid=r["config_id"]; y=int(r["year"])
        if y in train_years: bycfg[cid].append(r)
        meta[cid]=r
    for cid,rs in bycfg.items():
        m=metrics(sum_rows(rs)); z=meta[cid]
        mask=int(z["mask"]); share=float(z["market_share"])
        if mode=="HYBRID" and mask==0: continue
        if mode=="PURE_L1" and (mask==0 or share!=0.0): continue
        item=(key_primary(m,z),cid,m,z)
        if best is None or item[0]>best[0]: best=item
    if best is None: raise SystemExit(f"no config mode={mode}")
    return best[1],best[2],best[3]

def config_lookup(table,cid,year):
    for r in table:
        if r["config_id"]==cid and int(r["year"])==year: return r
    raise KeyError((cid,year))

def base_meta(row):
    return {"mask":int(row["mask"]),"share":float(row["market_share"]),"members":row["members"],"member_count":int(row["member_count"])}

def jaccard(a,b):
    a=set(a); b=set(b); u=a|b
    return len(a&b)/len(u) if u else 1.0

RESCUE_POINTS=(6,9,12,15,18)
RESCUE_VOTES=(2,3,4)
RESCUE_AGREE=("ANY","LOW","HIGH","VERY_HIGH")
RESCUE_MARKET_MIN=(0,4,6)

def agree_ok(mode,j):
    if mode=="ANY": return True
    if mode=="LOW": return j<=0.5
    if mode=="HIGH": return j>=0.5
    if mode=="VERY_HIGH": return j>=0.8
    return False

def best_outsider(rec,base):
    b6=set(base[:6])
    mr={h:r for h,r in zip(rec["horses"],rec["market_ranks"])}
    opts=[]
    for hid,pts,votes,best in rec["outsider_support"]:
        if hid in b6: continue
        opts.append((-int(pts),-int(votes),int(best),mr.get(hid,999),hid))
    if not opts: return None
    opts.sort()
    _,_,best,mrank,hid=opts[0]
    row=next(x for x in rec["outsider_support"] if x[0]==hid)
    return {"hid":hid,"points":int(row[1]),"votes":int(row[2]),"best":int(row[3]),"market_rank":int(mrank)}

def apply_rescue(base,cand,action):
    if cand is None: return list(base)
    out=list(base)
    hid=cand["hid"]
    if hid in out: out.remove(hid)
    if action=="TOP6_REPLACE":
        return out[:5]+[hid]+out[5:]
    if action=="TOP3_REPLACE":
        return out[:2]+[hid]+out[2:]
    raise ValueError(action)

def rescue_rule_key(rule,action):
    p,v,ag,mm=rule
    return f"{action}:P{p}:V{v}:A{ag}:M{mm}"

def rescue_eval(rows,base_mask,base_share,rule,action):
    pmin,vmin,ag,mm=rule
    c=defaultdict(float); fired=0
    for rec in rows:
        if rec["market_order"] is None: continue
        base=order_for(rec,base_mask,base_share)
        cand=best_outsider(rec,base)
        fire=False
        if cand is not None:
            jj=jaccard(base[:3],rec["market_order"][:3])
            fire=(cand["points"]>=pmin and cand["votes"]>=vmin and cand["market_rank"]>=mm and agree_ok(ag,jj))
        order=apply_rescue(base,cand,action) if fire else base
        addc(c,metric_counts(rec,order))
        fired+=int(fire)
    c["fired"]=fired
    return c

def rescue_choose(train_rows,base_mask,base_share,action):
    basec=defaultdict(float)
    for rec in train_rows:
        if rec["market_order"] is None: continue
        addc(basec,metric_counts(rec,order_for(rec,base_mask,base_share)))
    bm=metrics(basec)
    best=("NONE",bm,None,basec)
    for p in RESCUE_POINTS:
        for v in RESCUE_VOTES:
            for ag in RESCUE_AGREE:
                for mm in RESCUE_MARKET_MIN:
                    rule=(p,v,ag,mm)
                    c=rescue_eval(train_rows,base_mask,base_share,rule,action)
                    m=metrics(c)
                    if action=="TOP6_REPLACE":
                        key=(m["top6_podium_precision_pct"],m["winner_top6_capture_pct"],-c["fired"])
                        bkey=(best[1]["top6_podium_precision_pct"],best[1]["winner_top6_capture_pct"],-(best[3].get("fired",0)))
                    else:
                        key=(m["top3_podium_precision_pct"],m["winner_top3_capture_pct"],m["top1_top3_pct"],-c["fired"])
                        bkey=(best[1]["top3_podium_precision_pct"],best[1]["winner_top3_capture_pct"],best[1]["top1_top3_pct"],-(best[3].get("fired",0)))
                    if key>bkey: best=(rescue_rule_key(rule,action),m,rule,c)
    return best,metrics(basec)

def cmd_final(a):
    table=[]
    for p in a.shard_csv: table+=read_csv(p)
    cfgs={r["config_id"] for r in table}
    if len(cfgs)<850: raise SystemExit(f"config coverage too low {len(cfgs)}")
    root=Path(a.race_root)
    races={y:read_races(root/f"y{y}.jsonl.gz") for y in YEARS}
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    pooled=[]
    bycfg=defaultdict(list); meta={}
    for r in table:
        bycfg[r["config_id"]].append(r); meta[r["config_id"]]=r
    for cid,rs in bycfg.items():
        m=metrics(sum_rows(rs)); z=meta[cid]
        pooled.append({
            "config_id":cid,"mask":z["mask"],"market_share":z["market_share"],
            "members":z["members"],"member_count":z["member_count"],**m
        })
    pooled.sort(key=lambda r:(-r["top3_podium_precision_pct"],-r["winner_top3_capture_pct"],-r["top1_top3_pct"]))
    write_csv(out/"pooled-leaderboard.csv",pooled)

    oos=[]
    selected={}
    for test in (2022,2023,2024,2025):
        train=tuple(y for y in YEARS if y<test)
        for mode in ("UNCONSTRAINED","HYBRID","PURE_L1"):
            cid,tm,z=choose(table,train,mode)
            tr=config_lookup(table,cid,test)
            testm=metrics(sum_rows([tr]))
            row={
                "test_year":test,"train_years":"|".join(map(str,train)),"mode":mode,
                "config_id":cid,"members":z["members"],"member_count":z["member_count"],
                "market_share":z["market_share"],
                **{f"train_{k}":v for k,v in tm.items()},
                **{f"test_{k}":v for k,v in testm.items()},
            }
            oos.append(row); selected[(test,mode)]=row
    write_csv(out/"strict-oos-champions.csv",oos)

    freq=[]
    for mode in ("HYBRID","PURE_L1"):
        counts=defaultdict(int); shares=[]
        for test in (2022,2023,2024,2025):
            r=selected[(test,mode)]
            for e in (r["members"].split("|") if r["members"] else []): counts[e]+=1
            shares.append(float(r["market_share"]))
        for e in EXPERTS:
            freq.append({"mode":mode,"expert":e,"selected_folds":counts[e]})
        freq.append({"mode":mode,"expert":"__MARKET_SHARE_MEAN__","selected_folds":statistics.mean(shares)})
    write_csv(out/"selection-frequency.csv",freq)

    rescue_rows=[]
    for test in (2022,2023,2024,2025):
        base=selected[(test,"HYBRID")]
        mask=int(next(r["mask"] for r in table if r["config_id"]==base["config_id"]))
        share=float(base["market_share"])
        train_rows=[rec for y in YEARS if y<test for rec in races[y]]
        test_rows=races[test]
        for action in ("TOP6_REPLACE","TOP3_REPLACE"):
            best,bm=rescue_choose(train_rows,mask,share,action)
            rule_name,trainm,rule,trainc=best
            if rule is None:
                testc=defaultdict(float)
                for rec in test_rows:
                    if rec["market_order"] is None: continue
                    addc(testc,metric_counts(rec,order_for(rec,mask,share)))
            else:
                testc=rescue_eval(test_rows,mask,share,rule,action)
            testm=metrics(testc)
            base_test=metrics(sum_rows([config_lookup(table,base["config_id"],test)]))
            rescue_rows.append({
                "test_year":test,"action":action,"base_config":base["config_id"],
                "base_members":base["members"],"base_market_share":base["market_share"],
                "selected_rule":rule_name,"train_fired":int(trainc.get("fired",0)),
                "test_fired":int(testc.get("fired",0)),
                "train_top3_podium_precision_pct":trainm["top3_podium_precision_pct"],
                "train_winner_top3_capture_pct":trainm["winner_top3_capture_pct"],
                "train_top6_podium_precision_pct":trainm["top6_podium_precision_pct"],
                "train_winner_top6_capture_pct":trainm["winner_top6_capture_pct"],
                "base_test_top3_podium_precision_pct":base_test["top3_podium_precision_pct"],
                "rescue_test_top3_podium_precision_pct":testm["top3_podium_precision_pct"],
                "delta_test_top3_precision_pp":testm["top3_podium_precision_pct"]-base_test["top3_podium_precision_pct"],
                "base_test_winner_top3_capture_pct":base_test["winner_top3_capture_pct"],
                "rescue_test_winner_top3_capture_pct":testm["winner_top3_capture_pct"],
                "delta_test_winner_top3_pp":testm["winner_top3_capture_pct"]-base_test["winner_top3_capture_pct"],
                "base_test_top6_podium_precision_pct":base_test["top6_podium_precision_pct"],
                "rescue_test_top6_podium_precision_pct":testm["top6_podium_precision_pct"],
                "delta_test_top6_precision_pp":testm["top6_podium_precision_pct"]-base_test["top6_podium_precision_pct"],
                "base_test_winner_top6_capture_pct":base_test["winner_top6_capture_pct"],
                "rescue_test_winner_top6_capture_pct":testm["winner_top6_capture_pct"],
                "delta_test_winner_top6_pp":testm["winner_top6_capture_pct"]-base_test["winner_top6_capture_pct"],
            })
    write_csv(out/"rescue-strict-oos.csv",rescue_rows)

    summary={
        "contract":"L1_COUNCIL_REBUILD_V1",
        "search_space":{
            "pure_l1_subsets":127,
            "market_share_grid":list(MARKET_SHARES),
            "market_share_definition":"market vote weight = selected L1 expert count * market_share",
            "market_only_included":True,
        },
        "selection":{
            "walk_forward":"expanding prior years only; test years 2022-2025",
            "primary":"Top3 podium precision",
            "tie_breaks":["winner Top3 capture","Top1 podium rate","winner Top6 capture","fewer experts","lower market share"],
            "modes":["UNCONSTRAINED","HYBRID","PURE_L1"],
        },
        "outsider_rescue":{
            "source":"saved tie-safe 13-Outsider Top3 ballots",
            "candidate":"highest aggregate 3-2-1 point horse outside base Top6",
            "features":["aggregate points","vote count","base-vs-market Top3 Jaccard","candidate market rank"],
            "actions":["replace base rank6","replace base rank3"],
            "rule_selection":"prior years only; NONE allowed",
        },
        "2026_locked":True,"promotion":False,"paid_compute":False,"artifact_cache":False
    }
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("===== STRICT OOS CHAMPIONS ====="); print((out/"strict-oos-champions.csv").read_text())
    print("===== SELECTION FREQUENCY ====="); print((out/"selection-frequency.csv").read_text())
    print("===== RESCUE OOS ====="); print((out/"rescue-strict-oos.csv").read_text())
    print("L1_COUNCIL_REBUILD_V1_READY")

def main():
    p=argparse.ArgumentParser()
    sp=p.add_subparsers(dest="cmd",required=True)
    y=sp.add_parser("year")
    y.add_argument("--year",type=int,required=True); y.add_argument("--expert",action="append",required=True)
    y.add_argument("--ballots",required=True); y.add_argument("--backfill-root",required=True)
    y.add_argument("--output",required=True); y.add_argument("--summary",required=True)
    s=sp.add_parser("search")
    s.add_argument("--root",required=True); s.add_argument("--shard",type=int,required=True); s.add_argument("--shards",type=int,default=4); s.add_argument("--output",required=True)
    f=sp.add_parser("final")
    f.add_argument("--shard-csv",action="append",required=True); f.add_argument("--race-root",required=True); f.add_argument("--out-dir",required=True)
    a=p.parse_args()
    if a.cmd=="year": cmd_year(a)
    elif a.cmd=="search": cmd_search(a)
    else: cmd_final(a)

if __name__=="__main__": main()
