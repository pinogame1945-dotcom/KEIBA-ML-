#!/usr/bin/env python3
"""Race-ID-only replay of saved 7 king + 13 outsider research predictions.

No BACKFILL checkout, no live scrape, no model retraining. This replays
historical fold outputs and does not imply that FINAL fields were truly
available at bet time. No result, finish, odds or payouts are used.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from urllib.parse import urlparse,parse_qs

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE_PY=ROOT/"research"/"replay-archived-council.py"

def parse_id(raw):
    raw=str(raw or "").strip()
    if raw.isascii() and len(raw)==12 and raw.isdigit():return raw
    url=urlparse(raw)
    if url.scheme not in ("https","http") or url.hostname not in (
        "race.netkeiba.com","db.netkeiba.com"):
        raise ValueError("race ID or netkeiba URL required")
    candidates=parse_qs(url.query).get("race_id",[])
    if len(candidates)==1 and len(candidates[0])==12 and candidates[0].isascii() and candidates[0].isdigit():
        return candidates[0]
    pieces=[p for p in url.path.split("/") if p]
    if len(pieces)>=2 and pieces[-2]=="race" and len(pieces[-1])==12 and pieces[-1].isascii() and pieces[-1].isdigit():
        return pieces[-1]
    raise ValueError("netkeiba URL has no canonical 12-digit race ID")

def load_module():
    s=importlib.util.spec_from_file_location("archived_council_replay",ARCHIVE_PY)
    module=importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
    return module

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()

def render(race_id,council,ballots):
    module=load_module()
    base=module.replay(race_id,str(council),str(ballots))
    if base.get("outsider_status")!="VERIFIED_SAVED_BALLOTS":
        raise ValueError("all 13 outsider ballots must pass cross-source validation")
    horses=[]
    for row in base["horses"]:
        ranks=row["king_ranks"]
        votes=row["outsider_top3_from_each"]
        horses.append({
            "horse_id":row["horse_id"],
            "historical_7king_rank_sum_rank":row["pure_7king_rank_sum_comparator_rank"],
            "historical_7king_individual_ranks":ranks,
            "historical_13outsider_top3_votes":votes,
            "historical_outsider_points":row["outsider_points"],
            "historical_outsider_voters":row["outsider_votes"],
            "source_prediction_phase":"FINAL_RESEARCH_ARCHIVED"
        })
    assert len(horses)==len({h["horse_id"] for h in horses})
    return {
        "contract":"KEIBA_RACE_ID_ONLY_ARCHIVE_REPLAY_V1",
        "race_id":race_id,"race_date":base["race_date"],
        "status":"HISTORICAL_OOF_REPLAY_ONLY",
        "not_live_prediction":True,
        "result_or_payout_used":False,
        "final_market_odds_used":False,
        "original_l21_router_reconstructed":False,
        "model_families":{
            "seven_king":7,"outsider_top3_ballot":13,
            "scope":"archived individual ranks and Top3 votes only"
        },
        "runners":len(horses),
        "horses":horses,
        "source_checksums":{
            "research_council_archive_sha256":sha(council),
            "research_outsider_ballots_sha256":sha(ballots)
        },
        "source_qualification":"Stored walk-forward research outputs; historical FINAL archive features are not independently proven to be available at the actual pre-race decision time."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--race",required=True,help="12-digit canonical race ID or Netkeiba URL")
    p.add_argument("--output-dir",default="out/archived-race-only")
    p.add_argument("--council-file",help="Local saved yYEAR.jsonl.gz; must pair with ballots")
    p.add_argument("--ballots-file",help="Local saved ballots.csv.gz")
    a=p.parse_args()
    race_id=parse_id(a.race)
    if not 2021<=int(race_id[:4])<=2025:
        raise ValueError("Only 2021-2025 historical folds; 2026 unknown-year stays sealed")
    if bool(a.council_file)!=bool(a.ballots_file):
        raise ValueError("--council-file and --ballots-file must be supplied together")
    out=Path(a.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if a.council_file:
        council=Path(a.council_file).resolve()
        ballots=Path(a.ballots_file).resolve()
    else:
        session=out/"_saved_council_session"
        fetched=subprocess.run(
            ["bash",str(ROOT/"scripts"/"fetch-saved-council-github.sh"),
             race_id[:4],str(session)],
            cwd=ROOT,capture_output=True,text=True)
        if fetched.returncode:
            raise RuntimeError("Missing verified saved research files: "+(
                fetched.stderr or fetched.stdout)[-1000:])
        council=session/("council-y"+race_id[:4]+".jsonl.gz")
        ballots=session/("outsider-ballots-y"+race_id[:4]+".csv.gz")
    report=render(race_id,council,ballots)
    file=out/"race-id-archive-analysis.json"
    file.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "race_id":race_id,"status":report["status"],"horses":report["runners"],
        "models":20,"report":str(file)
    },ensure_ascii=False,indent=2))

if __name__=="__main__":main()
