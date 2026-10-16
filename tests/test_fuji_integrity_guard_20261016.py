import importlib.util,json,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
sp=importlib.util.spec_from_file_location("guard",Path("scripts/fuji_integrity_guard_20261016.py"))
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
bs=m.BASE
ref=[dict(h,entry_status="LISTED") for h in bs["runners"]]
CURRENT={"race_id":m.RID,"horses":ref}
OFFICIAL=[{"horse_name":h["horse_name"],"gate":h["gate"],"horse_number":h["horse_number"],"entry_status":"LISTED"} for h in ref]
CLOCK="2026-10-16T05:00:00+00:00"
NOW="2026-10-16T05:00:10+00:00"
def clone(v):return json.loads(json.dumps(v))
def metadata(url,t=CLOCK):
 return {"race_id":m.RID,"url":url,"sha256":"a"*64,"fetched_at_utc":t}
def inputs(change=None,official=None,net=None,jra=None,odds=None,max_age=120,max_skew=40):
 c=change or {"status":"UNCHANGED_NO_RECOMPUTE","prediction_invalidated":False}
 return m.integrity(clone(CURRENT),clone(official or OFFICIAL),c,
  net or metadata(m.ORIGINAL_CARD),jra or metadata(m.OFFICIAL),
  now_utc=NOW,max_age_seconds=max_age,max_skew_seconds=max_skew,odds=odds)
class Tests(unittest.TestCase):
 def test_real_card_plus_jra_consistent_but_no_odds_not_ready(self):
  r=inputs()
  self.assertEqual(r["status"],"CARD_CROSSCHECKED_WAITING_FOR_LIVE_ODDS")
  self.assertFalse(r["quote_available"])
  self.assertEqual(r["cross_source"]["official_verified_runners"],14)
 def test_jockey_switch_forces_invalidation(self):
  r=inputs(change={"status":"INVALIDATE_RECOMPUTE_REQUIRED","prediction_invalidated":True})
  self.assertEqual(r["status"],"BLOCKED_INTEGRITY")
  self.assertTrue(r["recalculation_needed"])
  self.assertIn("INPUT_CHANGED_OLD_PREDICTION_INVALID",r["reasons"])
 def test_crosscheck_official_wrong_number(self):
  wrong=clone(OFFICIAL);wrong[0]["horse_number"]=99
  r=inputs(official=wrong)
  self.assertIn("BLOCKED_SOURCE_DISAGREEMENT",r["reasons"])
 def test_official_missing_horse(self):
  r=inputs(official=OFFICIAL[:-1])
  self.assertIn("BLOCKED_RUNNER_SET_DIVERGENCE",r["reasons"])
 def test_stale_source(self):
  meta=metadata(m.ORIGINAL_CARD,"2026-10-16T04:57:00+00:00")
  r=inputs(net=meta)
  self.assertIn("NET_SOURCE_STALE",r["reasons"])
 def test_mixed_observation_times(self):
  meta=metadata(m.OFFICIAL,"2026-10-16T04:58:30+00:00")
  r=inputs(jra=meta,max_age=130)
  self.assertIn("SOURCE_TIMESTAMPS_TOO_FAR_APART",r["reasons"])
 def test_future_timestamp(self):
  meta=metadata(m.OFFICIAL,"2026-10-16T05:02:00+00:00")
  r=inputs(jra=meta)
  self.assertIn("JRA_SOURCE_CLOCK_IN_FUTURE",r["reasons"])
 def test_source_wrong_race(self):
  meta=metadata(m.OFFICIAL);meta["race_id"]="202605040610"
  r=inputs(jra=meta)
  self.assertIn("SOURCE_RACE_ID_MISMATCH",r["reasons"])
 def test_unverified_source_link(self):
  meta=metadata("https://example.org/race.html")
  r=inputs(jra=meta)
  self.assertIn("JRA_UNTRUSTED_SOURCE_URL",r["reasons"])
 def test_requires_explicit_freshness_policy(self):
  r=inputs(max_age=None,max_skew=None)
  self.assertIn("FRESHNESS_LIMIT_NOT_SUPPLIED",r["reasons"])
  self.assertIn("CROSS_SOURCE_SKEW_LIMIT_NOT_SUPPLIED",r["reasons"])
 def test_unauthenticated_placeholder_odds_blocks(self):
  odds={"race_id":m.RID,"source_url":"https://www.jra.go.jp/JRADB/accessO.html",
    "source_error":"ERROR_013","quotes":[]}
  r=inputs(odds=odds)
  self.assertIn("ODDS_CONTRACT_ODDS_NOT_AVAILABLE",r["reasons"])
if __name__=="__main__":unittest.main()
