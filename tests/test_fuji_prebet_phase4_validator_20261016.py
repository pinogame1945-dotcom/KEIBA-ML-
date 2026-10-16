import importlib.util, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location("p4",Path("scripts/fuji_prebet_phase4_odds_validator_20261016.py"))
p4=importlib.util.module_from_spec(spec);spec.loader.exec_module(p4)
SOURCE="https://www.jra.go.jp/JRADB/accessO.html?CNAME=verified-from-live-page"
def batch():
 return dict(race_id="202605040611",source_url=SOURCE,race_header_verified=True,
  snapshot_type="PRE_RACE_LIVE_ODDS",fetched_at_utc="2026-10-16T10:00:00+00:00",
  page_updated_at=None,quotes=[
  dict(bet_type="WIN",combination=[1],odds=2.6),
  dict(bet_type="PLACE",combination=[1],odds_min=1.2,odds_max=1.6),
  dict(bet_type="BRACKET_QUINELLA",combination=[1,2],odds=4.2),
  dict(bet_type="QUINELLA",combination=[1,2],odds=7.2),
  dict(bet_type="WIDE",combination=[1,2],odds_min=1.5,odds_max=2.2),
  dict(bet_type="EXACTA",combination=[2,1],odds=11.3),
  dict(bet_type="TRIO",combination=[1,2,3],odds=16.4),
  dict(bet_type="TRIFECTA",combination=[2,1,3],odds=36.2)])
class Tests(unittest.TestCase):
 def test_all_eight_types_with_horse_crosswalk(self):
  b=batch();d=p4.validate_quote_batch(b)
  self.assertEqual(d["records"],8)
  self.assertTrue(d["timestamp_is_observation_only"])
  self.assertTrue(all(v==1 for v in d["by_bet_type"].values()))
 def test_parameter_error_013_blocks(self):
  b=batch();b["error013"]=True
  with self.assertRaisesRegex(ValueError,"ODDS_NOT_AVAILABLE"):p4.validate_quote_batch(b)
 def test_wrong_race_not_accepted(self):
  b=batch();b["race_id"]="202505040611"
  with self.assertRaisesRegex(ValueError,"RACE_ID_MISMATCH"):p4.validate_quote_batch(b)
 def test_unverified_page_blocks(self):
  b=batch();b["race_header_verified"]=False
  with self.assertRaisesRegex(ValueError,"RACE_HEADER_NOT_VERIFIED"):p4.validate_quote_batch(b)
 def test_missing_updated_at_is_not_lying(self):
  b=batch();b["page_updated_at"]="today"
  with self.assertRaisesRegex(ValueError,"UNVERIFIED_UPDATED_TIME"):p4.validate_quote_batch(b)
 def test_wrong_order_unordered_bet_blocks(self):
  b=batch();b["quotes"][3]["combination"]=[2,1]
  with self.assertRaisesRegex(ValueError,"UNORDERED_COMBINATION_NOT_SORTED"):p4.validate_quote_batch(b)
 def test_no_odds_records_blocks(self):
  b=batch();b["quotes"]=[]
  with self.assertRaisesRegex(ValueError,"NO_REAL_ODDS_QUOTES"):p4.validate_quote_batch(b)
if __name__=="__main__":unittest.main()
