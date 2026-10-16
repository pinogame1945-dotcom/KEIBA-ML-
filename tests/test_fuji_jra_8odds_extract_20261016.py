import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location("fuji_odds",Path("scripts/fuji_jra_8odds_extract_20261016.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class OddParsing(unittest.TestCase):
 def test_no_votes_is_not_zero(self):
  j=m.quote("票数なし")
  self.assertEqual(j["status"],"NO_PUBLISHED_PRICE")
  self.assertIsNone(j["odds"])
 def test_valid_scalar(self):
  self.assertEqual(m.quote("3.0")["odds"],3.)
 def test_valid_range(self):
  j=m.quote("1.1 - 2.5",ranged=True)
  self.assertEqual((j["odds_min"],j["odds_max"]),(1.1,2.5))
 def test_invalid_range_fails(self):
  with self.assertRaisesRegex(ValueError,"ODDS_RANGE_INVALID"):m.quote("4.0 - 1.0",True)
 def test_bad_numeric_fails(self):
  with self.assertRaisesRegex(ValueError,"ODDS_FORMAT_UNVERIFIED"):m.quote("unknown")
 def test_horse_count(self):
  self.assertEqual(len(m.expected),14)
  self.assertEqual(sorted(m.expected),list(range(1,15)))
 def test_no_final_odds_label(self):
  self.assertEqual(set(m.JRA_TYPES),{"単勝・複勝","枠連","馬連","ワイド","馬単","3連複","3連単"})
if __name__=="__main__":unittest.main()
