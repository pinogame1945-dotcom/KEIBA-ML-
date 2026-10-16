import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location("p3",Path("scripts/fuji_prebet_phase3_20261016.py"))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
ids=[x["horse_id"] for x in a.expected["runners"]]
def fixture(weights=False):
 rows="".join(f'<tr><td>{i}</td><td>{i}</td><td><a href="/horse/{id}/">馬</a></td><td>{("480(+2)" if weights else "")}</td></tr>' for i,id in enumerate(ids,1))
 return '<html><div class="RaceName">富士S</div><div class="RaceData01">芝1600m(左A) 天候:晴 / 芝:稍重</div><table class="Shutuba_Table"><tr><th>枠</th><th>馬番</th><th>馬名</th><th>馬体重 (増減)</th></tr>'+rows+"</table></html>"
class Tests(unittest.TestCase):
 def test_unreleased(self):
  z=a.parse_card(fixture());self.assertEqual(z["weight_published_count"],0);self.assertEqual(z["weather"],"晴");self.assertEqual(z["turf_going"],"稍重")
 def test_published(self):
  z=a.parse_card(fixture(True));self.assertEqual(z["weight_published_count"],14);self.assertEqual(z["runners"][0]["body_weight"],480);self.assertEqual(z["runners"][0]["body_weight_diff"],2)
 def test_stale_jra_not_misused(self):
  d=a.parse_jra_baba("<h2>第4回東京競馬第5日（2026年10月12日）</h2><p>馬場状態（10月9日正午現在）天候:晴 芝 良</p>")
  self.assertEqual(d["status"],"STALE_OR_UNDATED")
 def test_missing_runners_blocks(self):
  with self.assertRaisesRegex(ValueError,"UNKNOWN_HORSE"):a.parse_card(fixture().replace(f'/horse/{ids[0]}/','/horse/9999999999/'))
if __name__=="__main__":unittest.main()
