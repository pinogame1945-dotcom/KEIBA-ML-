import importlib.util
import unittest
from pathlib import Path

path=Path(__file__).resolve().parents[1]/"scripts/live_race_source_probe_v1.py"
spec=importlib.util.spec_from_file_location("race_probe",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
URL="https://www.jra.go.jp/JRADB/accessD.html?CNAME=pw01dde0105202604061120261017%2F22"
N="""<table class="Shutuba_Table">
<tr><td class="Waku">4</td><td class="Umaban">1</td><td class="HorseInfo"><a href="/horse/2022105396/">アドマイヤズーム</a></td></tr>
<tr><td class="Waku">6</td><td class="Umaban">2</td><td class="HorseInfo"><a href="https://db.netkeiba.com/horse/2021110030/">アンパドゥ</a></td></tr>
</table><a href="/horse/9999999999/">サイドバー</a>"""
J="""ここから本文です 2026年10月17日（土曜）4回東京6日 11レース
<table><tr><th>枠</th><th>馬番</th><th>馬名</th></tr>
<tr><td><img alt="枠4青"></td><td>1</td><td>アドマイヤズーム</td></tr>
<tr><td><img alt="枠6緑"></td><td>2</td><td>アンパドゥ</td></tr></table>
<a onclick="doAction('/JRADB/accessO.html','official')">オッズ</a>"""

class TestRaceSourceProbe(unittest.TestCase):
    def test_id_parity(self):
        self.assertEqual(m.official_jra_identity(URL)["race_id"],"202605040611")
    def test_net_horse_ids(self):
        rows=m.parse_netkeiba_entries(N)
        self.assertEqual([r["horse_id"] for r in rows],["2022105396","2021110030"])
    def test_jra_crosswalk(self):
        n=m.parse_netkeiba_entries(N)
        m.verify_jra_card_header(J,m.official_jra_identity(URL))
        j=m.parse_official_jra_entries(J,{r["horse_name"] for r in n})
        out=m.validate_crosswalk("202605040611",URL,n,j,2)
        self.assertEqual(out["state"],"ID_CROSSWALK_VALIDATED")
        self.assertEqual([(r["horse_no"],r["horse_id"]) for r in out["entries"]],[(1,"2022105396"),(2,"2021110030")])
    def test_no_guess_before_draw(self):
        n=m.parse_netkeiba_entries(N)
        j=m.parse_official_jra_entries(J.replace("<td>2</td>","<td></td>"),{r["horse_name"] for r in n})
        self.assertEqual(m.validate_crosswalk("202605040611",URL,n,j,2)["state"],"WAIT_OFFICIAL_NUMBERING")
    def test_wrong_race(self):
        with self.assertRaisesRegex(ValueError,"JRA_NETKEIBA_RACE_ID_MISMATCH"):
            m.validate_crosswalk("202605040612",URL,[],[],2)
    def test_name_mismatch(self):
        n=m.parse_netkeiba_entries(N)
        j=m.parse_official_jra_entries(J.replace("アンパドゥ","別の馬"),{r["horse_name"] for r in n})
        with self.assertRaisesRegex(ValueError,"JRA_NETKEIBA_RUNNER_SET_MISMATCH"):
            m.validate_crosswalk("202605040611",URL,n,j,expected=None)
    def test_horse_id_absent(self):
        n=m.parse_netkeiba_entries(N.replace("/horse/2021110030/","/horseprofile/2021110030/"))
        self.assertEqual(len(n),1)
        with self.assertRaisesRegex(ValueError,"EXPECTED_RUNNER_COUNT_MISMATCH"):
            m.validate_crosswalk("202605040611",URL,n,m.parse_official_jra_entries(J,{"アドマイヤズーム"}),2)
    def test_gate_conflict(self):
        n=m.parse_netkeiba_entries(N.replace('class="Waku">4','class="Waku">5'))
        j=m.parse_official_jra_entries(J,{r["horse_name"] for r in n})
        with self.assertRaisesRegex(ValueError,"NETKEIBA_JRA_NUMBERING_CONFLICT"):
            m.validate_crosswalk("202605040611",URL,n,j,2)
    def test_heading_mismatch(self):
        with self.assertRaisesRegex(ValueError,"JRA_OFFICIAL_HEADER_IDENTITY_MISMATCH"):
            m.verify_jra_card_header(J.replace("4回東京6日","4回京都6日"),m.official_jra_identity(URL))
    def test_odds_not_prices(self):
        o=m.detect_jra_odds_navigation(J)
        self.assertEqual(o["state"],"OFFICIAL_ODDS_NAVIGATION_DISCOVERED")
        self.assertFalse(o["prices_collected"])
    def test_foreign_site(self):
        with self.assertRaisesRegex(ValueError,"JRA_OFFICIAL_CARD_URL_REQUIRED"):
            m.official_jra_identity(URL.replace("www.jra.go.jp","other.test"))

if __name__=="__main__": unittest.main()
