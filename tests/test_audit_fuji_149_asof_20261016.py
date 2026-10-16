import importlib.util
import json
import unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/"scripts/audit_fuji_149_asof_20261016.py"
spec=importlib.util.spec_from_file_location("audit149",P)
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
def html(rows=14):
    header="""<html><head><title>富士ステークス(G2) 2026 出馬表</title></head><body>
    <div class="RaceList_NameBox"><span class="RaceName">富士S</span>
    <div class="RaceData01">15:45発走 / 芝1600m (左 A)</div>
    <div class="RaceData02">4回 東京 6日目 サラ系３歳以上 オープン (国際)(指) 別定 14頭</div></div>
    <table class="Shutuba_Table"><tr><th>枠</th><th>馬番</th><th>印</th><th>馬名</th>
    <th>性齢</th><th>斤量</th><th>騎手</th><th>厩舎</th><th>馬体重</th></tr>"""
    entries="".join(f"""<tr class="HorseList"><td>{(i+1)//2}</td><td>{i}</td><td></td>
     <td class="HorseInfo"><a href="https://db.netkeiba.com/horse/202210{5000+i:04d}/">馬{i}</a></td>
     <td>牡4</td><td>57.0</td><td><a href="/jockey/01182/">騎手</a></td>
     <td><a href="/trainer/result/recent/01061/">調教師</a></td><td></td></tr>""" for i in range(1,rows+1))
    return header+entries+"</table></body></html>"

class TestAsOf149(unittest.TestCase):
    def test_full_catalog_and_no_hindsight(self):
        features=json.loads(a.CATALOG.read_text())["features"]
        rows,raw,meta=a.parse(html())
        self.assertEqual(len(rows),14)
        self.assertEqual(len({x["horse_id"] for x in rows}),14)
        self.assertEqual(rows[0]["horse_number"],1)
        self.assertEqual(rows[0]["jockey_id"],"01182")
        self.assertEqual(rows[0]["trainer_id"],"01061")
        self.assertEqual(rows[0]["carried_weight"],57.0)
        ledger=a.audit({"features":features},rows,raw)
        self.assertEqual(len(ledger),149)
        d={x["feature_id"]:x for x in ledger}
        self.assertEqual(d["entry.horse_id"]["coverage"],{"present":14,"expected":14})
        self.assertEqual(d["race.distance_m"]["observed_value"],1600)
        self.assertEqual(d["race.grade"]["observed_value"],"G2")
        self.assertEqual(d["race.designated"]["observed_value"],True)
        self.assertEqual(d["entry.body_weight"]["status"],"NOT_YET_PUBLISHED")
        self.assertEqual(d["target.finish_position"]["status"],"NOT_YET_PUBLISHED")
        self.assertEqual(d["market.final_win_odds"]["status"],"POST_RACE_FINAL_ONLY")
        self.assertEqual(d["history.prior_starts"]["status"],"BACKFILL_HISTORY_REQUIRED")
        self.assertTrue(meta["designation_legacy_regex_conflict"])
    def test_missing_runners_block(self):
        with self.assertRaisesRegex(ValueError,"RUNNER_COUNT"):
            a.parse(html(rows=13))
    def test_wrong_race_block(self):
        with self.assertRaisesRegex(ValueError,"RACE_NAME_MISMATCH"):
            a.parse(html().replace("富士","別物"))
    def test_not_guess_missing_weight(self):
        rows,raw,_=a.parse(html().replace("<td>57.0</td>","<td></td>"))
        ledger=a.audit({"features":json.loads(a.CATALOG.read_text())["features"]},rows,raw)
        w=next(x for x in ledger if x["feature_id"]=="entry.carried_weight")
        self.assertEqual(w["status"],"PARTIAL_OR_NOT_EXTRACTED")
        self.assertEqual(w["coverage"]["present"],0)

if __name__=="__main__":unittest.main()
