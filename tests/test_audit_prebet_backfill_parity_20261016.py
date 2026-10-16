import importlib.util
import unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"scripts/audit_prebet_backfill_parity_20261016.py"
spec=importlib.util.spec_from_file_location("audit",p)
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)

class TestParity(unittest.TestCase):
  def test_parse_horse_identifiers(self):
    html="""<div class="RaceList_NameBox"><div class="RaceData02">3歳以上オープン(国際)(指)別定</div></div><table class="Shutuba_Table">
    <tr class="HorseList"><td class="Waku">2</td><td class="Umaban">3</td><td class="HorseInfo"><a href="https://db.netkeiba.com/horse/2021100953/">ウォーターリヒト</a></td><td class="Barei">牡4</td><td class="Kinryo">57.0</td>
    <td class="Jockey"><a href="/jockey/01170/">菅原明</a></td><td class="Trainer"><a href="/trainer/01234/">厩舎</a></td></tr></table>"""
    rows,h=a.parse_netkeiba(html)
    self.assertEqual(len(rows),1)
    self.assertEqual(rows[0]["horse_id"],"2021100953")
    self.assertEqual(rows[0]["jockey_id"],"01170")
    self.assertEqual(rows[0]["trainer_id"],"01234")
    self.assertEqual(rows[0]["carried_weight"],57.)
    self.assertEqual(h["condition"],"3歳以上オープン(国際)(指)別定")
  def test_exact(self):
    x={"horse_id":"2021100953","horse_name":"ウォーターリヒト","horse_number":3,"gate":2,"sex":"牡","age":4,
       "carried_weight":57.0,"jockey_id":"01170","trainer_id":"01234"}
    self.assertTrue(a.compare([x],[dict(x)])["strict_value_match"])
  def test_missing_not_success(self):
    x={"horse_id":"2021100953","horse_name":"ウォーターリヒト","horse_number":3,"gate":2,"sex":"牡","age":4,
       "carried_weight":57.0,"jockey_id":"01170","trainer_id":"01234"}
    y=dict(x);y["trainer_id"]=None
    self.assertFalse(a.compare([x],[y])["strict_value_match"])
    self.assertEqual(a.compare([x],[y])["field_differences"][0]["type"],"MISSING")
  def test_conflict_not_success(self):
    x={"horse_id":"2021100953","horse_name":"ウォーターリヒト","horse_number":3,"gate":2,"sex":"牡","age":4,
       "carried_weight":57.0,"jockey_id":"01170","trainer_id":"01234"}
    y=dict(x);y["horse_number"]=5
    self.assertFalse(a.compare([x],[y])["strict_value_match"])
  def test_missing_horse_not_success(self):
    x={"horse_id":"ABC","horse_name":"X"}
    r=a.compare([x],[])
    self.assertFalse(r["strict_value_match"])
    self.assertEqual(r["missing_in_public"],["ABC"])

if __name__=="__main__":unittest.main()
