import importlib.util,json,unittest
from pathlib import Path
from html import escape
spec=importlib.util.spec_from_file_location("change",Path("scripts/fuji_change_guard_20261016.py"))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
B=json.loads(g.BASE.read_text("utf-8"))
def fixture(rows=None,remove_id=None):
 source=rows or B["runners"]
 records=[]
 for h in source:
  if h["horse_id"]==remove_id:continue
  status={"SCRATCHED":"出走取消","EXCLUDED":"競走除外"}.get(h.get("entry_status","LISTED"),"")
  weight=str(h["carried_weight"])
  body=(f'{h["body_weight"]}({h["body_weight_diff"]:+d})' if h.get("body_weight")!=None and h.get("body_weight_diff")!=None else "")
  records.append('<tr>'+''.join([
   f'<td>{h["gate"]}</td>',
   f'<td>{h["horse_number"]}</td>',
   f'<td><a href="/horse/{h["horse_id"]}/">{escape(h["horse_name"])}</a>{status}</td>',
   f'<td>{h["sex"]}{h["age"]}</td>',
   f'<td>{weight}</td>',
   f'<td><a href="/jockey/result/recent/{h["jockey_id"]}/">騎手</a></td>',
   f'<td><a href="/trainer/result/recent/{h["trainer_id"]}/">調教師</a></td>',
   f'<td>{body}</td>'
  ])+"</tr>")
 return ('<div class="RaceName">富士ステークス</div><div class="RaceData01">芝1600m(左A)</div>'
         '<div class="RaceData02">4回東京6日目サラ系3歳以上オープン別定14頭</div>'
         '<table class="Shutuba_Table"><tr><th>枠</th><th>馬番</th><th>馬名</th>'
         '<th>性齢</th><th>斤量</th><th>騎手</th><th>厩舎</th><th>馬体重</th></tr>'
         +''.join(records)+'</table>')
class Tests(unittest.TestCase):
 def assertInvalidate(self,rr,event=None,field=None):
  d=g.compare(B,g.parse(fixture(rr)))
  self.assertEqual(d["status"],"INVALIDATE_RECOMPUTE_REQUIRED")
  self.assertTrue(d["prediction_invalidated"])
  self.assertFalse(d["actual_prediction_rerun"])
  if event:self.assertIn(event,[e["event"] for e in d["changes"]])
  if field:self.assertIn(field,[e.get("field") for e in d["changes"]])
 def test_identical(self):
  d=g.compare(B,g.parse(fixture()))
  self.assertEqual(d["status"],"UNCHANGED_NO_RECOMPUTE")
  self.assertEqual(d["changed_count"],0)
 def test_scratched_kept_in_card(self):
  rr=[dict(h) for h in B["runners"]];rr[1]["entry_status"]="SCRATCHED"
  self.assertInvalidate(rr,field="entry_status")
  self.assertIn(rr[1]["horse_id"],g.compare(B,g.parse(fixture(rr)))["current_scratched_or_excluded"])
 def test_excluded_is_not_dnf(self):
  rr=[dict(h) for h in B["runners"]];rr[2]["entry_status"]="EXCLUDED"
  self.assertInvalidate(rr,field="entry_status")
 def test_removed_entirely(self):
  d=g.compare(B,g.parse(fixture(remove_id=B["runners"][0]["horse_id"])))
  self.assertEqual(d["status"],"INVALIDATE_RECOMPUTE_REQUIRED")
  self.assertIn("REMOVED_FROM_CARD",[e["event"] for e in d["changes"]])
 def test_added_new_horse(self):
  rr=[dict(h) for h in B["runners"]]
  h=dict(rr[-1],horse_id="2024100001",horse_number=15,horse_name="確認用追加馬")
  rr.append(h)
  self.assertInvalidate(rr,event="ADDED_TO_CARD")
 def test_switched_jockey(self):
  rr=[dict(h) for h in B["runners"]];rr[4]["jockey_id"]="01234"
  self.assertInvalidate(rr,field="jockey_id")
 def test_switched_trainer(self):
  rr=[dict(h) for h in B["runners"]];rr[4]["trainer_id"]="01234"
  self.assertInvalidate(rr,field="trainer_id")
 def test_changed_weight(self):
  rr=[dict(h) for h in B["runners"]];rr[4]["carried_weight"]=58.
  self.assertInvalidate(rr,field="carried_weight")
 def test_announcement_of_body_weight(self):
  rr=[dict(h) for h in B["runners"]];rr[4]["body_weight"]=480;rr[4]["body_weight_diff"]=-2
  self.assertInvalidate(rr,field="body_weight")
 def test_duplicate_number_blocks(self):
  rr=[dict(h) for h in B["runners"]];rr[4]["horse_number"]=rr[3]["horse_number"]
  with self.assertRaisesRegex(ValueError,"HORSE_NUMBER_DUPLICATE"):g.parse(fixture(rr))
 def test_wrong_race_blocks(self):
  with self.assertRaisesRegex(ValueError,"RACE_IDENTITY"):g.parse(fixture().replace("4回東京6日目","5回東京6日目"))
if __name__=="__main__":unittest.main()
