import importlib.util,itertools,unittest,sys
from pathlib import Path
from bs4 import BeautifulSoup
sys.path.insert(0,"scripts")
spec=importlib.util.spec_from_file_location("odds",Path("scripts/fuji_8type_live_odds_20261016.py"))
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
def make_table(typ):
 opts={"BRACKET_QUINELLA":("waku",8),"QUINELLA":("umaren",13),"WIDE":("wide",13),
       "EXACTA":("umatan",14),"TRIO":("fuku3",78),"TRIFECTA":("tan3",182)}
 cl,n=opts[typ]
 out=[]
 for idx in range(n):
  if typ=="BRACKET_QUINELLA":
   a=idx+1;prefix=(a,);seconds=range(a,9)
  elif typ in ("QUINELLA","WIDE"):
   a=idx+1;prefix=(a,);seconds=range(a+1,15)
  elif typ=="EXACTA":
   a=idx+1;prefix=(a,);seconds=range(1,15)
  elif typ=="TRIO":
   prefix=list(itertools.combinations(range(1,14),2))[idx];seconds=range(prefix[-1]+1,15)
  else:
   prefix=list(itertools.permutations(range(1,15),2))[idx];seconds=[x for x in range(1,15) if x!=prefix[0]]
  rows=[]
  for second in seconds:
   invalid=(typ=="EXACTA" and second==prefix[0]) or (typ=="TRIFECTA" and second==prefix[1])
   invalid=invalid or (typ=="BRACKET_QUINELLA" and second==prefix[0] and sum(h["gate"]==second for h in o.RUNNERS.values())<2)
   val="" if invalid else "8.8 - 9.9" if typ=="WIDE" else "100.0"
   rows.append(f"<tr><th>{second}</th><td>{val}</td></tr>")
  out.append(f'<table class="{cl}">{"".join(rows)}</table>')
 return BeautifulSoup("".join(out),"html.parser")
class Tests(unittest.TestCase):
 def test_theoretical_coverage(self):
  e=o.expected()
  self.assertEqual({k:len(v) for k,v in e.items()},{"WIN":14,"PLACE":14,
   "BRACKET_QUINELLA":34,"QUINELLA":91,"WIDE":91,"EXACTA":182,"TRIO":364,"TRIFECTA":2184})
 def test_every_matrix_layout_complete(self):
  e=o.expected()
  for bet in ("BRACKET_QUINELLA","QUINELLA","WIDE","EXACTA","TRIO","TRIFECTA"):
   parsed=o.parse_bet(make_table(bet),bet)[bet]
   self.assertEqual(set(k for k,v in parsed),e[bet],bet)
 def test_no_votes_not_made_into_positive_odds(self):
  self.assertEqual(o.parse_one_odd("票数なし"),{"status":"NO_VOTES","odds":None})
  self.assertEqual(o.parse_one_odd("票数なし",True),{"status":"NO_VOTES","odds_min":None,"odds_max":None})
 def test_range_and_numeric(self):
  self.assertEqual(o.parse_one_odd("5.3 - 6.1",True),{"status":"QUOTED","odds_min":5.3,"odds_max":6.1})
  self.assertEqual(o.parse_one_odd("7.2"),{"status":"QUOTED","odds":7.2})
 def test_invalid_odds_fail_closed(self):
  for raw in ("0.0","-1.4","1.2 foo","","未知"):
   with self.assertRaises(ValueError):o.parse_one_odd(raw)
 def test_missing_table_rejected(self):
  s=make_table("TRIFECTA")
  s.select_one("table.tan3").decompose()
  with self.assertRaisesRegex(ValueError,"TABLE_COUNT_MISMATCH"):o.parse_bet(s,"TRIFECTA")
 def test_swapped_row_number_rejected(self):
  s=make_table("QUINELLA")
  s.select("table.umaren")[0].select("tr")[0].th.string="14"
  with self.assertRaisesRegex(ValueError,"MATRIX_ROW_NUMBER_ORDER"):o.parse_bet(s,"QUINELLA")
 def test_no_votes_preserved_as_non_numeric_in_matrix(self):
  s=make_table("EXACTA")
  s.select("table.umatan")[0].select("tr")[1].td.string="票数なし"
  parsed=o.parse_bet(s,"EXACTA")["EXACTA"]
  self.assertEqual(o.parse_one_odd(dict(parsed)[(1,2)]),{"status":"NO_VOTES","odds":None})
if __name__=="__main__":unittest.main()
