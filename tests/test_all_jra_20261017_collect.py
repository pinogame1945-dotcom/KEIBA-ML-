import itertools
import math
import sys
import unittest
from bs4 import BeautifulSoup
sys.path.insert(0,"scripts")
import all_jra_20261017_collect as x

class AllJraTests(unittest.TestCase):
 def test_thirty_six_id_scheme_only_discovery_from_html(self):
  urls=[
   "pw01dde0105202604061120261017/22",
   "pw01dde0108202604060120261017/AE",
   "pw01dde0104202604011220261017/4E"]
  actual=[x.CARD_CNAME.search(t) for t in urls]
  self.assertTrue(all(actual))
  ids=[f"2026{m.group('venue')}{m.group('meet')}{m.group('day')}{m.group('no')}" for m in actual]
  self.assertEqual(ids,["202605040611","202608040601","202604040112"])
  self.assertIsNone(x.CARD_CNAME.search("pw01dde0105202604061120261018/22"))
 def test_coverage_dynamic(self):
  for n in (7,8,9,14,18):
   frames={k:((k-1)*8//n)+1 for k in range(1,n+1)}
   e=x.theoretical(n,frames)
   self.assertEqual(len(e["TRIFECTA"]),n*(n-1)*(n-2))
   self.assertEqual(len(e["TRIO"]),math.comb(n,3))
   self.assertEqual(len(e["QUINELLA"]),math.comb(n,2))
   self.assertEqual(len(e["EXACTA"]),n*(n-1))
 def build(self,bet,n):
  cl=x.BET_TABLE[bet]
  tables=[]
  count={"QUINELLA":n-1,"WIDE":n-1,"EXACTA":n,"TRIO":math.comb(n-1,2),"TRIFECTA":n*(n-1)}[bet]
  for index in range(count):
   if bet in ("QUINELLA","WIDE"):
    pre=(index+1,);seconds=range(pre[0]+1,n+1)
   elif bet=="EXACTA":
    pre=(index+1,);seconds=range(1,n+1)
   elif bet=="TRIO":
    pre=list(itertools.combinations(range(1,n),2))[index];seconds=range(pre[-1]+1,n+1)
   else:
    pre=list(itertools.permutations(range(1,n+1),2))[index];seconds=[b for b in range(1,n+1) if b!=pre[0]]
   rows=[]
   for b in seconds:
    bad=(bet=="EXACTA" and b==pre[0]) or (bet=="TRIFECTA" and b==pre[1])
    odds="" if bad else "4.0 - 5.0" if bet=="WIDE" else "17.4"
    rows.append(f"<tr><th>{b}</th><td>{odds}</td></tr>")
   tables.append(f'<table class="{cl}">{"".join(rows)}</table>')
  return BeautifulSoup("".join(tables),"html.parser")
 def test_three_horse_matrix_sizes(self):
  for n in (7,14,18):
   gates={k:min(8,k) for k in range(1,n+1)}
   coverage=x.theoretical(n,gates)
   for bet in ("QUINELLA","WIDE","EXACTA","TRIO","TRIFECTA"):
    actual=x.parse_matrix(self.build(bet,n),bet,n,gates)[bet]
    self.assertEqual(set(combo for combo,_ in actual),coverage[bet],(n,bet))
 def test_no_votes_preserved(self):
  self.assertEqual(x.base_odds.parse_one_odd("票数なし")["odds"],None)
  self.assertEqual(x.base_odds.parse_one_odd("票数なし",True)["odds_min"],None)
 def test_wrong_row_blocked(self):
  n=7
  soup=self.build("TRIFECTA",n)
  soup.select("table.tan3")[0].select("tr")[0].find("th").string="99"
  with self.assertRaisesRegex(ValueError,"JRA_MATRIX_ROW_ORDER"):
   x.parse_matrix(soup,"TRIFECTA",n,{k:k for k in range(1,n+1)})

if __name__=="__main__":unittest.main()
