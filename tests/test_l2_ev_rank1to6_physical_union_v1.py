import csv,gzip,sys,tempfile,unittest
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank1to6_physical_union_v1 as study

class ExactUnionTest(unittest.TestCase):
    def test_adopted_source_six_rows_unchanged(self):
        rows=study.read_source()
        self.assertEqual([rows[i]["adopted_method"] for i in range(1,7)],
                         ["MARKET_10","MARKET_25","PROB_25","MARKET_75","LEARNED_10","LEARNED_10"])
        self.assertEqual(sum(x["purchased_tickets"] for x in rows.values()),682592)
        for rank,(policy,_,_,num) in study.EXPECT.items():
            lines,by_year=study.read_rank_lines(rank,policy)
            self.assertEqual(len(lines),num)
            self.assertEqual(set(by_year),{"2023","2024","2025","2023-2025"})
    def _write(self,p,rows):
        with gzip.open(p,"wt",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(("year","race_id","bet_type","selection","payout_yen"));w.writerows(rows)
    def test_cross_rank_same_physical_ticket_keeps_one_payout(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/"one.gz";b=Path(t)/"two.gz"
            self._write(a,[(2023,"raceA","TRIO","1,2,3",1510),
                           (2023,"raceA","EXACTA","1,2",0)])
            self._write(b,[(2023,"raceA","TRIO","1,2,3",1510),
                           (2023,"raceA","EXACTA","2,1",2000)])
            individual=defaultdict(lambda:dict(n=0,hits=0,payout=0))
            unique={}
            study.merge_file(a,1,individual,unique)
            study.merge_file(b,2,individual,unique)
            self.assertEqual(len(unique),3)
            self.assertEqual(sum(x[1] for x in unique.values()),3510)
            self.assertEqual(unique["raceA","TRIO","1,2,3"][2].bit_count(),2)
            self.assertEqual(individual[1,2023]["payout"],1510)
            self.assertEqual(individual[2,2023]["payout"],3510)
    def test_mismatched_refund_raises(self):
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/"one.gz";b=Path(t)/"two.gz"
            self._write(a,[(2024,"raceB","TRIO","1,2,3",200)])
            self._write(b,[(2024,"raceB","TRIO","1,2,3",300)])
            individual=defaultdict(lambda:dict(n=0,hits=0,payout=0))
            unique={}
            study.merge_file(a,1,individual,unique)
            with self.assertRaises(ValueError):
                study.merge_file(b,2,individual,unique)
    def test_no_2026_data_in_replay(self):
        self.assertEqual(tuple(study.YEARS),(2023,2024,2025))

if __name__=="__main__":unittest.main()
