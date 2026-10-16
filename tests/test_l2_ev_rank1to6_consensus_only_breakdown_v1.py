import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_ev_rank1to6_consensus_only_breakdown_v1 as agree

class ConsensusOnlyAuditTest(unittest.TestCase):
    def setUp(self):
        self.test={
            (2023,"r1","TRIO","1,2,3"):(3500,(1<<0)|(1<<1)),
            (2023,"r1","TRIFECTA","3,2,1"):(0,(1<<1)),
            (2024,"r2","QUINELLA","1,4"):(0,(1<<0)|(1<<1)|(1<<4)),
            (2025,"r3","WIN","9"):(300,(1<<5)),
        }
    def test_union_filter_requires_distinct_ev_rank_agreement(self):
        together=agree.filter_category(self.test,"CONSENSUS")
        alone=agree.filter_category(self.test,"SINGLE_RANK")
        self.assertEqual((len(together),len(alone)),(2,2))
        self.assertEqual(sum(v[0] for v in together.values()),3500)
        self.assertEqual(sum(v[0] for v in alone.values()),300)
        self.assertEqual(set(together)|set(alone),set(self.test))
        self.assertFalse(set(together)&set(alone))
    def test_three_rank_ticket_is_purchased_once(self):
        counts=agree.Counter(v[1].bit_count() for v in self.test.values())
        self.assertEqual(sum((k-1)*v for k,v in counts.items()),3)
        self.assertEqual(agree.take((1<<0)|(1<<1)|(1<<4)),(1,2,5))
    def test_year_and_type_detail(self):
        a=agree.metric_row("BASE","CONSENSUS",2023,"TRIO",
                           agree.filter_category(self.test,"CONSENSUS"))
        self.assertEqual((a["tickets"],a["hits"],a["stake_yen"],a["payout_yen"]),
                         (1,1,100,3500))
        self.assertEqual(a["roi_pct"],3500.0)
        b=agree.metric_row("BASE","CONSENSUS","2023-2025","ALL",
                           agree.filter_category(self.test,"CONSENSUS"))
        self.assertEqual((b["tickets"],b["hits"],b["stake_yen"],b["payout_yen"]),
                         (2,1,200,3500))
    def test_rank_pairwise_is_not_exact_2_only(self):
        _,_,_,pairs,combos=agree.build_reports(self.test,"BASE")
        pair={(r["rank_a"],r["rank_b"]):r for r in pairs}
        self.assertEqual(pair[(1,2)]["tickets"],2)
        self.assertEqual(pair[(1,5)]["tickets"],1)
        self.assertEqual(pair[(2,5)]["tickets"],1)
        self.assertEqual(sum(r["tickets"] for r in combos),2)
    def test_fixed_source_regression_totals(self):
        self.assertEqual(agree.base.EXPECTED["BASE"],(653734,3371,77239240))
        self.assertEqual(agree.STAGES,
            ("BASE","ALL_YEAR_BLACK","STABLE_MEDIUM","STABLE_STRICT"))
        self.assertEqual(agree.YEARS,(2023,2024,2025))

if __name__=="__main__":unittest.main()
