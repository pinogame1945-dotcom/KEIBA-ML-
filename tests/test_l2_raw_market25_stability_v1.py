#!/usr/bin/env python3
"""No historical outcomes required: frozen MARKET25 and L2.1 stability invariants."""
import sys, unittest
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"research"))
import run_l2_raw_market25_stability_v1 as m
import run_l2_fixed_ticket_pruning_3040_v1 as fixed


class TestOfficialRawMarket25(unittest.TestCase):
    def test_fixed_policy_contract(self):
        self.assertEqual(len(m.GATES),17)
        self.assertEqual(len(m.CONFIG),2)
        self.assertEqual(m.CONFIG["STABLE_MEDIUM"]["per_year_hits"],8)
        self.assertEqual(m.CONFIG["STABLE_STRICT"]["per_year_hits"],12)
        self.assertEqual(m.EXPECTED_SOURCE_BLACK,235)

    def test_market_threshold_equivalence_to_original(self):
        history=[]
        stats=defaultdict(lambda:[0,0,0.])
        bucket=defaultdict(list)
        all_kind=defaultdict(list)
        for kind in ("WIN","EXACTA","TRIO","TRIFECTA"):
            for bz in (0,3,11,19):
                for i in range(130):
                    score=-(i%43)/17-(bz/11)-(i%5)/400
                    card=(float(2+i%9),int(i%17==0)*100,
                          (kind,bz,i%3,i%6,i%7),-.1,score,0.)
                    history.append(card)
                    bucket[kind,bz].append(score)
                    all_kind[kind].append(score)
                    fixed.add_learning(stats,card)
        original=fixed.create_thresholds(history,stats)
        optimized=m.prior_market25_thresholds(bucket,all_kind)
        for card in history:
            kind,bz=card[2][:2]
            exact=m.market25_keep(card,2024,optimized)
            reference=fixed.mask(card,stats,original)["MARKET_25"]
            self.assertEqual(exact,reference)
        self.assertTrue(m.market25_keep(history[0],2023,{}))

    def test_annual_and_jackpot_screen(self):
        key=("AI_RAW",3,"TRIO","30-50")
        yearly=defaultdict(lambda: defaultdict(m.new_counter))
        for year in (2023,2024,2025):
            st=yearly[year][(key,"ALL")]
            st.update(n=800,hits=80,payout=128000,top3=[1600,1600,1600])
        kept,audit=m.select_stable(yearly,{key:{}})
        self.assertIn((key,"ALL"),kept["STABLE_MEDIUM"])
        self.assertIn((key,"ALL"),kept["STABLE_STRICT"])
        self.assertIn((key,"ALL"),kept["ALL_YEAR_BLACK"])
        self.assertEqual(len(audit),17)
        yearly[2024][(key,"ALL")]["payout"]=10
        kept2,_=m.select_stable(yearly,{key:{}})
        self.assertNotIn((key,"ALL"),kept2["STABLE_MEDIUM"])
        self.assertNotIn((key,"ALL"),kept2["ALL_YEAR_BLACK"])

    def test_physical_dedup_and_strict_subset(self):
        line=("AI_RAW",3,"TRIO","30-50")
        a=("TRIO",(1,2,3))
        gates=set([("ALL_YEAR_BLACK",line,"ALL"),
                   ("STABLE_MEDIUM",line,"ALL"),
                   ("STABLE_STRICT",line,"ALL")])
        accepted={p:{(line,"ALL")} for p in m.RULES}
        events=[(2023,"race1",line,a,800,m.GATES.index("ALL") or 1),
                (2023,"race1",line,a,800,1)]
        # The same physical ticket can originate from two accepted rules.
        r=m.dedup_portfolios(events,accepted)
        for p in m.RULES:
            row=next(x for x in r if x["policy"]==p and x["period"]=="2023-2025")
            self.assertEqual(row["tickets"],1)
            self.assertEqual(row["payout_yen"],800)


if __name__=="__main__": unittest.main()
