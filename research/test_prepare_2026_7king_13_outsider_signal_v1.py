#!/usr/bin/env python3
"""Synthetic-only coverage gate for original 13 outsider ballot adapter."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from prepare_2026_7king_13_outsider_research_signal_v1 import (
    canonical_candidates, load_consensus, run,
)

class Outsider2026ResearchIntakeSafetyTests(unittest.TestCase):
    def make_inputs(self,root):
        ids=[f"2026{v}0403{n:02d}" for v in ("05","08") for n in range(1,13)]
        ids += [f"2026{v}04{d}{n:02d}" for v,d in
                (("04","01"),("05","06"),("08","06")) for n in range(1,13)]
        self.assertEqual(len(ids),60)
        sources=root/"consensus.jsonl"
        fields={}
        with sources.open("w",encoding="utf-8") as out:
            for i,rid in enumerate(ids):
                # 10 Oct = 24 races/315 horses, 17 Oct = 36 races/471 horses
                n=(14 if i<3 else 13) if i<24 else (14 if i<3+24 else 13)
                horses=[]
                for k in range(n):
                    hid=str(2000000000+i*30+k)
                    horses.append({
                        "horse_id":hid,"consensus_rank":k+1,
                        "borda_score":float(7*(n-k)),
                        "probability_mean":1/n,
                        "expert_ranks":{x:k+1 for x in
                            ("core4","pedlegacy","condition","full","light","pedv1","condrc")}
                    })
                fields[rid]=[h["horse_id"] for h in horses]
                out.write(json.dumps({
                    "contract":"L16_FULL_RANK_CONSENSUS_V1",
                    "race_id":rid,"expert_count":7,"candidate_policy":"ALL_HORSES",
                    "field_size":n,"horses":horses
                })+"\n")
        self.assertEqual(sum(len(x) for x in fields.values()),786)
        ballot_dir=root/"outsider-top3"
        ballot_dir.mkdir()
        for cand in canonical_candidates():
            with (ballot_dir/(cand+".csv")).open("w",newline="",encoding="utf-8") as out:
                w=csv.DictWriter(out,fieldnames=[
                    "race_id","candidate","top1_horse_id","top2_horse_id","top3_horse_id"])
                w.writeheader()
                for rid,horseids in fields.items():
                    w.writerow({"race_id":rid,"candidate":cand,
                                "top1_horse_id":horseids[0],
                                "top2_horse_id":horseids[1],
                                "top3_horse_id":horseids[2]})
        return sources,ballot_dir,ids

    def test_13_frozen_candidates_exact(self):
        names=canonical_candidates()
        self.assertEqual(len(names),13)
        self.assertEqual(len(set(names)),13)
        self.assertIn("outsider_rotation",names)
        self.assertIn("outsider_chimera",names)

    def test_all_fake_ballots_keep_seven_ranks_and_no_calibration(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            council,ballots,ids=self.make_inputs(root)
            dest=root/"joined.jsonl"
            run(council,ballots,dest)
            data=[json.loads(s) for s in dest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(data),60)
            self.assertEqual(sum(z["horse_count"] for z in data),786)
            self.assertTrue(all(z["ready_for_betting"] is False for z in data))
            for x in data:
                self.assertEqual(x["ranking_policy"],"SEVEN_KING_UNCHANGED")
                self.assertEqual([p["king_rank"] for p in x["horses"]],
                                 list(range(1,x["horse_count"]+1)))
                self.assertEqual(x["horses"][0]["outsider_support_count"],13)
                self.assertEqual(x["horses"][0]["outsider_score"],39)
                self.assertEqual(x["horses"][1]["outsider_score"],26)
                self.assertEqual(x["horses"][2]["outsider_score"],13)
                self.assertTrue(all(h["p3_calibrated"] is None for h in x["horses"]))

    def test_missing_one_of_13_is_hard_failure_no_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            council,ballots,_=self.make_inputs(root)
            (ballots/"outsider_rotation.csv").unlink()
            target=root/"not-written.jsonl"
            with self.assertRaisesRegex(ValueError,"MISSING_OR_EXTRA_13"):
                run(council,ballots,target)
            self.assertFalse(target.exists())

    def test_unobserved_ballot_horse_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            council,ballots,_=self.make_inputs(root)
            ballot=ballots/"outsider_chimera.csv"
            lines=ballot.read_text().splitlines()
            lines[1]=lines[1].replace(lines[1].split(",")[2],"999999999999")
            ballot.write_text("\n".join(lines)+"\n")
            target=root/"not-written.jsonl"
            with self.assertRaisesRegex(ValueError,"NOT_IN_FROZEN_KING_FIELD"):
                run(council,ballots,target)
            self.assertFalse(target.exists())

if __name__=="__main__":
    unittest.main()
