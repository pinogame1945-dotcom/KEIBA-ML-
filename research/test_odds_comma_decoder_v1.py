#!/usr/bin/env python3
from build_l2_bet_kings_dataset_v1 import decode_odds, finite

assert finite("2,173.2")==2173.2
assert finite("15,010.7")==15010.7
assert finite("10,326.5")==10326.5
assert finite(" 1,234.5 ")==1234.5
assert finite("bad") is None

record={
  "odds":{
    "1":{"01":["1,234.5",None,None]},
    "4":{"0102":["2,173.2",None,None]},
    "6":{"0102":["15,010.7",None,None]},
    "7":{"010203":["10,326.5",None,None]},
    "8":{"010203":["123,456.7",None,None]},
  }
}
got=decode_odds(record)
assert got[("WIN",(1,))]==1234.5
assert got[("QUINELLA",(1,2))]==2173.2
assert got[("EXACTA",(1,2))]==15010.7
assert got[("TRIO",(1,2,3))]==10326.5
assert got[("TRIFECTA",(1,2,3))]==123456.7
print("ODDS_COMMA_DECODER_REGRESSION_OK")
