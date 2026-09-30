# L2 TRIO Market Gap V0

Diagnostic stage only. Generates every priced three-horse TRIO combination from full-field L1.7, builds three-horse consensus/expert structure features, normalizes inverse final TRIO odds into a within-race market distribution, trains a walk-forward LightGBM LambdaRank model, and compares MARKET vs L1.7-product vs L1.7-sum vs market-aware model ranks. No LAW discovery, no ticket selection, no stake allocation, and no 2026 data. The primary outputs are Top1/3/5/10/20 winner capture and model-only versus market-only rescue counts.
