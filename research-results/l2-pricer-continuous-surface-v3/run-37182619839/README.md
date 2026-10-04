# L2 PRICER Continuous Surface V3

A pooled Huber LightGBM ensemble learns context + continuous volatility parameters -> trifecta NLL delta. The deployable policy starts from the prior-only static optimum and only switches when the learned surface predicts enough incremental benefit. Dense sigma/lambda candidates are used at prediction time. Odds, ROI, ticket selection and staking are excluded. 2026 sealed.
