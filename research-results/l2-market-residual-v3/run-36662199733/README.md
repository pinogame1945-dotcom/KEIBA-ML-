# L2 Market Residual V3

A market-context model and a market+L1.7 model predict the same ticket-hit target. The decision signal is delta_logit = logit(p_full) - logit(p_market_context), so market odds are not multiplied into a raw EV score. Policy selection uses 2023-2024 only and maximizes the worse yearly ROI, with a minimum 20% execution coverage in both years. 2025 is confirmation only and is not used to select thresholds. 2026 remains sealed.
