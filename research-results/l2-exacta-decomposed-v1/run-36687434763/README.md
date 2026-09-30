# L2 EXACTA V1 — Direct vs Decomposed

The frozen corrected V0 direct model is reused as the baseline and is not retrained. The new path halves the candidate matrix to unordered first-two pairs, ranks those pairs, then chooses exactly one orientation with a separate direction classifier. This compares equal ticket budgets at Top-K: direct uses K ordered tickets; decomposed uses K unordered pairs with one chosen orientation each. No LAW/ROI search is performed. 2026 outcomes remain sealed.
