# Additional novelty audit before held predictions

All six dataset hashes match the original G2 files, and its declared semantic-group hashes are disjoint across SFT splits. However, a stronger canonicalization of commutative arithmetic operands found equivalent arithmetic cases in the SFT training set: validation2, IID4, unseen-surface3, extrapolation0, counterfactual3. The primary data/protocol were NOT changed after this audit. The completed aggregate will separately show a diagnostic excluding these known overlaps plus G1 corpus overlaps.

Defined hash-group disjointness is not a proof that every algebraically equivalent problem is unseen. Presence in the original G1 corpus is also not proof of actual G1 training exposure because only a small sample of its corpus was processed. Both trained arms start from exactly the same G1 bytes, so the matched loss-mask comparison is still valid within this limited suite. Do not call it contamination-free broad evaluation.
