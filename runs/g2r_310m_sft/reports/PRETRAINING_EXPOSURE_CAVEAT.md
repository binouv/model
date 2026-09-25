# Pretraining-exposure caveat, audited before test prediction

G2R semantic groups are disjoint across its train/validation/test sets and all six regenerated JSONL SHA256 hashes match the original published G2 data. However, a separate semantic comparison with G1's corpus found some matching ordered arithmetic operands/operator: IID5/32, surface4/16, extrapolation2/16, counterfactual5/40 arithmetic cases. Thus these tests are held out from the current SFT, NOT guaranteed never represented in the earlier G1 corpus.

G1 used sampled short blocks and only98,304token presentations. Presence in its corpus does not prove that a particular case was actually processed. Both trained arms and baseline share identical G1 starting weights, so the paired loss comparison remains meaningful. Do not claim contamination-free broad benchmark results. No test sets, criterion or training were changed after this audit. Local JSON preserves matching caseIDs.
