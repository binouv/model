# G7S — verified trace curriculum then final-only consolidation

Primary hypothesis: a two-phase curriculum—first emit an independently verified compact reasoning trace plus final answer, then consolidate with final-only targets—improves final-only free generation versus final-only training for all updates.

Same causal d-a-d-a hybrid, initialization within seed, cases, batch sequence, optimizer and 2000 updates. Main uses trace+final for steps1–1000 and final-only for1001–2000; control is final-only throughout. Parameters are identical (469648). Main spends more supervised/output tokens during phase1, so the test is not compute-neutral; wall time and token counts are recorded.

All G7 canonical groups are fresh and disjoint from every known G5S/G6S train/validation/held group. Held data are never used for checkpoint selection or hyperparameter tuning. Generation is unrestricted byte259 final-integer format, greedy, max32 new bytes, no trace/gold/case/tool input.

This is a small mechanism probe, not a 300–800M model and not Qwen parity.
