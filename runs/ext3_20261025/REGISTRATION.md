# EXT3 preregistration

One hypothesis: remove the shortcut of a constant answer letter using per-update exactly counterbalanced cyclic orders of unchanged published choices and a normalized choice discrimination loss. Update only the last two blocks and final norm (13,391,640 of100,028,328 parameters); retain a20% Alpaca language anchor. This is a warm start from EXT2 selected3584 with new AdamW, not a restart.

768 maximum updates,768 unique OBQA,768 PIQA,768 Alpaca source examples; four/two cyclic QA views, no invented question/choice/gold content. Each source example belongs to existing EXT2 train. Frozen new dev and test exclude prior MCQ train/dev/test and ARC128; new Alpaca held inputs/IDs exclude prior train/dev/test.

Select minimum equal-task normalized label NLL on new32 OBQA and32 PIQA dev questions under all cyclic orders (192 views). ARC diagnostic excluded. Parent eligible. Checkpoint64, dev128, cutoff07:55 UTC at next complete64-step boundary. Mark a time stop explicitly; do not claim768 completed unless reached.

One final test after selection:128 each new OBQA official test, ARC-Challenge official test, and reserved PIQA validation;64 new Alpaca. Compare original EXT256, EXT2 selected3584 and selected EXT3 with identical prompts and16-token full-vocabulary greedy budget. Primary endpoint: paired mean label-ranking accuracy gain versus EXT2, positive95% stratified paired-bootstrap interval, positive direction on>=2 tasks, and no worse Alpaca NLL. Format/entropy improvements alone are not success. Report position prior and dev permutation consistency. Original EXT256 is an additional measured anchor. No scientific ablation or test-driven retry.

Only local CPU; exact Qwen3.5-0.8B and MiniCPM-V-4.7-1B inaccessible via official HuggingFace CONNECT403 on2026-10-25; no replacement/leaderboard numbers. Publish only after completion to authorized binouv/model, ordinary binary parts<90MB and clean remote readback.
