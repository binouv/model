# G3L: six finite optimizer-state continuations, not a new architecture or a victory claim

Registered before inspecting G3held outputs. The original nine2000-step G3runs keep their own fixed endpoint and metrics. This separate stage selects shared/hybrid only by their mean validation NLL, then continues that variant and full-attention for allthree seeds to totalstep8000, restoring actual model/optimizer/RNG/sampler state. Learning rate restarts under the fixed schedule inconfig; not a pure token-budget-only intervention. Same training data and ordered examples for the pair. Completed outputs only; no best-held checkpoint selection.

Status at registration: waiting for the finite G3parent campaign; no duplicate parent run. Six continuations x6000newupdates,batch32. Actual300M/800Mtraining is NOT claimed. This tests whether architecture effects persist or emerge after a more substantial optimization regime. The held400case suite is synthetic and reused; this cannot establish broad Qwenparity.

Separate BF16Qwen3.5-4Bthinking baseline is onbranch flygraph/g3-thinking-bf16-20260926 (32768tokens, same40fixture); low-bitQ4thinking and actual model-release CI are separately preserved. Do not mix their scopes. No paid API/service is used.
