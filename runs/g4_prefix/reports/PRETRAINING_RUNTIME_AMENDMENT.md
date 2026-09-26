# Pre-training runtime amendment

The original 1200-step/batch32 preregistration was amended **before any completed training run or held-set evaluation**. A compute-only probe showed that byte-level prompts made the parallel triangular-solve delta implementation resource-prohibitive. One incomplete causal run reached only about step200 before the execution window ended; it is explicitly excluded from all results.

The delta update was replaced by the mathematically identical sequential recurrence (same decay, prediction error, outer-product state update and read). Engineering tests are rerun. The finite paired screen is now 800 updates, batch16, seeds7501/7502/7503 for both arms. Success gates and held sets are unchanged. This amendment is based only on runtime, not accuracy.