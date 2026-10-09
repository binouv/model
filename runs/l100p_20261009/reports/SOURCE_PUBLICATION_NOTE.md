# Exact local source publication

The initial Git commit contained a compact protocol summary. This source commit publishes the actual running trainer, architecture, tests and full preregistered configuration fields. Hyperparameters,768-step endpoint,selection rule and effect gates are unchanged; no held result was used to change them. Original local config formatting/hash is also retained in the downloadable source archive.

Supplemental Qwen0.8 BF16 thinking inference is isolated on its own branch; it does not modify this primary experiment or provide training data. The generic phrase in primary config `qwen_run_this_stage=false` means no Qwen execution in the local training/evaluation stage. The separate finite reference CI is recorded in bench/CI_RECEIPT.json.

Data and parent weight bytes are prerequisites, preserved in conversation archives. Neither this source commit nor a manifest means the large checkpoint bytes have been uploaded. No claim of Qwen superiority is made.
