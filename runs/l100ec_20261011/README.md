# L100EC — recovered results and runnable inference package

This directory preserves historical summary metrics and a separately verified recovery/publication layer. It does not report new training or a new quality benchmark.

The earlier result reports two100,028,328-parameter models, curriculum and shuffled-order control, trained for1536updates each. Saved IID counts47/240 versus37/240; surface15/120 versus11/120; length0/120 for both; counterfactual42/120 versus28/120, with19/60 versus13/60 both-correct pairs. Registered hypothesis NOT_CONFIRMED_IN_REGISTERED_PROTOCOL. No Qwen superiority claim.

## New: executable inference recovery

`delivery/` now contains the exact inherited L100M model and tokenizer implementation, a hash-verifying loader, CPU/CUDA-selectable inference CLI, configurations, requirements and12tests. Both actual BF16 models were rehashed and loaded strictly; their100,028,328parameter values are finite. Three sequences per arm matched after independent reload. The runnable local archive was extracted cleanly, all32payload hashes verified and12tests plus actual CLI generation executed. This is engineering recovery, not evidence of useful reasoning. One sanity arithmetic prompt did not finish within48tokens; that output is not hidden or called correct.

The new downloadable archive is `FlyGraph_L100EC_Runnable_BF16_2026-10-11.zip`,320,988,187bytes, SHA2562af9d1c3b438098dce71168961f84f406fd857a70e731667cff0ae685b365d3a. It includes actual weights, exact tokenizer, runnable source, available historical metrics and report. The Source ZIP contains all of that except weights. Manifest and replay receipts are under delivery/.

## Remaining publication/recovery gaps

The source/config/audit files are in Git. **The200,069,384-byte curriculum and shuffled weights are still NOT GitHub Release assets.** Use the conversation's runnable ZIP for actual bytes and place its model/tokenizer files at the paths in delivery/ASSETS.json when using the Git source directory. A manifest is not an upload.

Native FP32/AdamW/RNG, the original EC-specific trainer, complete EC training data and all primary held predictions are not available in this execution. Their presence in an old manifest is not a present-file check. The inherited model source has been recovered; that does not recover the missing trainer. Earlier publication status snapshots remain unchanged for provenance.

Current status: reports/CURRENT_PUBLICATION.json and delivery/reports/NEXT_STATE.json. L100CS/L100V and other parallel research runs were not changed or retrained by this publication task. The GitHub connector supports Git-object writes but not streamed local-file Release upload; direct container DNS access to GitHub failed. No unrelated storage account or paid service was used.
