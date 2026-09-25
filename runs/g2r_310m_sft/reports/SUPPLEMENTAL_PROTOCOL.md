# Separate external40case protocol

The primary G2R experiment remains400cases,greedy8newtokens,no added integer-only prefix. Its criteria and final checkpoint were fixed before testing.

An independently completed GitHub Actions Qwen4B run is available from the parallel G2 namespace. Its forty prompts are exactly the first8IIDcases per family in original file order. Reconstructing the local fixture produced the published SHA256ddef92124f5fe5f9058711480ceba522472606f06d4e2568aadf232f7317828d. Rejoining forty raw external responses to these IDs and gold values reproduced32/40correct. Workflow36186603496atcommit450d46f7cfaa0e8ee6954ab190abeb14e7f69c1c was independently fetched and verified completed/success.

Supplemental FlyGraph evaluation will use the SAME prefix `Give only the final integer. Do not explain.\n`,same40cases,greedy64newtokens,no tools. Qwen uses its native no-thinking chat convention; FlyGraph uses its native plain completion convention. Equal token caps are not equal FLOPs or identical tokenizers. Qwen Q4_K_M non-thinking is not its full-precision/thinking ceiling.

The old8tokenQwen smoke was completely truncated and is excluded, not scored as zero intelligence. This64tokencheck is post-smoke/supplemental and cannot replace the primary400case experiment. External Qwen outputs are evaluation only, never training data.
