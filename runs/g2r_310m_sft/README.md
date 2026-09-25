# FlyGraph G2R-310M — completed paired SFT

**Two full-parameter 310M training arms completed. Hypothesis NOT CONFIRMED in this protocol.**

Each arm: 309,993,253 trainable parameters; 256 optimizer updates; 49,716 input tokens; 1,024 presentations / 891 unique training examples. Answer-only supervised 5,139 response tokens; full-sequence supervised 49,716 tokens. Same G1 step192 BF16 bytes upcast to FP32, NEW Adafactor, seed7201/sampler72011, identical minibatches/LR/shapes. No architecture modification or external teacher. This is not broad pretraining.

| Primary test | G1 | Answer-only | Full-sequence |
|---|---:|---:|---:|
| IID | 0/160 | 4/160 | 4/160 |
| Unseen surface | 0/80 | 1/80 | 1/80 |
| Extrapolation | 0/80 | 0/80 | 0/80 |
| Counterfactual questions | 0/80 | 0/80 | 1/80 |
| Both-correct counterfactual pairs | 0/40 | 0/40 | 0/40 |
| All questions | 0/400 | 5/400 | 6/400 |

Full 8192 vocabulary, greedy 8 tokens, no tools; gold never enters generation. IID main/control delta0pp, paired2wins/2losses,p=1.0. Main/G1 delta2.5pp,p=.125. All preregistered effect gates failed. Lower response NLL does not establish reasoning: main teacher-forced numeric tokens15/79 versus whitespace/EOS115/120. This extremely small budget on an undertrained base does not refute response-only SFT in general.

## Actual Qwen comparison, explicitly supplemental

Verified external GitHub Actions run36186603496, commit450d46f7cfaa0e8ee6954ab190abeb14e7f69c1c: Qwen3.5-4B Q4_K_M non-thinking scored32/40. Exact fixture SHA256 and raw outputs were independently checked and rescored. Local checkpoints received the same forty prompts, integer-only prefix and64-token cap: G1 0/40, answer-only1/40, full-sequence3/40. Native tokenizers/conventions; NOT equal FLOPs. This post-smoke supplemental does not replace the primary400-case test. Earlier8-token Qwen truncations are excluded. No27B run or parity claim. Qwen outputs were never training data.

## Reproduction and actual raw data

`src/setup_data.py` reconstructs exact tokenizer/data from committed vocabulary bytes and deterministic code; all six JSONL hashes match G2. `metrics/PREDICTIONS_COMPACT.json.gz` contains ALL1,200 primary predictions for400 cases x3 models, losslessly encoded with70(text,tokenIDs,EOS) dictionary entries. Decompressed SHA256: e0b8293dd9c5001d787df8d490d5f95cc470a94a596ec61ca7e4e3f697674d9f. `python src/score_prediction_archive.py` verifies and independently scores them without loading weights.

`bench/SUPPLEMENTAL_RAW_TEXT.json.gz` contains all160 supplemental generated strings/EOS values (40cases x4models), in the exact fixture order. Decompressed SHA256: 4d951c0e7db90b01f42fcf2adbfb3445f8bd18deb30be92857fab2938e271b5a. These are native compressed data bytes, not weight files or model compression. Full per-case timings and training logs are in the local SourceData archive.

All18 engineering tests passed. Both final FP32 and BF16 models reloaded;10 fixed predictions per model/precision matched original FP32 outputs (40checks, not another400-case BF16 benchmark). All70 final weight shards were SHA256-verified; every parameter value was checked finite. FP32 optimizer/RNG saved. No training OOM/timeouts or partial aggregates in this completed run.

## Limitations and lineage

G2R repeats the same seed/data as parallel G2, NOT another independent training seed. Original G2step64 metadata was available at startup, but its weights were not; this run starts from actual G1 bytes. Separate namespaces are preserved. Repo ECC R52-R61 and conversation raw96/C1 numbered experiments are distinct lineages.

Some algebraically equivalent arithmetic problems cross the declared SFT semantic groups; some are also in the G1 corpus. Diagnostics flag these; primary data were not changed retrospectively. These are synthetic tests, not GSM8K/MBPP or a general intelligence score. The graph remains a synthetic within-token readout, NOT persistent inter-token memory.

## Publication status

Core source, exact tokenizer/data reconstruction, research, summaries and raw predictions are in Git. **Large FP32/BF16 weight bytes, optimizer files and all historical C1 archives are NOT all uploaded. No GitHub Release asset publication was performed here.** A manifest is not a weight upload. Local downloadable archives contain actual weights and complete source/data/raw results. Historical progress files are stage snapshots; final result files supersede them.
