# L100S: completed local span-reader experiment

Two 100,823,594-parameter variants completed 1000 reader-only updates and64 full-backbone updates each. All135 backbone parameter tensors changed; the two arms used the same batch sequence. These are learned extractive memory readers attached to the existing generative backbone, NOT general Qwen-level language models.

| Arm | IID answers/400 | IID complete query pairs/200 | Surface/200 | Long/200 | Renamed/200 | Original LM regression/120 |
|---|---:|---:|---:|---:|---:|---:|
| margin |131|16|48|1|21|0|
| control |146|22|50|0|19|0|

Primary pair-margin hypothesis NOT_CONFIRMED: -3pp, paired bootstrap95 [-7.5,1.5]. No Qwen superiority. Separate preregistered Qwen BF16 thinking reference: run38088594524; not complete at this publication.

Run `python runs/l100s_20261011/restore.py` to restore exact executed source, configs, tests, data generator, reports and summary metrics. Source XZ SHA256 a955c5d59c69ca8615a4f9121103075f777a4f4331b86b4d539ad123a4f85085. Both uploaded source-part Git blob hashes were checked against local bytes.

`SPAN_COORDINATES.b64` preserves ALL2000 primary extractive predictions losslessly as start/end/exact in fixed dataset order; the selected token IDs and raw output text are reconstructed from the original prompt. Dataset generator/seed/hash manifests are in the source bundle. The full local Source_Data_Results ZIP also preserves all training log rows, all historical predictions and full dataset bytes: SHA2563a836066c7ce643e7945da389c60f2a415681ac04cf2042d070d019bc2f2eb4c,1367240bytes. An additional private owner-accessible Drive backup of this new artifact is file1TGIgOdHE5EX3FnF4C_5Quxs-12eMMIoo; no existing user file sharing was changed.

Native final weights, both optimizer states and RNG exist locally. Weight bytes are NOT in this Git commit and no model Release is claimed. Parent was verified original L100M BF16; missing L100A weight bytes were not invented. Subsequent L100SQ was preregistered from development results, before reading these held scores, and continues fixed margin endpoint rather than choosing the better control after test.
