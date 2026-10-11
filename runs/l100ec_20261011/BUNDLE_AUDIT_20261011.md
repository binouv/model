# L100EC runnable distribution audit — 2026-10-11

Completed **artifact packaging only**, zero new optimizer steps. The available **curriculum** BF16 checkpoint, with original 100,028,328-parameter architecture and 8192-token tokenizer, was packaged into a runnable local ZIP. Source archive: `FlyGraph_L100EC_Runnable_Source_2026-10-11.zip`.

- Output: `FlyGraph_L100EC_Curriculum_100M_Runnable_2026-10-11.zip` (160,318,030 bytes); SHA256 `a518d70fe05abb738d8d3e2cd3007c84e43fdfb9e13b495c3630c8d1d0e4260e`.
- Included checkpoint: 200,069,384 bytes; SHA256 `07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25`.
- Removed unavailable shuffled-arm entry and stale two-arm manifest; updated CLI arm choices and README.
- **pytest 12 passed, 0 failed**; strict hash-verified CPU model load and inference succeeded; ZIP integrity checked.
- Prompt `Calculate 5 + 8. Answer:` yielded `2\n    x += 5\n    ` at eight decoding tokens; **incorrect arithmetic**, no EOS.
- The binary archive is a **conversation artifact**, **not** uploaded to GitHub/GitHub Release. No general model-quality improvement or Qwen superiority claimed.
- L100QL primary arm was previously reported 512 steps complete, but neither its trained weights nor the 512-step control final metrics were verified as available in this active runtime; thus **L100QL is not a completed paired result**. L100CS/L100V untouched.

## Reproduce

Extract ZIP, then in `flygraph_l100ec_runtime`:

```bash
pip install -r requirements.txt
pytest -q tests
python src/infer.py --arm curriculum --prompt "Calculate 5 + 8. Answer:" --max-new-tokens 8
```
