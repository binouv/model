# L100EC curriculum BF16 — binary weights published

Three real binary Git blobs are present: 80,000,000 + 80,000,000 + 40,069,384 bytes. An independent HTTPS clone and complete reconstruction matched SHA256 `07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25`.

This is the inherited L100EC checkpoint, not L100QL; no new training was performed. Original optimizer/RNG were not recovered. See GITHUB_BINARY_RECEIPT.json for the binary commit and verification.

From repository root:

```sh
python runs/artifact_recovery_20261012/checkpoint_parts.py restore models/l100ec_curriculum_bf16_07e33eb2 model.safetensors
```

The recovery toolkit and inference runtime are also available in the conversation/private backup. Earlier pending-publication notes are historical, superseded by this receipt.
