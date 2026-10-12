# R03 — semantic bottleneck: real Qwen3-0.6B state → 172K student

**Status: completed and binary-verified.** This is a controlled English synthetic copying/composition task, **not** general-purpose natural-language reasoning.

## Main result

| Heldout criterion (48 questions) | Pretrained Qwen features | Hash-BOW |
|---|---:|---:|
| Both response lengths exact, new combinations | 48/48 | 48/48 |
| Both response lengths exact, unseen verb synonyms | **30/48** | **5/48** |
| Three facts correctly extracted, unseen verb synonyms, linear probe | **33/48** | **4/48** |

Pretrained teacher: frozen `Qwen/Qwen3-0.6B`, revision `167b8104f88905a951069f5f95f9776908da5f68`. Extracted from Qwen in completed R02; no new teacher calls in R03. Cached pretrained feature SHA256: `e0add6315af0ac9f344c90d7339881d8711b49da28576eaae8e5e05055f8ff96`.

## Files

- [Full Russian scientific report](REPORT_RU.md)
- [Preregistration and falsifiable controls](PREREGISTRATION.md)
- [Training: probes + supervised semantic bottleneck](src/probe_and_decode.py)
- [Unit / checkpoint reloading tests](src/test_semantic_bottleneck.py)
- [Demo: one latent → two outputs without rerunning Qwen](src/demo_inference.py)
- [Metrics, all predictions, real student weights, SHA manifests](results/)

## Offline checkpoint demo (no network / no 0.6B weights needed)

Requires pinned R02 feature cache and the published trained R03 checkpoint already in this Git checkout.

```bash
python -m pip install torch numpy
python research/token_delta/r03_semantic_bottleneck/src/demo_inference.py --index 0 --kind ood
python research/token_delta/r03_semantic_bottleneck/src/demo_inference.py --all --kind ood
```

The demo loads **precomputed Qwen hidden vectors**, verifies their SHA256 and then calls only the small neural student. The semantic representation is built **once** and reused for both short and detailed answers. This is not a stand-alone 172K language model capable of answering arbitrary questions; new questions still require invoking the original Qwen teacher to obtain their latent features.

## Interpretation

The 1024D Qwen states had mean pairwise cosine similarity 0.99348, yet linear semantic probes extracted all three facts on 48/48 ordinary heldout combinations and 33/48 synonym holdouts. Direct language-only loss in R02 collapsed; explicit three-slot supervision restored generalization on this narrowly defined task. **Lossless VRAM/KV compression and distribution-exact decoding are separate hypotheses**, not claimed results here.
