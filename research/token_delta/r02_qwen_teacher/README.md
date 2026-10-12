# R02: real pretrained Qwen teacher → tiny latent text decoder

This is an experimental side project under `binouv/model`. It does **not** replace FlyGraph training and does not claim to beat Qwen.

## Core protocol
- Teacher: official **Qwen/Qwen3-0.6B** revision `167b8104f88905a951069f5f95f9776908da5f68`, frozen, fp32 on GitHub Actions CPU.
- Input to teacher: synthetic *question only*, no answer labels or short/detailed style.
- Feature: final layer's **last prompt hidden state** (1024 floats).
- Learned bridge: 1024→96→48, **one 48D latent per question**, style-agnostic.
- Tiny GRU renderer conditions on the latent and generates one of two lengths; no additional Qwen calls within realization.
- English scene triples: 168 train / 48 heldout. Additional 48 OOD paraphrases with verbs not used in training questions.
- Controls: matched-size small renderer using deterministic hash-BOW question encoding, and zero/shuffled latent at test-time.
- **Do not infer 99% of a future answer is precomputed.** Frozen Qwen feature extraction measures an information signal at the answer boundary, not availability of exact future token logits.

## Files
- [Registration](PREREGISTRATION.md)
- [Source (frozen teacher + train + evaluation)](src/real_qwen_latent.py)
- [Unit/reload tests](src/test_real_qwen_latent.py)
- [Frozen configuration](config.json)
- [GitHub Actions workflow](../../../.github/workflows/token-delta-r02-qwen-cpu.yml)
- [R01 small trained synthetic model](../README.md)

## Reproduce on CPU with Internet
```bash
python -m pip install torch==2.10.0 transformers==4.57.1 accelerate==1.11.0 huggingface_hub==0.36.0 numpy==2.3.5 safetensors
python research/token_delta/r02_qwen_teacher/src/real_qwen_latent.py \
  --out research/token_delta/r02_qwen_teacher/results --steps 800 --teacher-batch 4 --threads 2
cd research/token_delta/r02_qwen_teacher/src
python -m unittest -v test_real_qwen_latent
```

After a full run, the `results/` folder should include metrics, all predictions, Qwen-derived hidden vectors, and **actual tiny student weights plus optimizer states**. The original teacher weights remain at the pinned upstream model revision. Negative findings must be published unchanged.

## Scientifically meaningful next step
Replace artificial copying with held-out factual question answering, numeric reasoning, code and RU/EN instructions. Compare against teacher-only, student-only, short text planning, and exact speculative verification. Measure true token rate on CPU+DDR5 and RTX 3090 with context 2K, 20K, 100K: no extrapolation from one-step tiny decoder timings.
