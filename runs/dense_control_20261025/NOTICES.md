# Provenance and scope

Model implementation is an adaptation of the user's existing local FlyGraph components. SOURCE_PROVENANCE.json pins the origin. No external model code or weights were copied; no new blanket license for the repository is assigned or inferred. PyTorch/safetensors are runtime dependencies under their own terms.

Architecture ideas are referenced in README_RU.md. MobileLLM's FAIR Noncommercial Research implementation/weights were not imported. SmolLM2/Qwen/MiniCPM citations identify design/training ideas, not reused binaries and not measured baselines.

External numerical fixtures are original OpenBookQA official dev records by Todor Mihaylov, Peter Clark, Tushar Khot and Ashish Sabharwal (2018), Apache-2.0: https://github.com/allenai/OpenBookQA . They were transported in OpenCompassData-core-20240207 release0.2.2.rc1: https://github.com/open-compass/opencompass/releases/tag/0.2.2.rc1 . Exact archive/member and local file hashes, IDs and role are in data/manifest.json. No question or gold answer was generated for this scaffold.

The inherited tokenizer is included for an identical-vocabulary engineering control. No pretrained backbone weights or Alpaca text are copied into this directory. Tiny saved weights are UNTRAINED random initialization; no optimizer steps or benchmark quality claims apply.
