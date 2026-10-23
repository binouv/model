# Published data and reuse notices

This research continuation uses externally published examples only. Questions,
answers and rationales were not invented by this run. Prompt wrappers and label
serialization are explicitly recorded in `src/prepare_data.py`.

- Stanford Alpaca: Tatsunori Hashimoto's group, Stanford, 2023; repository
  https://github.com/tatsu-lab/stanford_alpaca, data revision
  `761dc5bfbdeeffa89b8bff5d038781a4055f796a`. Data are CC-BY-NC-4.0;
  the authors restrict resulting models to research use. Preserve attribution;
  do not infer that the Apache license of their code applies to the data.
  https://creativecommons.org/licenses/by-nc/4.0/
- OpenBookQA: Todor Mihaylov, Peter Clark, Tushar Khot, Ashish Sabharwal, 2018.
  https://github.com/allenai/OpenBookQA and
  https://github.com/allenai/OpenBookQA/blob/main/LICENSE (Apache-2.0).
  Only Main question/choice/answer records are used, without injecting auxiliary
  knowledge facts or test-specific rationales into training.
- PIQA: Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, Yejin Choi, 2020.
  https://github.com/ybisk/ybisk.github.io/tree/master/piqa
  The author's README specifies Academic Free License v3.0 (AFL-3.0):
  https://github.com/ybisk/ybisk.github.io/blob/master/piqa/README.md
  https://opensource.org/license/afl-3-0-php
- AI2 ARC Challenge: Peter Clark et al., ARC-V1-Feb2018, CC-BY-SA-4.0.
  https://huggingface.co/datasets/allenai/ai2_arc
  https://creativecommons.org/licenses/by-sa/4.0/
  ARC is used only for held-out transfer evaluation; no ARC training is performed.
- GSM8K: OpenAI, 2021, MIT license:
  https://github.com/openai/grade-school-math
  https://github.com/openai/grade-school-math/blob/master/LICENSE
  An additional fixed64-question official-test subset is used only after model
  selection. Its separate registration records the original member hash, seed,
  source indices, overlap checks and equal64-token inference budget.

The immutable transport for OpenBookQA, PIQA and ARC is the OpenCompass release
`0.2.2.rc1 / OpenCompassData-core-20240207.zip`:
https://github.com/open-compass/opencompass/releases/tag/0.2.2.rc1
Archive SHA-256:
`c259fa1b2f8f5c9ae2df58bb6cc3200e705e2945b24e896988006c546a572607`.
Transport does not change the original authorship or licenses. Exact member
hashes, source row indices/IDs, local split seeds and selected records are stored
in `data/manifest.json` and the accompanying JSONL files. Original records are
preserved inside each selected item; wrappers and added indexing are marked.

The resulting checkpoint is a research, noncommercial artifact because its
training includes Alpaca. This notice is not a claim that all upstream assets
have the same license or that unknown historical pretraining data were audited.
