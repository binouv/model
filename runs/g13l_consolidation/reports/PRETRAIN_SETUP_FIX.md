# G13L pre-training setup correction

The first setup attempt failed before producing data and before any optimizer update because it included G13 schedule JSONL files that have no canonical field. The correction restricts prior-data denial to the six semantic G13 split files. Model, objective, seeds, parent checkpoints, held sizes, continuation schedule seed and success gates are unchanged.
