# R04-H1c — Does the head compute its answer progressively?

**Status (2026-10-12):** hypothesis registered; 13/13 CPU tests passed; complete GPU runner published; **no GPU H1c results**. R04-H1 stays NOT_CONFIRMED; H1b remains an independent diagnostic.

## Core separation
- Readout-only: extract two answers from the *last prompt hidden state* of frozen Qwen3.5-4B, without additional steps.
- Latent recurrence: a small, trained, tied-weight recurrent block performs 0, 2, or 8 **additional computations**, then emits both digits. This is a **new task-specific solver**, not a passive hidden-state parser.
- Controls: matched lexical and structured input versions with the same recurrent core, plus shuffled train labels. If an input-only solver matches Qwen, no pretrained latent-computation advantage is established.
- Adaptive stop: use a threshold chosen using validation data, report actual accuracy and number of latent steps; the current implementation skips future GRUCell operations for already-complete examples (verified against the full trajectory). Actual end-to-end latency improvements are NOT established.
- Later writer stage: Qwen3.5-0.8B receives only the finished plan, with no user prompt or explicit verbosity flag. This is NOT implemented/verified by H1c.

See [preregistered gates](PREREGISTRATION.md). All results from the prior failed H1 remain unchanged.

## Dataset and commands
New independently locked 1700 two-digit arithmetic / Python-output cases, 1200 train / 200 val / 300 test, seed 20261014, no overlap with the 1700 H1 cases or 128 H1b cases. Canonical SHA256:
`ac764eed8a0e06486dee8452219293941e208618fd95cbfa7495395f71184af7`

The published GitHub directory is canonical: all tests, extraction/training scripts and locked dataset are committed. The ZIP is an earlier backup and does not contain the real-early-exit fix. Under `D:\AI\model_r04\research\token_delta\r04_head_decides`:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s h1c_latent_compute\tests -v
& $py h1c_latent_compute\src\dataset.py --out h1c_latent_compute\data
& $py h1c_latent_compute\src\extract.py --data h1c_latent_compute\data --out h1c_latent_compute\cache --batch 4 --shard 50
& $py h1c_latent_compute\src\train.py --data h1c_latent_compute\data --cache h1c_latent_compute\cache --out h1c_latent_compute\results --device cuda --steps 600 --batch 96 --lr 0.0015
```

**Source-publishing status:** complete R04-H1c code and locked 1700-case dataset are now in `binouv/model`, with 13 passing CPU engineering tests. There are **no completed Qwen GPU scientific results yet**. Real adapter weights, optimizer state, teacher features and metrics will be published only after the operator's verified GPU run. The GitHub version supersedes the original ZIP.

## Related research
- [Latent Recurrent Thoughts](https://arxiv.org/abs/2609.01117): frozen LLM with an auxiliary recurrent latent processor.
- [LOTUS](https://arxiv.org/abs/2606.31779): looped latent states supervised with reasoning steps.
- [Thinking States](https://arxiv.org/abs/2602.08332): reasoning while reading input.
- [RecurTrace](https://arxiv.org/abs/2609.03379): adaptive latent stopping.
