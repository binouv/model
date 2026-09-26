# FlyGraph G10AR — query-readout architecture screen

**Completed locally at the preregistered fixed step-200 endpoints. Registered hypothesis NOT confirmed.**

Six models continued from the exact released G8C verified-trace step-2000 parents (seeds 7601/7602/7603). The 469,648-parameter backbone was frozen. Each arm trained exactly 36,865 adapter parameters; total parameters were 506,513. Main: 4-head causal cross-token query-readout attention. Control: parameter-matched per-token 96→192→96 MLP. Both received the same positive-NLL examples, minibatch group sequence, update count and optimizer schedule.

| Fresh held metric | Query readout | MLP control | Delta |
|---|---:|---:|---:|
| IID individual exact | 10.583% | 10.750% | -0.167 pp |
| IID both-correct pairs | 0/200 each seed | 0/200 each seed | 0 pp |
| IID query sensitivity | 6.50% | 4.83% | +1.667 pp |
| IID other-key confusion | 10.417% | 10.917% | -0.500 pp |
| Surface individual exact | 8.000% | 7.167% | +0.833 pp |
| Extrapolation individual exact | 0.667% | 0.667% | 0 pp |
| Mixed non-memory exact | 0.000% | 0.167% | -0.167 pp |

IID exact by seed was 39/45/43 out of400 versus 40/46/43. The registered +3 pp gates for individual exact, both-correct pairs and query sensitivity all failed. Both-correct retrieval stayed zero in every seed. The readout reduced wrong-other-key confusion slightly and preserved the mixed suite within2pp, but it did not improve final addressing accuracy.

A local process interruption occurred during query-readout seed7602 after durable step100. Uncheckpointed work was discarded; training resumed from that exact native step100 model/optimizer/Python/torch/sampler-RNG checkpoint. The first100 logged minibatches and reconstructed sampler continuation were checked. It remains one seed, not an independent run.

All six final safetensors were reloaded; 16 generation cases/model matched. All six native step200 checkpoints reproduced the next update deterministically in two independent replays. Parameters were finite. Held sets never selected a checkpoint.

This rejects a generic post-backbone query-readout adapter at this scale. The result is narrow memory-mechanism evidence, not broad reasoning/code evidence and not Qwen parity. Next priority returns to the already preregistered fresh multi-family reasoning/code trace-curriculum line rather than extending this memory-only screen.
