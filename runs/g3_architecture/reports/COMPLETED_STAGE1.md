# G3: nine completed small generative architecture trainings

Three architectures x seeds7301/7302/7303, each2000AdamWupdates,batch32,64000presentations. Exactly3,471,293input/321,474supervised tokens per run. All nine batch-sequence SHA256 values match838069c1ce9a0a56bb1284427d235ebbab0a9c41368773b7ef0e09d5aef4b1e5. Train16384syntheticcases,16047unique presented. These are1.48-1.90Mprobes, NOT new300-800Mmodels or broad pretraining.

| Model | Params | Mean IID | Mean surface | Mean extrapolation | Both-correct counterfactual pairs |
|---|---:|---:|---:|---:|---:|
| full |1901696|7.2917%|2.0833%|0%|0%|
| shared |1475200|6.4583%|2.9167%|0.4167%|0%|
| hybrid |1903760|9.1667%|1.6667%|0%|0%|
| hybrid reset |1903760|2.7083%|0.4167%|0.4167%|0%|

Both preregistered hypotheses NOT_CONFIRMED_IN_THIS_PROTOCOL. Shared IID difference -0.833pp. Hybrid +1.875pp versus >=5pp required, descriptive crossed-bootstrap95% [-2.083,+6.25]pp; pooled surface/extrapolation difference -0.208pp. Three seeds share the SAME400unique cases, not1200independent examples. Counterfactual uncertainty resamples40pairs, not80independent members.

Reset confirms trained hybrid state affects output, not superiority to attention or persistent cross-episode memory. Reset is a mechanistic inference intervention, not compute-matched retraining. Generic parallel_delta usesT-by-Ttriangular solve; NOT a fused linear-time training kernel. Allmodels execute4blocks but have different FLOPs and parameters. Shared has2unique blocks, not trained adaptive depth.

13engineeringtests passed; all9finalcheckpoints/data/logs verified. All9local exports reloaded,90prediction checks matched stored outputs. A SEPARATE clean-source CPU replay of seed7301 trained3models and published real safetensors release assets; those weights are not claimed identical to local-container weights or independent seeds. See release receipt.

G3L was preregistered BEFORE held inspection, selecting hybrid by mean validationNLL .752851 vs shared .770511. It and full continue6000additional steps on all3seeds with actual optimizer/RNG; a learning-rate restart is part of this new regime. Do not rewrite this original fixedstep2000result. G3L is in progress at this publication. Qwen actual-thinking Q4/BF16 references are separate workflows; no result claimed until complete and censoring-checked.
