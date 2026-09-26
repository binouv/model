# G9QR — reconstructed query-conditioned counterfactual contrast

Separate preregistered experiment from completed G8C. The earlier G9Q branch did not contain recoverable experiment source/data bytes or completed model/release at inspection time; G9QR uses new data/sampler seeds and is not claimed as completion of old G9Q.

Two arms branch from the exact G8C verified_trace model+AdamW state for ancestral seeds 7601/7602/7603. Architecture is unchanged. `query_contrast` adds sequence-level ranking of the correct queried value over another current key value; `nll_control` scores identical positive/negative candidates but optimizes positive NLL only. Fixed 600 updates. Fresh history groups are split before rendering; all available G5S/G8C groups are denied; G8C held is never training data.

469,648 parameters per probe. This can test a mechanism only; it is not a 300–800M model and cannot establish Qwen parity.