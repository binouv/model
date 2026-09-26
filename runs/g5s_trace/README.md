# FlyGraph G5S — verified trace supervision screen

Preregistered paired mechanism experiment. Same 469,648-parameter causal hybrid, exact G4S semantic cases, three seeds, identical sampled case IDs and 1,000 optimizer updates. The only trained-objective difference is `verified_trace` versus `final_only` completion supervision.

`verified_trace` targets a deterministic independently replayable compact state trace followed by `F=<integer>`; the control predicts only `F=<integer>`. Inference receives the same problem prompt in both arms, uses greedy unrestricted 259-byte vocabulary generation, max128 bytes, EOS, no solver/tools/gold. Final accuracy is scored from a strict terminal `F=` field; trace exactness is reported separately and never substitutes for final-answer correctness.

Success criteria were fixed on main before training: >=5pp mean IID gain, positive pooled surface/extrapolation gain, <=2pp counterfactual pair regression, and positive IID direction in >=2/3 seeds. More trace target tokens mean more training FLOPs and are reported. This is a small English synthetic mechanism probe, not a 300–800M model and not evidence of Qwen parity.
