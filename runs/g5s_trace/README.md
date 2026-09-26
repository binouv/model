# G5S — verified trace objective, small causal hybrid

One preregistered hypothesis and final-only control, three seeds7601/7602/7603,1000updates x16cases,469648parameters each. Same case sequence and initialization within seed. Exact G4S byte-tokenized semantic cases, canonically disjoint across splits. Trace and final targets independently replayed. At generation the model receives only prompt strings and no solver/case/gold. The model is unchanged causal_hybrid with true inter-token delta state.

Main targets `T=<compact verified intermediate trace>;F=<integer>`, control `F=<integer>`. Both use the same prompt asking for F= and unrestricted byte259 greedy generation,max128newtokens. Extra trace tokens cost extra FLOPs at equal updates/examples; this is not compute-neutral. Small English synthetic probes, not300-800M,notQwenparity.

This namespace is separate from G5/G4R because those exact raw datasets were unavailable. `src/data.py` reconstructs the exact available G4S data; compare hashes in configs/preregistered.json. Local datasets, raw predictions, checkpointbytes and optimizer/RNG are retained. Git source publication is NOT a remote upload of model bytes.

Training is finite and checkpointed every200updates; completed1000step endpoints only determine results. No held-based checkpoint selection. Native final checkpoints contain optimizer and torch/python/sampler RNG. Safetensors is model-only. Checkpoints are reloaded and a next-update replay compared independently. See final status/metrics once complete; no partial aggregate here.
