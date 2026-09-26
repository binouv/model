# G4S: complete the already registered prefix-hybrid mechanism test

This finite recovery executes the exact model/data/run.py published on a9b355e9c8d13f77d8c8616f470132661f4cb53b. No checkpoints for this G4S protocol were found locally or in the checked releases and no active CI run existed at start. It does not repeat G3 or invent an optimizer for its model-only release.

One primary architecture hypothesis: bidirectional visibility inside a fully observed prompt improves a persistent-delta + attention generator versus causal-only prompt processing. Same 469648 parameters, same initial seeds7501/7502/7503,800updates,batch16,data sampler75311. Original sources/configuration are unchanged. No new best-test selection or dataset modification. Train8192, validation400, held400 including40counterfactual pairs. Byte vocabulary259, unrestrained greedy generation, no solver at inference. These are small English synthetic mechanism probes, NOT300-800M models or Qwen parity.

A wrapper adds publication after each completed arm: actual native model+optimizer+torch/python/sampler RNG, safetensors, configuration and raw metrics. Every uploaded asset is downloaded and SHA256 verified before finalizing a prerelease. Draft until all six trainings finish. Non-force progress commits touch only this directory on a dedicated branch. No recurring workflow, paid runner or billing change.

Latest G3/Qwen audit: main NEXT_STATE reports completed40uncensored Q4 thinking outputs (34/40 strict integer strings,40/40 posthoc final-integer parse), and real G3 seed7301 model-only safetensors release396974545. The incomplete BF16 thinking run is not scored. Do not equate this older40case fixture to the different G4S held set.

The longer experimental driver built locally was NOT uploaded as its first compressed transmission failed byte-hash verification; that unreferenced blob is excluded. This CI executes the already-published G4S trainer instead. It saves native checkpoints at the completed800step endpoints, not periodic200step resumes. Local additional core tests passed15cases; CI independently runs original tests and checkpoint reload checks.
