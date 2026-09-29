# FlyGraph G9Q — query-conditioned hard-negative continuation

Registered paired continuation from the **actual G8C trace→final native checkpoints**, not a fresh model and not a 300–800M scale claim. Each ancestral seed 7601/7602/7603 is cloned into two arms. Main uses the sibling answer from the same mutable-memory state as a hard negative; control uses a same-length answer from another state. Both score four prompt-answer sequences per pair and use the same positive NLL, optimizer state, data, batches and update count.

Primary inference metric is free generation from prompt strings only. Candidate answers and gold are training/evaluation records and are not supplied during generation. Fresh values 300..999 make every canonical G9 group disjoint from known G5S/G8C generated values <=255; train/validation/held groups are also disjoint before rendering.

This is a small mechanism probe (469,648 parameters), intended to decide whether query grounding deserves transfer into the future 300–800M RU/EN/code backbone. It is not a Qwen comparison.
