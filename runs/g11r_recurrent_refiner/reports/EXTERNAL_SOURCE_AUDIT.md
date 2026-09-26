# External source audit

G11R uses no external code, weights or pretrained base. The only outside architectural inspiration is the shared-weight internal recursive-refinement idea from Less is More: Recursive Reasoning with Tiny Networks (arXiv:2510.04871) and its official SamsungSAILMontreal/TinyRecursiveModels repository. The official repository is MIT licensed.

FlyGraph does not import TRM code, checkpoints, datasets, loss implementation, ACT/halting machinery or pretrained parameters. This experiment implements a separate causal byte-generative model and tests only the high-level hypothesis that reusing the same small refinement block several times can improve effective reasoning depth.
