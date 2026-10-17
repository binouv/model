# L100FC: source/data, preregistered before training

The byte-exact source archive is an **ordinary Git binary blob**, not a hash-only placeholder.

- `archives/L100FC_src_data_pretraining.tar.xz` — 147,021 bytes; SHA256 `ba760aa62b4478f8d85c6848cb0b0a7599642cdd20e7f22ad695b34395981a61`.
- Restore using `python runs/l100fc_20261017/restore.py`.
- Data split and fixed 512-step training schedule were determined before L100FC optimization, exact source is in the archive. The run starts from L100FB step128 **full native model+Adafactor/RNG**; its weights are not hidden in this source archive.
- This source publication is `IN_PROGRESS`, not a scientific completion claim. Actual checkpoint bytes/metrics are separate and not claimed uploaded by this commit.
- Pre-registration: `runs/l100fc_20261017/REGISTRATION.md`.

Do not run `train` to "resume" after it has completed; it intentionally refuses to overwrite existing native checkpoints. Source archive includes trained/held data, generator, tests and fixed schedule, not only a seed.
