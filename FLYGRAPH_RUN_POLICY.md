# FlyGraph run policy

This file is the durable execution rule for ongoing FlyGraph research.

## Mandatory completion sequence

For every completed FlyGraph cognition run:

1. Execute exactly one primary cognition hypothesis and at most one confirming ablation.
2. Count only fully completed runs/episodes; never present timed-out or partial grids as results.
3. Save local downloadable artifacts:
   - implementation scripts
   - config/calibration files
   - JSON and CSV metrics
   - semantic/tests
   - Markdown research report
   - model/checkpoint artifacts when the run produces them
   - manifest/checksums
4. After local validation, commit/push the completed run to the connected GitHub repository `binouv/model` under `runs/rNN/`.
5. Push the report and model artifacts as part of the same completed-run publication workflow.
6. Do not wait for user confirmation between hypotheses. Move to the next hypothesis after the completed run is saved and pushed.
7. Keep cognition, reasoning, memory and hypothesis-selection stability ahead of audio/video/body work.
8. Do not resume 100M/200M/300M workspace scaling until memory uncertainty / trajectory selection is substantially stable.

If GitHub is temporarily unavailable, preserve the complete local run first and push it at the next available opportunity before treating repository synchronization as complete.
