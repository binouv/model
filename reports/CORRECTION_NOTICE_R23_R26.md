# Correction notice — R23–R26 long-hop benchmarks

**Date:** 2026-09-22

R27 found a generator inconsistency in R23–R26 long-hop synthetic tasks. Repeated path keys could be overwritten in final memory while the stored target/trace still described the pre-overwrite path. The probability of a fully consistent example collapsed with horizon (R25/R26: 67.7% @8, 39.1% @16, 9.81% @32, 0.217% @64, 0% @128).

Therefore old absolute long-hop scores and interpretations that relied on them are **invalidated**. Historical files remain in `history/` for provenance. Corrected benchmark v2 and revalidation results are in `reports/RESEARCH_REPORT_R27_R30_RU.md` and `runs/r27/`.

The qualitative R23 protected-identity hypothesis was re-run on corrected data and replicated strongly; it is not being discarded.