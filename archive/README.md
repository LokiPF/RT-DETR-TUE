# Archive: retired methods, read only

Nothing in this folder is imported, run, tested or maintained. The files are a snapshot, copied exactly as they last
were, so the project's earlier work stays readable next to the current code. To run any of it, check out the commit
named for it below, where its tests passed.

## `decoder_fingerprint/`: persistence fingerprints of the decoder queries (August 2026)

- **What it was.** Topological Uncertainty on the last decoder layer. It built a persistence fingerprint (335 sorted
  values) of the classification head for each of the 300 queries, kept a clean bank of fingerprints, and took
  each query's distance to its nearest clean fingerprints. A confidence-weighted mean turned those into one score
  per image.
  - **Code:** `benchmark.py`, `bank.py`, `scoring.py`, `persistence.py`, `reporting.py`, `config.py` and
    `evaluation.py`, the old `extraction.py` (its `RTDETRExtractor`), `corruptions/`, and the old `cli.py` (its
    `benchmark-coco` command).
  - **Data:** `pretrained_weights/` holds its class-wise Fréchet means.
- **Why it was retired.** On 100 held-out images at severities 4 and 5, its mean AUROC was 0.822. That equals top-query
  entropy (0.822) and is barely above 1 − max confidence (0.813), so it added nothing over the detector's own
  confidence.
- **Last run.** Branch `fingerprint_bank` at commit `53d9897`: the full suite passed there (296 passed, 2
  skipped).
