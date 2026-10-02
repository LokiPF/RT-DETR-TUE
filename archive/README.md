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

## `conv_tu/`: Topological Uncertainty on the backbone's conv layers (30 September – 1 October 2026)

- **What it was.** For each of the four stride-1 3 × 3 convs `res_layers[s].blocks[1].branch2a.conv`, a graph whose
  edge weights are the products of input activations and kernel weights. Its fingerprint was the top 1% of the
  graph's persistence diagram (a maximum spanning tree), compared with a clean bank by kNN.
  - **Code:** `graph.py` (the diagram), `features.py` (fingerprints and kNN), `tap.py` (the conv inputs with their
    folded kernels), `channels.py` (the follow-up's per-channel statistics), `pipeline.py` (the phases) and
    `report.py`.
  - **Results:** `docs/conv-tu-pilot-results.md`, `docs/results/conv-tu-pilot/` and `docs/results/conv-tu-pilot-channels/`.
- **Why it was retired.** The topology added nothing measurable. On the pilot's 200 images, its AUROC on the common
  families was 0.661, against 0.658 for the same number of heaviest edges without the cycle rule. The plain mean
  |activation| per channel reached 0.798.
- **What it led to.** That channel-means control became the current method in `degradation_monitor/method/`: each
  channel's level and peak share, judged against similar clean scenes.
- **Last run.** Branch `fingerprint_bank` at commit `53d9897`: the full suite passed there (296 passed, 2
  skipped).
