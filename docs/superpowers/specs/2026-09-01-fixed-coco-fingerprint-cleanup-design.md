# Fixed COCO fingerprint experiment cleanup

## Goal

Replace the current research framework with one understandable experiment for image-corruption detection. The reference bank is built from clean COCO training images. The frozen detector is then evaluated on separate COCO validation images under clean conditions and severity 4 and 5 for all 19 corruption families.

Only the empirically strongest fingerprint method remains. Maximum-confidence and Shannon-entropy scores remain as baselines. The implementation is for one researcher running one process, not for concurrent or adversarial filesystem use.

This design supersedes the earlier final-run plan that split COCO validation images into reference, selection, and validation cohorts.

## Command

Keep one command, `python -m differential_uncertainty benchmark-coco`, with these required experiment inputs:

- `--checkpoint`
- `--coco-train-images`
- `--coco-val-images`
- `--reference-count`
- `--evaluation-count`
- `--output`

Device and batch size remain ordinary optional runtime flags. A single optional `--seed` defaults to 44. Both image-count flags are required, so a small pilot and a large run are always intentional.

Remove the legacy `run` command and the generic manifest-driven workflow. COCO annotation files are no longer needed because the retained bank does not distinguish matched and background queries.

## Fixed data flow

At startup, set the Python, NumPy, and PyTorch random seeds from the one seed value. Enumerate each COCO directory, sort its image paths, shuffle deterministically, and select the requested number. Training images are used only for the bank; validation images are used only for evaluation.

For every selected clean training image:

1. Run the frozen RT-DETR extractor and compute the layer-2 persistence vectors.
2. Remove padded detector queries.
3. Keep queries whose maximum sigmoid confidence is at least 0.5.
4. Feed those vectors into Algorithm R reservoir sampling.

The final bank contains 2,000 persistence vectors. The capacity, confidence threshold, layer, and seed default are fixed constants, not search dimensions.

For every selected validation image, evaluate the clean image and severity 4 and 5 for all 19 corruption families. Levels 1 through 3 are not generated. For each clean/corrupted family triplet, remove the union of padded query positions before calculating all three scores.

The retained image scores are:

- Fingerprint: for every query, take the mean cosine distance to its five nearest bank vectors; combine query distances using confidence-weighted averaging across the scene.
- Confidence baseline: one minus the largest sigmoid confidence.
- Entropy baseline: normalized Shannon entropy of the highest-confidence retained query.

There is no bank search, distance search, aggregation search, learned pooling, matched/background bank, k-means, selection split, seed sweep, or confidence-conditioned diagnostic.

## Simple restart support

The output directory contains a small `run_config.json` with the supplied paths, counts, seed, and fixed method constants. An existing file must equal the current arguments; otherwise the command stops and asks for a different output directory. There are no source hashes, checkpoint hashes, dependency snapshots, or provenance graphs.

Bank construction writes one replaceable `bank_progress.pt` containing the processed training-image position, reservoir, eligible-query count, and random-generator state. Once complete, it writes `bank.pt` and removes the progress file.

Evaluation writes one small score file per completed validation image. Restarting skips those images. A temporary file followed by a rename prevents an interruption from publishing a partial result. The program supports one writer only; it has no locks, leases, concurrent-writer registry, directory descriptors, transactional bundles, or cache-authentication framework.

## Results

After all validation images finish, write:

- `per_image_scores.csv` with the retained image-level scores;
- `results.csv` with fingerprint, confidence, and entropy AUROC for each corruption separately at severity 4 and severity 5;
- `summary.json` with the same aggregate results and paired fingerprint-minus-baseline bootstrap intervals;
- `report.md` with all per-corruption values and a short plain-language interpretation;
- `corruption_auroc_bars.png` with separate severity-4 and severity-5 panels.

The bar chart is the final element of the Markdown report. Aggregate values supplement rather than replace the 38 per-corruption results. The report does not claim that the fingerprint beats a baseline when the paired interval includes zero.

## Code cleanup

Rewrite around the fixed path instead of preserving the old framework:

- Keep and simplify detector extraction, persistence computation, corruption generation, fixed bank construction, fixed scoring, AUROC evaluation, reporting, and the CLI.
- Remove generic artifact stores, manifest fingerprinting, provenance validation, filesystem race defenses, transactional report publication, experimental policy-selection code, and the legacy six-radius Gaussian-blur-only workflow. Gaussian blur remains one of the 19 fixed corruption families at severity 4 and 5.
- Stop saving detector boxes because query-to-annotation matching is gone.
- Delete tests whose only purpose was to protect removed behavior. Replace them with focused tests for deterministic image selection, reservoir resume, fixed scoring, required CLI flags, severity coverage, per-corruption results, and the final chart placement.
- Keep historical result documents as records; they are not executable interfaces.

## Errors and verification

Validate only failures that would otherwise waste a long run: missing directories or checkpoint, non-positive counts, requested counts larger than the available images, too few eligible bank vectors, malformed restart files, and non-finite model outputs or scores. Error messages name the offending input and stop before mixing results.

Verification uses small synthetic unit tests and a tiny end-to-end fixture, followed by the full retained test suite. It does not run thousands of COCO images during automated tests and does not recreate exhaustive tests for deleted defensive machinery.
