# Request: four detector-free IQA baselines on COCO-C (2026-10-05)

The "paper" session sent this request, at the user's request, to the implementing session on 5 October 2026. It is kept word for word as the spec of `docs/superpowers/plans/2026-10-05-iqa-baselines-coco-c.md`.

---

From the paper session, at the user's request: please add four detector-free image-quality (IQA) baselines to the COCO evaluation. Write a plan first for the user to approve, before implementing anything or using the GPU.

The four baselines, each scored so that higher means more degraded:
1. NIQE (Mittal et al., IEEE SPL 2013).
   - Main row: its pristine model refitted on clean COCO train images.
   - Sensitivity row: the published default model. Both rows use the same test-image features, so the second costs almost nothing.
2. ARNIQA quality score (Agnolucci et al., WACV 2024), with the KADID-10k regressor, as chosen in docs/driving-benchmark-baselines-and-metrics.md. Score = -quality.
3. ARNIQA embedding + clean prototype, following Becker, Weiss, Hübner & Arens, "Self-aware object detection via degradation manifolds" (arXiv 2602.18394).
   - Score: 1 - cosine similarity between the image's ARNIQA embedding and the mean embedding of clean COCO train images.
   - Follow their description of this baseline, and ARNIQA's own embedding (return_embedding in github.com/miccunifi/ARNIQA).
   - It shares the encoder pass with row 2.
4. CLIP-IQA (Wang et al., AAAI 2023), zero-shot, with its "Good photo." / "Bad photo." prompts. Not CLIP-IQA+, whose prompts were learned from human ratings. Score = -quality.

Protocol:
- Clean reference: the same clean COCO train images as the other COCO baselines (all 118,287), for both the ARNIQA prototype and the NIQE refit. If the NIQE refit is too slow on all of them, use a seeded subset and say so in the plan.
- Test images: the 5,000 COCO val images in the seed-44 order, with the 5 folds and all 96 conditions. Make them with degradation_monitor.corruptions and check them against runs/coco/'s digests, as the four-detector shared pass does.
- Each model gets the corrupted image at its native size, through its own preprocessing.
- These models read the image, not the detector. Like DisCoPatch, one set of scores serves all four detectors' reports.
- Report:
  - the existing metrics and intervals: AUROC common and extra, AUPR and FPR95, by severity and by family, with the paired bootstrap (1,000 draws, seed 44);
  - the two-axis score minus each IQA row, with intervals, on all images and on the untouched ones;
  - a results doc under docs/ and its tables under docs/results/.
- Timing: ms per image for each model, with the timing stage's protocol (batch 1, card otherwise idle), next to the detector's 5.9 ms.
- Record what each model saw in training, for the paper's fairness statement:
  - ARNIQA's encoder (rows 2 and 3) was trained on synthetic distortions, including blur, noise, compression, brightness, colour and contrast, but no weather;
  - CLIP-IQA saw no distortions;
  - NIQE sees only clean images.
- Keep the design reusable for Cityscapes-C later, with the reference taken from Cityscapes train. Don't build the Cityscapes part now.

Constraints:
- pyiqa is not installed in the UE env. Install it, pin it in requirements.txt (tests/test_package.py checks the imports), and verify the model names before writing the plan.
- The GPU is shared with explore: ask it before each GPU stage, set a memory cap, and send "done" afterwards.
- Run tests on the CPU only (CUDA_VISIBLE_DEVICES=). runs/coco/ is read only. The golden and equivalence tests stay green.
- Work on a new branch in its own worktree, cut from fingerprint_bank at b54a196. Never push, and git add only the files a task names.
- Don't touch IV_2027_Yuchen/ (I'm writing the paper there) or docs/superpowers/plans/2026-10-04-cityscapes-c-evaluation.md (it stays uncommitted).
- Dispatch every subagent on Opus.

The user will review the plan in your session. Please message me (the "paper" session) with the plan's path when it's ready, and with the results doc's path and the headline numbers when the run is done, so I can put them in the paper.
