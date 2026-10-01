# To-do: ablations and experiments

Open experiments for the corruption-detection paper. Newest additions go at the end of their section. The method is the dev log's 2026-10-01 (night) entry. Its 5,000-image confirmation is in `docs/superpowers/plans/2026-10-01-content-conditioned-confirmation.md`.

**Rule for every ablation.** The headline is frozen: the two-axis score with a 2,000-image COCO-train bank, k = 50 neighbours, the stage-4 key and stages 1–3. Ablations are reported as ablations, on all 5,000 val images. If an ablation suggests changing the headline, the changed version needs its own check on images it was not chosen on.

## Ablations requested by the user (1 October)

- [ ] **Bank size: how many clean images the bank needs, up to 5,000** (user: "you can go for up to 5000").
  - Score the two-axis row (and the level and peak-share rows) with banks of 100, 250, 500, 1,000, 2,000 and 5,000 clean COCO-train images.
  - The smaller banks are random subsets of the 5,000-image bank, with several seeds per size. The 500 z-statistics images stay fixed and disjoint from every bank.
  - **Cost:** the 5,000-image bank needs the channel means and top-1% means of 3,000 more clean train images. Draw them disjoint from the existing calibration, bank and z-statistics splits. That is a short GPU pass: one forward each, no corruptions. The rest is CPU, once the 5,000-image val pass is complete.
- [ ] **k: how many of the most similar clean images to compare with.** Do this after the bank-size ablation, because its upper end depends on that result (user).
  - Score with k = 1, 5, 10, 20, 50, 100, 200 and 500, then upwards to the size of the bank the first ablation favours: up to 5,000 if the 5,000-image bank wins.
  - With k equal to the bank size, the neighbours' mean becomes the mean of all clean images, so the top of this curve meets the next ablation.
  - **Cost:** CPU only.
- [ ] **No content matching: compare with the average of all clean images.**
  - Replace the 50 neighbours' mean with the mean over the whole bank, for the level, the peak share and the two-axis score.
  - The level version already exists as a row ("Level vs the average of all clean images"). The peak-share and two-axis versions are still to add.
  - This measures how much of the gain comes from comparing with similar scenes rather than from the peak share itself.
  - **Cost:** CPU only.

## Other open experiments (from the dev log and the detection roundtable)

- [ ] **Driving data, the deciding test.**
  - A Cityscapes-train clean bank (2,000 images plus 500 for the z-statistics), with the headline row unchanged.
  - Scored on Cityscapes-C (val), Foggy Cityscapes (fog densities 0.005, 0.01 and 0.02, real-scene haze) and ACDC (fog, night, rain, snow).
  - **Needs:** downloads with the user's Cityscapes account, a written plan, and an agreed GPU window.
  - The roundtable's kill test: the peak share must still fall under real fog at s1, and the headline must not lose to the level score there.
- [ ] **Content key.** Stage 4 (current) vs stage 3 vs stages 3 + 4. On the 200 screening images, other keys gave no gain.
- [ ] **Scored stages.** Stages 1–3 (current) vs adding stage 4 or the stem output (C1).
- [ ] **Peak fraction.** Top 1% of positions (current) vs 0.5%, 2% and 5%, and the 99th percentile. The 99th percentile gave 0.886 / 0.853 on the 200 screening images.
- [ ] **Combining the two arms.**
  - The larger of the two (current) vs their sum.
  - The roundtable's C4 variant: a rectified flattening arm, which counts only channels that got flatter. It has screen numbers only (0.931 / 0.871).
- [ ] **A second detector or backbone**, for example RT-DETRv2-R50, to show that the result is not specific to R18.
- [ ] **Runtime.** Milliseconds per image for the top-1% means plus the neighbour search, next to the detector's 5.9 ms.
- [ ] **False alarms at a fixed threshold.** Set the threshold on clean train images and report the false-alarm rate on clean val. The roundtable's check gave 0.088 / 0.047 / 0.012 at α = 0.10 / 0.05 / 0.01.
- [ ] **Weak families: brightness, saturate and spatter.** Brightness is about 0.53 at severity 1. Check whether the stem output or the input image itself carries what the detector's features ignore.
- [ ] **Position on driving data.** A coarse grid of channel statistics could matter for a fixed camera (sky at the top, road at the bottom), unlike on COCO.
