# Roundtable summary: detecting corruption from a frozen detector's early channels

## 1. Bottom line

All five panelists ended up backing one method, and all five gave their first vote to A's merged proposal. For each early channel (s1–s3), the method takes the log ratio of the channel's strongest 1% of responses to its mean. That ratio is the "peak share", and it is compared with the 50 clean training images nearest in stage-4 features (Section 4's reference). The row was fixed on the 200 screening images. It then held up on two later, non-overlapping sets: 0.893 / 0.858 AUROC (common / extra) on 859 images and 0.901 / 0.862 on 911 more. That is about +0.08 / +0.05 over the activation CDFs, and +0.11 at severity 1. The panel now describes the idea on two axes. Corruptions that remove structure (fog, contrast, blur) flatten a channel's peaks, and noise-like ones shift its level. Taking the larger of the two scores (row M1) reaches 0.922 on common but only ties on extra. Three things are still open: which row leads the paper, which rule picks it, and whether the gain survives real fog. On a 24-frame driving test, the plain level did just as well.

## 2. The proposals

| Panelist (seat) | Title | Thesis in one line | Final status |
|---|---|---|---|
| A (Mechanism) | Shape, not level | Scenes set how much early filters fire; corruptions set how sharply | **Merged** with C and D as the lead. Now "peaks fall, bulk stays", with the level added back |
| B (Statistics) | Invisible per channel, obvious jointly | Whiten log channel energies by their clean covariance | Thesis **withdrawn**. Merged, contributing the reasoning for a one-direction test and a baseline |
| C (Topology) | Content scales, corruption flattens | One scale-free peak share per channel, compared with similar scenes | **Merged** (same row as A). Adds "opposite moves" and a challenger row, C4 |
| D (Vehicles) | Level and crest | Judge each channel's raw crest alongside its level | Statistic **withdrawn**; **merged**. Supplies the driving-data plan |
| E (Skeptic) | Aliased, not blind | Compared with the same scene, the detector sees mild corruption | Method **withdrawn**; **merged**. Its same-scene oracle stays as analysis |

## 3. The confrontation

No panelist drifted into predicting performance drop. Every proposal and check was scored on AUROC and FPR95.

**A: Shape, not level.**
- **Ratio artefact** (B, C, D): the clean "level–peakiness trade-off" (r = −0.74 to −0.88) is what any ratio shows against its own denominator. Under independence it would be −0.82 to −0.88. A conceded; the trade-off and K3's correlation clause are gone. **Landed.**
- **"Not the mean"** (E): the gain over the level comes from fog, contrast and zoom blur, and the level wins on 9 of 19 families. A confirmed this on 753 unread images and now reads both. **Landed.**
- **Real fog** (B, D): B predicted the share might rise under real fog. D's depth-fog test found the level tying the shape. A reran it: the s1 share still falls in 70–74% of channels, but by about half the shift seen with the package's fog, and s3 does not flatten. **D's point landed; B's predicted sign flip did not happen.**
- **Selection on the screen** (A's own worry) and **NAP not cited** (C, D, E): answered by held-out runs of the frozen row; NAP's published form scores 0.302 here. **Answered.**
- **Explaining away** (D, in Round 3, after A's last turn): A says fog lowers the stage-4 key, so the reference picks dimmer, peakier scenes. D measured this under severity-1 fog at s1:
  - the image moves −0.27 bank SD in level and −0.84 in share;
  - its neighbours move only −0.06 in each, so they are slightly flatter, not peakier;
  - even with the clean twin's neighbours, the level reaches only 0.545.

  **Landed, unanswered.** A's, B's and C's final write-ups still keep the claim.

**B: Invisible per channel, obvious jointly.**
- **Wrong geometry** (A, C, D): the flattening shift lies along a high-variance clean direction, which whitening divides away. Whitening the share costs −0.034 held out, and B only matches the CDFs on severity-1 fog (0.632 vs 0.634). **Landed.**
- **Per-channel control** (C): conditioned log means, scored channel by channel, trade evenly with B. B leads by 0.010 on common and trails by 0.013 on extra. **Landed.**
- **Novelty** (A, D, E): without the log, it is Rippel et al. 2020, which B had not cited. **Landed.**
- B's own pre-registered tests passed (+0.055 / +0.048 over the CDFs), so it remains the strongest baseline that needs no content key. B contributed a rule (O'Brien 1984): when all channels shift together, use a one-direction test; when the shift is scattered, use a whitened distance.

**C: Content scales, corruption flattens.** C's row matched A's to three decimals, so C merged.
- **"The ratio cancels content"** (A, B): the share is 60–69% predictable from the key. C had already withdrawn this in Round 2. **Landed.**
- **Averaged flattening points the wrong way** (D, E): C showed that only the average of raw ratios inverts (0.302); the log average gives 0.678. C's new explanation is "opposite moves": dense channels flatten while sparse ones sharpen. **Partly landed.** The new claim has not been reviewed.
- **Overclaim against magnitudes** (D): the share is 0.008 below Section 4 on extra. C conceded and registered C4, the larger of a rectified flattening score and the level. C4 has only screen numbers (0.931 / 0.871).
- **Topology:** three controlled tests, all negative. Nobody contested this.

**D: Level and crest.**
- **Raw crest beaten by its log** (all four): the log wins by +0.018 [+0.016, +0.019] on common, on 806 unread images. D also missed its own fog ≥ 0.75 bar three times (0.739 / 0.746 / 0.749), because adding a level that is blind to fog dilutes the crest. **Landed.**
- **Real fog raises the crest** (C): the driving test keeps the sign at s1–s2; only the size shrinks. **Partly landed.**
- **What survives:** the two-axis framing that M1 implements, the driving plan, and the correction of A's mechanism.

**E: Aliased, not blind.**
- **Aliasing depends on the statistic** (all four): the shape closes 59–67% of the fog, contrast and zoom gap from a single image. E added that the signed test leaves only 0.054 / 0.058 to the same-scene ceiling on severity-1 fog / contrast. **Landed.**
- **The oracle needs the clean twin, the "leak-free" key is also an oracle, and the deployable rows are the weakest** (B, C, D): E conceded, confirmed the key point in its own code, and withdrew both methods. The oracle survives as a measurement: using another image's key drops it to 0.707.
- **What survives:** frost, saturate and elastic sit 0.17–0.19 below their ceiling. Brightness and spatter are nearly invisible to the detector even with the twin (0.649 / 0.674).

## 4. Checks that were run

All checks ran on the CPU. The 5,000-image pass stores images in the seed-44 order, so each held-out check read a longer stretch of the same sequence. Two non-overlapping sets are clean for the rows frozen in Round 1:
- positions 200–1058 (859 images, A in Round 2);
- positions 1059–1969 (911 images, B in Round 3). A's 753 and D's 806 images are subsets of this and agree within 0.002.

Nobody has read position 1970 onward.

**Verified on positions 1059–1969 (911 images):**

| Row | Frozen | Common | Extra | Sev-1 | FPR95 common |
|---|---|---|---|---|---|
| M1 = max(signed flattening arm, level arm) | Round 2, after seeing 200–1058 | **0.922** | 0.861 | **0.869** | **0.216** |
| E4 = flattening + level, summed | Round 1 | 0.921 | 0.836 | 0.861 | 0.226 |
| AC = two-sided peak share (A = C) | Round 1 | 0.901 | 0.862 | 0.818 | 0.251 |
| C2 = log level + log share | Round 1 | 0.895 | **0.871** | 0.808 | 0.264 |
| B = log-whitened means | Round 1 | 0.885 | 0.861 | 0.790 | 0.276 |
| Section 4 (the idea to beat) | Before the roundtable | 0.842 | 0.868 | 0.763 | 0.358 |
| Activation CDFs | Stored scores | 0.819 | 0.808 | 0.705 | 0.417 |

- **AC against the baselines:**
  - AC − CDFs: +0.082 [+0.075, +0.089] common, +0.053 [+0.048, +0.059] extra, +0.113 at severity 1.
  - AC − Section 4: +0.058 [+0.052, +0.065] common, −0.006 [−0.011, −0.001] extra.
- **M1:**
  - M1 − AC: +0.021 [+0.015, +0.027] common, −0.001 [−0.007, +0.004] extra, +0.051 at severity 1.
  - M1 − CDFs: +0.103 common, +0.052 extra, +0.164 at severity 1.
- **Severity-1 fog / contrast:** M1 0.959 / 0.947, AC 0.807 / 0.829, Section 4 0.517 / 0.531, CDFs 0.633 / 0.661.
- **Weak spots:** brightness (about 0.53 at severity 1), spatter and saturate stay weak for every row. Averaged over severities, M1 loses most there: brightness −0.075 and saturate −0.057 against Section 4, on 753 images.
- **First held-out set (200–1058):** AC gave 0.893 [0.884, 0.902] / 0.858 [0.850, 0.866]. Section 4 itself beat the CDFs there (+0.023 [+0.014, +0.031] common). This is partial evidence on the confirmation's own question.
- **False-alarm control (B):** thresholds set on 500 clean train images gave 0.088 / 0.047 / 0.012 false alarms at α = 0.10 / 0.05 / 0.01.

**Mechanism measurements** (on the screen or the clean bank unless noted):
- At severity 1, families that remove structure lower an s1 channel's log ceiling 4–14 times more than its log level (fog: −0.280 vs −0.061). Under them, 77–84% of channels flatten, against 52–53% under noise.
- The flattening is shared across channels. The direction in which all channels move together holds 12% of the clean variation but 40–59% of the shift under fog, contrast and the blurs. Whitening shrinks such directions, which is why it hurts.
- The share is nearly as content-driven as the level: the stage-4 key predicts 60–69% of its variation, against 69–77% for the level. That is why comparing with similar scenes still adds +0.048.
- **Same-scene oracle** (859 images, severity 1): at least 0.96 on 15 families; saturate 0.806, elastic 0.885, brightness 0.649, spatter 0.674.
- **Opposite moves** (C): under seven visibility-loss families at severity 3, the densest fifth of channels flattens by 0.93–1.98 SD and the sparsest fifth sharpens by 0.02–0.69. This holds in 14 of 14 cells each way.
- **Negative results:**
  - topology: the Euler characteristic adds +0.004 [−0.016, +0.022] over the means;
  - comparing small image cells instead of whole images: −0.112;
  - adding B's score to A's: −0.002;
  - alternative content keys: no gain.
- **Driving tests:** 12 and 24 Cityscapes frames, with fog synthesised from a depth map guessed from the labels; each AUROC is about ±0.08.
  - With a Cityscapes reference at β = 0.005 / 0.01: the level gives 0.792 / 0.908, the two-sided share 0.776 / 0.894, the signed flattening arm 0.760 / 0.814, and an M1-style max 0.826 / 0.920.
  - With the COCO bank, every content-conditioned row fell to 0.54–0.80.

**Chosen or tuned on the 200 screening images:**
- every Round-1 statistic: A chose its statistic after seeing Section 4 fail on fog; B screened about 40 variants, C about 7 and D 10;
- E4, registered in Round 1 but chosen after looking at the data;
- M1, chosen after seeing the 200–1058 numbers but frozen before anyone looked above position 1058;
- C4, which has only screen numbers.

**What I verified myself:**
- File times confirm that A's, B's and D's pre-registrations came before their scoring. M1 was frozen at 17:27, and the first look above position 1058 came at 17:37.
- C's Round-3 pre-registration file dates from 17:56, after its held-out checks of rows that were already frozen.
- The log numbers I sampled match the reports.

## 5. Votes

| Voter | First | Second | Reason given |
|---|---|---|---|
| A | A | B | Two held-out passes; B is the best independent fallback |
| B | A, as M1 | D | M1 is best on fresh images; D gave the two axes and the fog kill test |
| C | A | E | All criteria pass; E's oracle measures what remains |
| D | A | C | A holds the primary and the best fresh row; C has the same row and the best story |
| E | A, if it uses two arms (M1 or LS) | D | The shape alone loses on 9 families; D's structure survived family by family |

**Tally:** A has 5 of 5 first choices. Second choices go to D (2) and to B, C and E (1 each). C and D merged into A, so their first votes are partly votes for themselves. The unanimous vote also hides a split over the headline row.

## 6. Consensus and live disagreements

**Agreed:**
- **The method family:** the per-channel log peak share at s1–s3, compared with clean scenes of similar content, with the channel level as a second axis.
- **Withdrawn claims stay withdrawn:** the trade-off, "the ratio cancels content", "shape, not level", "invisible per channel", "one image cannot tell haze from fog", the raw crest, and "level monitors fail on real fog".
- **Settled questions:** whitening is the wrong tool for the shape, topology does not earn its place, and NAP is cited and beaten.
- **Limits and next test:** brightness, saturate and spatter stay weak (L10). The bank must come from the deployment camera. Foggy Cityscapes, with clean and foggy versions of the same scenes, is the decisive driving test.

**Still disputed:**
1. **The headline row and its rule.** AC is the frozen primary. B and E back M1, though E compares it with the summed LS row rather than with AC. C registered C4, and LS / C2 has the best extra. A's frozen rule currently hinges on a −0.001 extra difference, and B rewrote the rule after seeing the interim numbers.
2. **Why the level misses the package's fog.** A, B and C credit key drift. D's check and E's split say the level simply barely moves.
3. **Signed or rectified flattening.** C says a signed average lets sharpening sparse channels cancel flattening dense ones. A, B and E use the signed average anyway, and that arm alone scores 0.969 on severity-1 fog on the screen. C4 beat M1 by +0.010 on the screen, but C4 was chosen on that screen.
4. **The real-fog kill test.**
   - B kills the shape claim if the flattening arm alone fails to beat the level arm. On the driving test it fails (0.760 vs 0.792, and 0.814 vs 0.908).
   - D and E compare the primary row or M1 with the best level row.
   - A tests the sign of the s1 shift.
5. **Paper thresholds.** B raised A's: common 0.90 instead of 0.87, severity 1 0.85 instead of 0.77, FPR95 0.25 instead of 0.30, and fog and contrast at 0.90. E set its own thresholds, including a win over the CDFs on all 19 families.

## 7. What the panel recommends next

**Leading direction:** the merged A + C + D paper. Its thesis: content raises a channel's peaks along with its bulk; corruptions that remove structure drop the peaks, and those that add energy move the bulk. M1 leads the paper if it passes the rule, and AC otherwise. B's whitened means serve as the baseline and E's oracle as the analysis. **Runner-up**, if the extra families matter more: the summed LS / C2 row (best extra, about 0.87).

**First experiment (COCO; the scoring scripts already exist):** finish the 5,000-image pass, then score the frozen rows on the CPU, which takes minutes.
- **Report three ranges separately:** positions 200–4999, 1059–4999 (for the M1 decision) and 1970–4999. The pass's own pre-registered Section 4 decision comes from the same run.
- **Kill criteria:**
  - If AC − CDFs on common has an interval containing 0, there is no claim over the CDFs (C and D also check extra).
  - If AC − Section 4 on common is below +0.015, or its interval touches 0, the shape adds nothing and Section 4 stands.
- **State of the pass:** at 18:02 it was paused at 2,120 of 5,000 images. Its GPU slot ended at 17:48 on its own time limit, and nothing is running now. The log estimates about 96 GPU-minutes to finish. The confirmation plan's next full GPU gap is 22:00–23:50.

**Next, in order:**
1. **Foggy Cityscapes.**
   - **Setup:** a Cityscapes-train bank (2,000 images plus 500 for z-statistics), the row fixed by the COCO run, and Cityscapes-C on val.
   - **Cost:** one backbone-only pass of about 40–90 GPU-minutes, plus the download. D says the download needs your Cityscapes account.
   - **Kill:** the primary row does not beat the better level row at β = 0.005 and 0.01 (D, E), or the median s1 shift of the share is ≥ 0 at β = 0.01 (A, B).
   - **Power:** with 500 pairs the interval is about ±0.02, so D suggests adding the 475 spare train images.
2. **ACDC** fog, night, rain and snow (to download), mainly to see whether night glare fools the one-direction arm.
3. **CPU checks on unread positions:** C's opposite moves, the share of flattening channels (A's K3), and C's ReLU explanation checked against the BatchNorm parameters.
4. **A second backbone** to show the result generalises (C, D), not yet scoped.

## 8. Decisions only the user can make

1. **The headline row and its rule, decided before the remaining images are read.** The options are A's frozen rule (M1 against AC), B's later non-inferiority rule, E's rule (M1 against LS), or C's challenger C4.
2. **Which images count as confirmation.** Either positions 1059–4999, which were unread when M1 was frozen but have been looked at since, or only 1970–4999: about 3,030 images nobody has read, with intervals of about ±0.005.
3. **GPU windows on the shared card.** About 96 minutes to finish COCO, then about an hour for Cityscapes, each agreed with the other sessions. The Cityscapes run also needs your approval of a written plan.
4. **Downloads:** Foggy Cityscapes and ACDC.
5. **Framing:**
   - whether "flattened or shifted" is elegant enough for you;
   - whether to accept the risk of a reviewer calling it "NAP with a log at early layers, plus Section 4";
   - whether the IV paper goes ahead if real fog only ties the level.

## Rapporteur's note

- **The strongest result is the replication, not any mechanism.** A row fixed on 200 images barely moved on two non-overlapping sets: 0.897 on the screen, then 0.893 and 0.901.
- **I would apply A's frozen M1 rule unchanged on 1059–4999.** The interim look did not change that rule, so it still stands. Report 1970–4999 as the never-read check, and treat B's and E's later rules as secondary. With the extra difference at −0.001 ± 0.006, the rule is close to a coin flip, so report both rows either way.
- **Drop the explaining-away story unless A rebuts D's measurement.** D's simpler account fits the data: the package's fog barely moves channel means but lowers their peaks.
- **Expect M1 and C4 to differ little.** The signed arm's low score on its own (about 0.70 on common) comes from noise, which sharpens channels on average, not from cancellation. Both M1 and C4 cover noise with the level arm.
- **The driving case is the weakest link.** The evidence is 12–24 frames with a guessed depth map, where the level tied the shape and the signed arm did worst. Run Foggy Cityscapes before writing the IV framing.