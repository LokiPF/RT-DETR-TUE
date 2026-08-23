# Within-Image Corruption Contrast Experiment

This report explains a test of whether an image can provide its own reference point for detecting
blur. The experiment used saved results from 250 images. Each image had one clean version and five
increasingly blurred versions.

## Short answer

**The planned anchor idea was supported on the tuning, or practice, images.** An anchor is a group
of detector guesses that acts like a ruler inside the image. Seven versions of the planned method
passed the full checklist: the ruler was steady enough, the combined score beat both groups it
was built from, and it added something beyond the detector's confidence alone.

**The later differential idea did not pass its test.** It had the highest average score overall,
but it did not beat the two targets set for slight blur. It should not move on to the new,
untouched held-out test images.

These are two separate results. A high average score for the later idea does not cancel its failed
slight-blur test.

## The question we tested

An object detector makes many guesses about objects in an image. Some guesses have high
confidence and some have low confidence. Imagine sorting those guesses into groups from least
confident to most confident.

Earlier work showed that some groups change more under blur than others. This experiment asked:
can we compare two groups from the **same image** and get a better warning that the image is
blurred?

The first group is the **anchor**, or reference group. Think of it as a ruler drawn inside the
image. The second group is the **responsive group**, which we expect to move more when blur is
added. If the ruler stays fairly steady while the responsive group moves, the difference between
them may remove scene-to-scene variation and make blur easier to notice.

Two kinds of comparison were kept separate:

- **Planned anchor comparisons:** low-confidence groups were used as the ruler. These choices were
  made before reading the results of this experiment.
- **Later differential comparisons:** a very high-confidence group was compared with a middle
  group. These choices were suggested by earlier tuning results, so they had to pass tougher,
  pre-set targets before being allowed to continue.

## How the test worked

The experiment used 250 **tuning images**. Tuning images are practice images used to choose and
check ideas. They are not the new, untouched images used for a final test.

For every image, the study used six blur levels:

- level 0: clean;
- levels 1 and 2: slight blur;
- levels 3 to 5: increasingly strong blur.

The basic saved score was called **persistence distance**. It measures how far a detector's
internal fingerprint is from similar fingerprints collected from clean images. A larger distance
can mean that something is unusual. This experiment reused those saved distances; it did not
rerun the detector, rebuild the clean reference library, or search that library again.

Four ways of making a score were examined:

1. use the responsive group by itself;
2. subtract the anchor from the responsive group;
3. compare their relative difference, which adjusts for their size;
4. compare the responsive group with what a line fitted on other images would predict from the
   anchor.

Each new contrast had to face fair controls. It was compared with the responsive group alone,
the anchor alone, and a twin made only from the detector's confidence. The fitted-line method was
also cross-checked in five folds, so an image did not help fit the line used to score itself.

## Results at a glance

| Question | Result |
|---|---|
| How much data was used? | 250 tuning images at six blur levels |
| Was the planned anchor usually steady? | Yes: 28 of 30 anchor stability checks were below the limit |
| Did the anchor help explain differences between clean scenes? | Yes: all 6 planned anchor setups beat a constant guess |
| Did derived contrasts beat both raw groups? | 20 of 33 did |
| Did persistence always add more than confidence alone? | No: 8 of 45 persistence candidates failed that control |
| How many planned anchor candidates passed the full checklist? | 7 |
| Strongest planned anchor average ordering score | 0.648 macro AUROC |
| Strongest later differential average ordering score | 0.721 macro AUROC |
| Did any later differential candidate pass its slight-blur targets? | No |

## Result 1: The planned anchor idea worked on the tuning images

The first question was whether the low-confidence ruler stayed steady enough. The test compared
how far the ruler moved under blur with how different the clean scenes already were from one
another. A value below 1 means the ruler moved less than the clean scenes differed.

Among the planned anchor setups, 28 of 30 checks were below 1. The result was not perfect, but it
was steady often enough for some complete candidates to pass the full checklist.

The next question was whether the ruler explained normal differences between clean scenes. All
six planned setups beat a simple guess that ignored the ruler and always predicted the usual
middle value. That is useful because it shows the ruler contained real information about the
scene, rather than adding extra arithmetic with no benefit.

Seven planned anchor candidates passed every required check. The strongest used the lowest
10 percent confidence group as the ruler, the 50-to-60 percent group as the responsive group,
the mean scene summary, and a relative difference. Its technical name is
`decile_00_10__50_60 / mean / relative_gap`.

Its macro AUROC was 0.648. Its scores at blur levels 1 through 5 were:

| Blur level | AUROC |
|---|---:|
| 1 | 0.478 |
| 2 | 0.515 |
| 3 | 0.600 |
| 4 | 0.802 |
| 5 | 0.845 |

This table adds an important warning. The method was much better at finding strong blur than
slight blur. At level 1 it was near the 0.5 guessing line. So “supported on tuning” means the
anchor comparison added useful information and passed its controls. It does **not** mean that
slight blur has been solved.

The result also survived the experiment's resampling checks against the responsive group, the
anchor group, and its confidence-only twin. That makes the result less likely to depend on a
small handful of these 250 images, but it is still evidence from the same tuning set.

## Result 2: The later differential idea did not pass its test

The later idea compared the very highest-confidence group with a middle-confidence group. It was
suggested after earlier tuning work showed these groups moving in different directions. Because
the choice came after looking at tuning data, it was not allowed to succeed merely by having the
best average in this run.

The strongest later candidate was the simple gap between the two groups, using the mean scene
summary. Its technical name is `decile_90_100__50_60 / mean / raw_gap`.

It had the highest macro AUROC in the experiment: 0.721. It beat both raw groups and its
confidence-only twin. But the decision rule required it to beat earlier best results at the two
slightest blur levels. The full comparison with its matched confidence baseline was:

| Blur level | Test 2 AUROC | Plain confidence baseline | Difference | Decision target |
|---|---:|---:|---:|---|
| 1 | 0.512 | 0.522 | -0.010 | Above 0.538: No |
| 2 | 0.569 | 0.567 | +0.002 | Above 0.570: No |
| 3 | 0.669 | 0.645 | +0.024 | Reported only |
| 4 | 0.893 | 0.806 | +0.087 | Reported only |
| 5 | 0.961 | 0.895 | +0.066 | Reported only |
| Average | 0.721 | 0.687 | +0.034 | Not a separate target |

The third column is the **plain confidence baseline (“plain softmax” shorthand)**. This is not
literally a softmax calculation in the implementation. It starts with the detector's maximum
sigmoid class confidence, changes it to uncertainty using `1 - confidence`, and then compares the
same two confidence groups used by Test 2. It is therefore a matched confidence-only baseline,
not an unmatched score made from the whole image.

The difference column is Test 2 minus the confidence baseline. A negative number means the
baseline did better. At blur level 1, the confidence baseline was slightly better: 0.522 versus
0.512. Test 2 was better from level 2 onward, but only levels 1 and 2 had decision targets.
Levels 3 through 5 are shown to describe the result, not to turn them into extra passes.

The level-2 miss was very small, but the targets were fixed before this comparison was judged.
Moving a target after seeing the result would make the test unfair. No later differential
candidate passed the complete rule, so this idea is **not worth carrying to the held-out test**.

This is why the two verdicts must stay separate. The later idea won the average ranking because
it performed very well on heavy blur. The experiment was designed to ask whether it also
improved the difficult slight-blur cases. It did not show that.

## How the Test 1 and Test 2 scores were calculated

The winning versions of both tests first put detector guesses into confidence groups. They then
averaged the saved persistence distances inside each chosen group. The **reference** is the group
used as the ruler. The **responsive** group is the one expected to react more to blur.

The examples below use real values from image 885. They show the arithmetic for one image; they
do not replace the results over all 250 images. The displayed inputs are rounded to six decimal
places, so the equations use “approximately equal to.”

### Test 1: relative gap

Test 1 used the lowest-confidence 10 percent as the reference and the 50-to-60 percent group as
the responsive group. Its score was:

```text
relative gap = 2 × (responsive - reference) / (responsive + reference)
```

For image 885, the clean and strongest-blur calculations were:

```text
clean:  2 × (0.083252 - 0.085921) / (0.083252 + 0.085921) ≈ -0.03156
blur 5: 2 × (0.083908 - 0.078284) / (0.083908 + 0.078284) ≈ +0.06935
```

The subtraction measures how far the responsive group sits above or below the ruler. Dividing
by their average size makes the comparison less sensitive to one image having generally larger
distances than another. For example, doubling both inputs would leave this relative gap
unchanged. In image 885, the score rose from a small negative value to a positive value under
strong blur.

### Test 2: raw gap

Test 2 used the highest-confidence 10 percent as the reference and again used the 50-to-60
percent group as the responsive group. Its score was the simpler subtraction:

```text
raw gap = responsive - reference
```

For image 885:

```text
clean:  0.083252 - 0.102030 ≈ -0.01878
blur 5: 0.083908 - 0.077775 ≈ +0.00613
```

Here the gap also rose with blur. Unlike Test 1, this score does not divide by the size of the
two inputs. That makes it easier to calculate, but it also keeps the original distance scale.

### The matched confidence calculation

The plain confidence baseline first changes each selected query's maximum sigmoid confidence
into an uncertainty value using `1 - confidence`. It averages those uncertainty values in the
same two groups, then uses Test 2's `responsive - reference` raw gap. For image 885:

```text
clean:  0.958912 - 0.577506 ≈ 0.38141
blur 5: 0.912514 - 0.739659 ≈ 0.17286
```

This raw gap usually **fell** as blur increased. The experiment therefore locked its direction
as negative and multiplied its scores by `-1` when judging corruption. In this example, the
direction-adjusted values are `-0.38141` for clean and `-0.17286` for blur 5, so the blurred
value is still the larger corruption reading. A negative trend does not automatically mean a
bad detector; it means the useful direction is downward before the adjustment.

## What the measurements mean

AUROC was the main clean-versus-blur ordering measurement, but it was not the only measurement.
Spearman correlation first checked how each score moved inside the same image.

### Spearman: does one image move in order as blur grows?

For each image, **Spearman correlation** compared its six score ranks with the ordered blur
levels 0 through 5. It cares about order, not the exact size of each jump:

- `+1` means the score rose in perfect order as blur increased;
- `-1` means it fell in perfect order; and
- `0` means there was no consistent ranked direction.

The experiment calculated one signed Spearman coefficient for every image and then took the
middle, or median, value across the 250 images. A positive median locked an upward direction; a
negative median locked a downward direction. That choice was made once for each candidate and
was not changed separately at each blur level.

The top scores had these within-image measurements:

| Measurement | Test 1 | Test 2 | Plain confidence baseline |
|---|---:|---:|---:|
| Median signed Spearman | +0.657 | +0.829 | -0.914 |
| Images moving in the locked direction | 86.8% | 96.8% | 86.0% |
| Adjacent blur steps moving the locked way | 62.6% | 73.4% | 74.1% |
| Blur level 5 stronger than clean after direction adjustment | 86.4% | 97.2% | 89.6% |

The baseline's `-0.914` means its raw gap usually fell in a very orderly way. Its direction was
therefore locked downward. After this one adjustment, all three columns can be read as “larger
means more evidence of blur.”

The adjacent-step row asks a stricter question than Spearman: across the five steps from one blur
level to the next, how often did the score avoid moving the wrong way? The last row asks only
whether the strongest blur ended above clean after the direction adjustment. These checks show
the shape of the six-point curve, not how well scores from different images separate.

### AUROC: does a blurred image outrank a clean image?

After locking the direction, **AUROC** compares images with one another. At each blur level, the
experiment compared every one of the 250 blurred scores with every one of the 250 clean scores.
That makes `250 × 250 = 62,500` clean-versus-blurred pairs. A blurred score above a clean score
earns 1 point, a tie earns half a point, and a lower blurred score earns 0 points. AUROC is the
average number of points.

For a tiny example, imagine direction-adjusted clean scores `[0, 1]` and blurred scores `[1, 2]`:

- blurred `1` beats clean `0`;
- blurred `1` ties clean `1`;
- blurred `2` beats clean `0`; and
- blurred `2` beats clean `1`.

That is three wins and one tie, so the AUROC is:

```text
(1 + 0.5 + 1 + 1) / 4 = 0.875
```

A real AUROC of 0.5 is like guessing, while 1.0 means every blurred score was above every clean
score. A value below 0.5 is kept as a real result; the direction is not flipped afterward just
to make the number look better.

**Macro AUROC** is the ordinary average of the five separate AUROCs for blur levels 1 through 5.
For Test 2, the unrounded calculation was:

```text
(0.512216 + 0.569160 + 0.669056 + 0.893264 + 0.960832) / 5 = 0.7209056
```

AUROC values are **not probabilities**. A macro AUROC of 0.721 is not a 72.1-percent corruption
reading. It only describes how well the score ordered clean and blurred images in this
experiment.

### Other checks used in the decision

The study also checked whether all 250 images had a complete six-level curve; whether an anchor
was steady compared with normal differences between clean scenes; whether the anchor helped
explain the responsive group's clean value; and whether a contrast beat the two raw groups it
was made from. It also compared every persistence contrast with its matched confidence-only
baseline. These checks guard against a complicated score receiving credit for information that
was already available from one input or from detector confidence alone.

The study also used 2,000 **bootstrap resamples**. This means it repeatedly made new lists by
sampling from the same 250 tuning images, allowing an image to appear more than once. Candidate
and control scores used the same sampled image list. The study then recalculated their macro
AUROC difference to see how stable that advantage was. This is a useful stability check, but it
does not create new images and does not replace a held-out test.

## What this result does not prove

- **No held-out images were used.** The held-out images are the untouched test set meant to check
  whether a choice works beyond the practice data.
- The result does not prove that the planned anchor will work on new images, new corruption
  types, or a different detector.
- The result does not provide a probability that an image is corrupted.
- The result does not choose a safe deployment threshold.
- Agreement between several summaries of the same images is not the same as repeating the
  experiment on new data.
- The later differential result cannot be quoted as confirmed performance. Its groups were
  selected after reading earlier tuning results, and it failed its pre-set slight-blur rule.

## Bottom line

The experiment found a promising way to use one group of detector guesses as a ruler for another
group in the same image. The **planned anchor idea earned further validation**, with seven
candidates passing the full tuning checklist. It is a candidate for a carefully fixed held-out
test, not for deployment yet.

The **later differential idea stops here**. Its overall average looked strong, especially for
heavy blur, but it did not meet the slight-blur targets that mattered for deciding whether to test
it further.
