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
slightest blur levels:

| Blur level | Candidate | Required target | Passed? |
|---|---:|---:|---|
| 1 | 0.512 | above 0.538 | No |
| 2 | 0.569 | above 0.570 | No |

The level-2 miss was very small, but the targets were fixed before this comparison was judged.
Moving a target after seeing the result would make the test unfair. No later differential
candidate passed the complete rule, so this idea is **not worth carrying to the held-out test**.

This is why the two verdicts must stay separate. The later idea won the average ranking because
it performed very well on heavy blur, reaching 0.893 at level 4 and 0.961 at level 5. The
experiment was designed to ask whether it also improved the difficult slight-blur cases. It did
not show that.

## What the measurements mean

**AUROC** measures ordering. Here it asks how often a blurred image gets a stronger corruption
score than a clean image. A value of 0.5 is like guessing. A value of 1.0 means every blurred
example was ordered above every clean example. **Macro AUROC** is the average of this score over
blur levels 1 through 5.

AUROC values are **not probabilities**. A macro AUROC of 0.721 is not a 72.1-percent corruption
reading. It only describes how well the score ordered clean and blurred images in this
experiment.

The study also used 2,000 **bootstrap resamples**. This means it repeatedly made new lists by
sampling from the same 250 tuning images, allowing an image to appear more than once. It then
checked whether the comparisons kept the same direction. This is a useful stability check, but
it does not create new images and does not replace a held-out test.

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
