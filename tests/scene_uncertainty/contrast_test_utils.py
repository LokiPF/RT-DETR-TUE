"""A finished corruption-sensitivity bundle, small enough to write in a test.

`write_source_bundle` produces the two files `load_contrast_inputs` reads and nothing else.
It is deliberately not a call into `corruption_reporting`: a fixture built by the producer
under test cannot catch a producer/consumer disagreement, and the whole point of the loader's
validation is to notice when the upstream bundle is not what this command expects.

What it does copy from the producer is *shape*. The rows carry the seventeen columns
`PER_SCENE_KEYS` publishes in that order, and `summary.json` keeps `image_count`, `severities`
and `source_partition` under `run` and the row count under `validation`, because that is where
`corruption_reporting.build_summary` puts them. A fixture that invented a convenient key would
let a consumer be written against a file that does not exist.

And it copies *numbers*. Nine tasks read this fixture and only this task's tests read
`contrast_inputs.py`, so from Task 3 onwards every expectation anyone writes is an expectation
about these values. A fixture that disagrees with the producer does not fail -- it produces
plausible numbers nobody can tell from correct ones. So the five persistence levels, their
severity slopes, the `combined` z-scores, the confidence column, `selected_count` and
`clean_overlap` are all the completed 250-image tuning run's own dynamic/filtered `mean`
figures, and each constant's docstring says which measurement it came from.

The one import from `src` is `complete_trend_metrics`, and it is the opposite case to
`corruption_reporting` above. The four trend columns are not part of the *shape* this fixture
is asserting; they are a *derivation* the producer performs on the curve it just wrote, and
`corruption_reporting.summarize_candidates` performs it by calling exactly this function. A
second implementation here could only ever agree with the real one by luck, and the
disagreement would be invisible: `load_contrast_inputs` does not read those four columns at
all, so nothing in this task's suite would notice, and Tasks 3 through 9 would inherit trend
fields that contradict the scores sitting beside them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.scene_uncertainty.corruption_metrics import complete_trend_metrics

PER_SCENE_HEADER = (
    "image_id", "severity", "signal", "bucket_scheme", "confidence_bin",
    "membership_mode", "padding_mode", "aggregation", "score_scope",
    "source_partition", "score", "selected_count", "clean_overlap",
    "fully_measured", "signed_spearman", "absolute_spearman", "direction",
)

SERIES = (
    ("persistence", "decile", "decile_00_10", "layer_2"),
    ("persistence", "decile", "decile_50_60", "layer_2"),
    ("persistence", "decile", "decile_50_60", "combined"),
    ("persistence", "decile", "decile_90_100", "layer_2"),
    ("persistence", "decile", "decile_90_100", "combined"),
    ("persistence", "quintile", "quintile_00_20", "layer_2"),
    ("persistence", "quintile", "quintile_40_60", "layer_2"),
    ("confidence", "decile", "decile_00_10", "confidence"),
    ("confidence", "decile", "decile_50_60", "confidence"),
    ("confidence", "decile", "decile_90_100", "confidence"),
    ("confidence", "quintile", "quintile_00_20", "confidence"),
    ("confidence", "quintile", "quintile_40_60", "confidence"),
)

AGGREGATIONS = ("mean", "q90", "top20_mean")

SELECTED_COUNT = {"decile": 30, "quintile": 60}
"""How many queries one bin holds, by bucket scheme.

A quintile is two deciles wide, so it holds about twice as many of the ~300 padded queries an
image contributes. Writing one number for both would make the two resolutions
indistinguishable in the one column that records how much evidence each was summarised from --
and "a quintile estimate is pooled over twice the queries" is the only reason the coarser
scheme is in the experiment at all.
"""

CLEAN_OVERLAP_AT_MAX = {
    "decile_00_10": 0.053, "quintile_00_20": 0.132,
    "quintile_40_60": 0.121, "decile_50_60": 0.053,
    "decile_90_100": 0.200,
}
"""Severity 5's dynamic-membership Jaccard against severity zero's bins, per bin.

The completed run's own figures. They are low because dynamic membership is rebuilt from each
severity's own confidences: by the strongest blur, a bin holds almost none of the queries it
held when clean, which is the whole reason the column is published.
"""


def clean_overlap_for(
    confidence_bin: str, severity: int, membership_mode: str = "dynamic"
) -> float:
    """The `clean_overlap` a real bundle carries for one row, which is `1.0` twice over.

    `corruption_analysis` measures this against `memberships[0]["dynamic"]` -- and severity
    zero's dynamic bins *are* that mapping, so every dynamic severity-zero row compares a
    partition with itself and reads exactly `1.0`. Frozen membership reuses severity zero's
    bins at every severity, so every frozen row reads `1.0` too. Those are not "typically
    high" values a fixture may round off to something plausible: they are the two cases where
    the producer cannot write anything else, and a fixture that writes `0.2` there tells nine
    downstream tasks that severity zero moved.

    Between them the value decays geometrically to `CLEAN_OVERLAP_AT_MAX` at severity 5. Only
    the two endpoints are the run's; the intermediate shape is a choice, and it is a curve
    rather than a straight line because a straight line from 1.0 to 0.053 would put severity 1
    at 0.81 -- claiming the first blur step barely moves membership, which is the one thing
    this column exists to contradict.
    """
    if membership_mode != "dynamic" or severity == 0:
        return 1.0
    return round(CLEAN_OVERLAP_AT_MAX[confidence_bin] ** (severity / 5), 6)


PERSISTENCE_LEVEL = {
    "decile_00_10": 0.0861, "quintile_00_20": 0.0849,
    "quintile_40_60": 0.0833, "decile_50_60": 0.0834,
    "decile_90_100": 0.0981,
}
"""Severity zero of the `persistence`/`layer_2` column, per bin: the run's own five levels.

Near 0.086, not near 1.0. This is a mean nearest-neighbour distance in a normalised
fingerprint space, and its *between-bin* differences are small -- 0.0028 separates the two
lowest bins -- which is exactly why the experiment reads a within-image gap rather than a raw
score. A fixture that gave the bins a wide spread would make every contrast trivially large
and every AUROC trivially perfect.

All five differ, including across schemes. `decile_50_60` at 0.0834 and `quintile_40_60` at
0.0833 are one ten-thousandth apart and that is the point: they are two ways of cutting the
middle of the same confidence ranking, and the comparison this command exists to make is
between the two resolutions. A single shared level makes arms 1 and 2 produce identical
numbers at every image, severity, aggregation and method, right through to identical macro
AUROCs -- so a mutation that computed arm 2 out of the decile bins would pass the whole suite.
"""

PERSISTENCE_SLOPE = {
    "decile_00_10": -0.00006, "quintile_00_20": 0.00128,
    "quintile_40_60": 0.00344, "decile_50_60": 0.00318,
    "decile_90_100": -0.00368,
}
"""Per-severity drift of the same column, signed the way the run's median trend is signed.

Four of the five are `(severity 5 - severity 0) / 5` of the run's levels. `decile_00_10` is the
exception and it is a deliberate one: the run's mean level *rises* by 0.0003 over the sweep
while its median per-image signed Spearman is **-0.200**, which is what most images do. The
magnitude here is the run's, 0.0003 over five steps; the sign is the median trend's, because
the sign is what `corruption_metrics.choose_orientation` reads and the mean level is not read
by anything.

Nothing here may be zero. A constant curve makes `complete_trend_metrics` short-circuit to
`signed_spearman = 0.0` / `direction = "flat"`, `choose_orientation` returns `None` on a median
of exactly zero, and the arm whose reference that is has no orientation, no macro AUROC and a
`beats_both_inputs` that is structurally `False`. Both bins that carried a zero slope here were
the two *anchored* arms' references -- the two declared before any data was read -- so the
spec's "beats both its inputs" criterion could never have been demonstrated on either of them.
"""

COMBINED_LEVEL = {
    "decile_00_10": 0.1187, "quintile_00_20": 0.0537,
    "quintile_40_60": -0.1213, "decile_50_60": -0.1299,
    "decile_90_100": 0.1547,
}
"""Severity zero of `persistence`/`combined`, and three of the five are negative.

`decile_scoring.score_selection` builds this scope as the mean over decoder layers of
`(score - center) / scale`, where `center` is the clean median distance -- a signed robust
z-score, negative for every selection sitting below that median. These are the run's five.

It is emphatically *not* `layer_2` plus a constant. An offset cancels in `raw_gap` and in
`clean_residual`, which makes arms 3 and 4 identical for two of their three methods, and it
never produces a negative value -- so the loader's `COMBINED_SCOPE` exemption and Task 2's
exclusion of the symmetric relative gap at this scope, the two places the design pays for
`combined` being signed, would be exercised by bespoke overrides and never by the fixture
every later task is built on.
"""

COMBINED_SLOPE = {
    "decile_00_10": -0.01056, "quintile_00_20": -0.00056,
    "quintile_40_60": 0.02306, "decile_50_60": 0.02138,
    "decile_90_100": -0.08292,
}
"""`(severity 5 - severity 0) / 5` of the run's `combined` levels, and its own signs.

Not the `layer_2` signs. `quintile_40_60` and `decile_50_60` climb out of negative territory
towards zero while `decile_90_100` falls from `+0.155` to `-0.260` -- the top decile crossing
its own clean median, which is the single strongest movement in the whole run and the reason
arm 4 exists. Two of the five run the opposite way to their `layer_2` twins, so a consumer
that read one scope where it meant the other gets a different answer here rather than the same
answer twice.
"""

CONFIDENCE_UNCERTAINTY = {
    "decile_00_10": 0.985, "quintile_00_20": 0.979,
    "quintile_40_60": 0.936, "decile_50_60": 0.930,
    "decile_90_100": 0.645,
}
"""Severity zero of the `confidence` column, per bin -- and the column is `1 - confidence`.

`decile_scoring._checked_confidence` writes one minus the maximum class score, so **larger means
less confident** and the bins, which are named by ascending confidence, carry *descending* values.
The 0--10 percent bucket is the least confident and reads highest. Getting this backwards does not
break anything visibly: it flips the sign of all 36 confidence-twin contrasts and leaves every
count, key and coverage check intact, so it survives the entire loader suite and surfaces four
tasks later as an orientation nobody can explain.

The five levels are the completed run's own dynamic/filtered `mean` figures at severity zero,
rounded to three places. The two schemes stay distinguishable -- `decile_50_60` at 0.930 against
`quintile_40_60` at 0.936 -- so a consumer reading the wrong scheme's twin does not find identical
numbers, and that ordering is the run's too.
"""

CONFIDENCE_UNCERTAINTY_SLOPE = {
    "decile_00_10": -0.00044, "quintile_00_20": -0.00050,
    "quintile_40_60": 0.00022, "decile_50_60": 0.00044,
    "decile_90_100": 0.03074,
}
"""Per-severity drift of the same column, and the one place this fixture has to be steep.

Four of the five bins are flat to the third decimal across all six severities -- two drifting down
and two up, which is the run's own pattern and not a rounding artefact. `decile_90_100` climbs
0.645 to 0.799: the detector losing confidence in the queries it was surest about, which is the
single behaviour the redundancy control exists to expose. A fixture where every bin fell with
severity would let a consumer that had the top decile upside down pass every test.

Each slope is `(severity 5 - severity 0) / 5` from the completed run.
"""

TREND_TILT = (1.0, 1.6, 2.2, -0.4, 1.9, -0.3)
"""How much of its bin's severity slope one image gets, indexed by `image_id` modulo six.

This is the term that stops every image from carrying the same curve. Without it the per-image
term cancels in every gap method -- `raw_gap` and `clean_residual` both subtract two bins of the
*same* image -- so all six images produce one identical gap curve, every candidate's macro AUROC
saturates at 1.0, ties at the ceiling decide Task 6's redundancy comparisons, and the paired
bootstrap's interval collapses to a point. Varying the *level* per image does not fix that; only
varying the *slope* does, which is why this multiplies severity rather than being added to it.

Three properties, all load-bearing:

* the six sum to 6.0, so their mean is exactly 1.0 and a roster covering every residue
  reproduces `PERSISTENCE_LEVEL` and its slopes as its own average curve;
* two of the six are negative, so images genuinely disagree about which way the bin moves and
  the pooled clean-versus-corrupted ranking stops being perfect;
* the median is +1.3, comfortably positive, so `choose_orientation` still locks onto the bin's
  own sign rather than being decided by the tilt.

Deterministic and seed-free by construction: it is a six-entry table read with `image_id % 6`.
Nothing in a test fixture may reach for `random`, a clock or a hash whose iteration order can
move, because a bundle that is not byte-identical between two runs makes every failure below it
unreproducible.
"""

TREND_SHAPE = (0.0, 0.9, -1.2, 0.4, -0.6, 0.5)
"""A trendless six-point wobble, rotated per image and per bin, in units of the bin's slope.

Without it every curve is a straight line and every per-image `signed_spearman` is exactly
`+1` or `-1`. The run's medians are 0.086, 0.200, 0.629, 0.657 and 0.829 -- nowhere near
saturation -- because a real bin's mean distance does not climb monotonically through six blur
levels; it wanders, and the median trend measures how much of the climb survives the wandering.

The six entries sum to zero, so rotating them costs the curve nothing: any roster covering all
six rotations averages the wobble away and recovers the bin's own level and slope. The rotation
is `(severity + image_id + BIN_TREND_PHASE[bin]) % 6`, so it moves with the image -- which is
what makes images disagree -- and with the bin, which is what stops it cancelling between the
two bins of an arm the way a bin-blind wobble would.
"""

BIN_TREND_PHASE = {
    "decile_00_10": 3, "quintile_00_20": 0,
    "quintile_40_60": 2, "decile_50_60": 1,
    "decile_90_100": 4,
}
"""Where each bin starts reading `TREND_SHAPE`. Five distinct phases, chosen not decorative.

Two bins sharing a phase share their wobble exactly, and a shared wobble subtracts out of that
pair's gap -- which is the cancellation this whole term exists to avoid. The particular
assignment is the one that puts each bin's median signed Spearman closest to the run's, given
the noise ratios below; see `PERSISTENCE_TREND_NOISE`.
"""

SIGNAL_TREND_PHASE = {"persistence": 0, "confidence": 5}
"""Which `TREND_TILT` entry a signal reads for a given image, as an offset on `image_id`.

The two signals read the same table at different offsets so that an image which responds
strongly in persistence is not the same image that responds strongly in confidence. Sharing one
offset would make the control move with the signal it is controlling for, image by image, and
the redundancy question Task 6 asks -- does the persistence gap tell us anything its confidence
twin does not -- would be answered by the fixture's arithmetic instead of by the data.

Confidence's offset is 5, which hands image 1 the tilt `1.0`. That is not arbitrary either:
image 1 is the image every pinned confidence value in `test_contrast_inputs` reads, and a tilt
of 1.0 there keeps the column exactly the run's own curve at the one place it is asserted
against the run.
"""

PERSISTENCE_TREND_NOISE = {
    "decile_00_10": 8.05, "quintile_00_20": 12.05,
    "quintile_40_60": 3.25, "decile_50_60": 3.40,
    "decile_90_100": 2.15,
}
"""`TREND_SHAPE`'s amplitude for each bin, as a multiple of that bin's own severity slope.

Expressed as a ratio rather than an absolute size because Spearman is scale-free: the ratio,
the tilt table and the phase are the *whole* of what decides a curve's rank correlation, so
these five numbers are what set the median signed Spearman -- and they were solved for, not
picked. At six images the medians come out -0.200, +0.086, +0.657, +0.629 and -0.829, which is
the run's column exactly; at 250 they are the same except `quintile_00_20`, which the six-point
Spearman grid cannot place between +0.057 and +0.171.

The ordering is the point as much as the levels. `decile_90_100` moves hardest and most
consistently, the two middle bins next, and the two lowest bins barely hold a direction at all
-- so a bin whose noise ratio drifted would show up as a candidate that suddenly ranks
somewhere new, rather than as a number nobody can check.
"""

COMBINED_TREND_NOISE = {
    "decile_00_10": 6.35, "quintile_00_20": 10.05,
    "quintile_40_60": 7.60, "decile_50_60": 5.60,
    "decile_90_100": 2.15,
}
"""The same ratios for the `combined` scope, solved against that scope's own medians.

They differ from the `layer_2` ratios because the run's two columns disagree about how
consistently each bin moves: `quintile_40_60` and `decile_50_60` both fall from a signed
Spearman near +0.64 at `layer_2` to +0.371 combined, and `decile_00_10` and `quintile_00_20`
change sign outright. Reusing one map would make the two scopes rank-identical, and arms 3 and
4 -- which are the same bucket pair read at the two scopes -- would then differ only in scale.
"""

SCENE_BASELINE_SPREAD = 0.02
"""How far apart two scenes' baseline persistence distances sit, end to end.

Chosen the same size as the severity effect (the steepest bin moves 0.017 across the sweep) so
that neither swamps the other: a spread much larger makes every cross-scene AUROC chance-level
whatever the candidate does, and a spread much smaller makes the raw score a perfect corruption
detector and leaves the paired within-image design with nothing to be better than.
"""

COMBINED_SCALE_GAIN = 6.7
"""How much larger the `combined` scope's numbers are than `layer_2`'s, for the scene baseline.

`combined` divides by a clean scale, so a scene's baseline offset survives into it amplified.
6.7 is the median of the five ratios between the two tables' severity drifts -- 0.44, 6.7, 6.7,
22.5 and 176 -- rather than an invented constant. Applying no gain at all would make the scene
baseline negligible against a z-score of order 0.1, and severity 0 of an arm-4 gap would be the
same number on every image.
"""


def _scene_baseline(image_id: int) -> float:
    """One image's own distance level, distinct for every image on the tuning roster.

    `(89 * image_id) mod 251` is a bijection on 1..250 because 251 is prime and 89 is not a
    multiple of it, so all 250 tuning images get their own baseline and none collide -- while
    the value does not climb with `image_id`, which a plain `c * image_id` term does. That
    matters because a monotone-in-`image_id` baseline turns the image roster into a second
    severity axis: sort the images and the "clean" group of any cross-scene AUROC is the low
    half of an ordered list rather than a sample.

    Sized for 250 images, the largest roster `load_contrast_inputs` accepts. Past 251 it
    repeats, which no brief in this plan reaches and which is recorded here so the next person
    to grow the roster reads it before they do.
    """
    return SCENE_BASELINE_SPREAD * (((image_id * 89) % 251) / 251 - 0.5)


def default_score(
    image_id: int, severity: int, confidence_bin: str, signal: str, scope: str
) -> float:
    """A score that differs along every axis of the loader's key except `aggregation`.

    Four terms, and each one closes a hole that a fixture without it leaves open.

    * **the bin's level** -- five distinct values per signal and scope, so no two bins are one
      series. Without it the reference and responsive series are identical and every contrast
      is exactly zero; with it shared between `decile_50_60` and `quintile_40_60` the two
      *arms* are identical, which is worse, because the numbers all look reasonable.
    * **the scene baseline** -- `_scene_baseline(image_id)`, so images are not copies of each
      other. It cancels out of any within-image gap by design; that is what the paired
      comparison is for, and it is why it cannot be the only per-image term.
    * **the bin's slope, tilted per image** -- `TREND_TILT`, the term that survives the gap. It
      is what makes six images produce six different gap curves instead of one repeated six
      times, and therefore what keeps a candidate's macro AUROC off 1.0.
    * **a trendless wobble** -- `TREND_SHAPE` scaled by the bin's own noise ratio and rotated
      per image and per bin, so a per-image `signed_spearman` is a number between the extremes
      the way the run's are, rather than `+1` or `-1` on every row.

    The `scope` split is between `combined` and everything else, and it is a different map
    rather than an offset: see `COMBINED_LEVEL`.

    The confidence branch runs *opposite* to the persistence one and that is not a slip. The
    column is `1 - confidence`, so its bins descend where persistence ascends, and only the top
    decile rises with severity where persistence at that bin falls. A fixture that made the two
    signals parallel would hide the redundancy the confidence twin exists to detect. It takes
    the tilt but no wobble: the column's shape is asserted directly against the run in
    `test_the_confidence_column_runs_the_way_the_producer_writes_it`, including that four of
    its five bins move by less than 0.01 across the whole sweep, and a wobble large enough to
    bend a curve that flat would be larger than the curve.

    Its per-image term subtracts, where persistence's is centred on zero, for the same reason.
    `1 - confidence` is bounded above by 1.0 -- `decile_scoring._checked_confidence` refuses a
    source confidence outside [0, 1] -- and these levels start at 0.985, so an *additive* term
    crosses the bound at fifteen images and reaches 1.235 on the 250-image roster the real
    command runs at. The loader checks negativity and finiteness, and would accept every one of
    those impossible rows.

    `aggregation` is the one exception, and it is a gap rather than a decision: three
    aggregations of one selection are three summaries of one population, and this callback is
    not handed the aggregation to vary on. The signature is five positional arguments because
    every later task's wrapper delegates to it with exactly those five, so a test that needs the
    three aggregations to differ has to build its rows some other way -- and a consumer that
    read `q90` where it meant `mean` would not be caught here.
    """
    if signal == "confidence":
        tilt = TREND_TILT[(image_id + SIGNAL_TREND_PHASE["confidence"]) % 6]
        return round(
            CONFIDENCE_UNCERTAINTY[confidence_bin]
            - 0.001 * image_id
            + CONFIDENCE_UNCERTAINTY_SLOPE[confidence_bin] * tilt * severity,
            6,
        )
    if scope == "combined":
        level = COMBINED_LEVEL[confidence_bin]
        slope = COMBINED_SLOPE[confidence_bin]
        noise = COMBINED_TREND_NOISE[confidence_bin]
        gain = COMBINED_SCALE_GAIN
    else:
        level = PERSISTENCE_LEVEL[confidence_bin]
        slope = PERSISTENCE_SLOPE[confidence_bin]
        noise = PERSISTENCE_TREND_NOISE[confidence_bin]
        gain = 1.0
    tilt = TREND_TILT[(image_id + SIGNAL_TREND_PHASE["persistence"]) % 6]
    wobble = TREND_SHAPE[(severity + image_id + BIN_TREND_PHASE[confidence_bin]) % 6]
    return round(
        level
        + gain * _scene_baseline(image_id)
        + slope * tilt * severity
        + abs(slope) * noise * wobble,
        6,
    )


def write_source_bundle(
    directory: Path,
    *,
    image_ids=range(1, 7),
    partition: str = "tuning",
    severities=range(6),
    membership_mode: str = "dynamic",
    padding_mode: str = "filtered",
    score=default_score,
    drop=(),
    extra_rows=(),
) -> Path:
    """Write `per_scene.csv` and `summary.json` into `directory` and return it.

    `drop` removes `(signal, confidence_bin, score_scope)` series so a test can prove the
    loader refuses an incomplete source. `extra_rows` appends raw dictionaries so a test can
    prove it refuses duplicates and held-out rows, and so a test can add the `frozen` and
    `unfiltered` twins a real bundle carries for the same bins this command reads.

    The four trend columns are computed here rather than written as literals, by the same
    `complete_trend_metrics` `corruption_reporting.summarize_candidates` calls, on the same
    six-severity curve the rows beside them carry. That is the only way the fixture cannot
    contradict itself: `direction` is derived from the sign of `signed_spearman` inside that
    function, `absolute_spearman` from its magnitude, and a curve that is not the full ladder
    comes back `fully_measured=False` with both correlations `None` instead of a plausible
    number. A literal `0.6 / 0.6 / "increasing"` is wrong on every row of a real bundle, and
    wrong in a way `load_contrast_inputs` cannot see, because it does not read these columns --
    only Tasks 3 through 9 do.

    The trend is computed once per series and image and reused across the three aggregations,
    because nothing in `default_score` varies with aggregation and the three curves really are
    identical. If an aggregation term is ever added, this must move inside that loop.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_ids = [int(value) for value in image_ids]
    severities = [int(value) for value in severities]
    rows = []
    for signal, scheme, confidence_bin, scope in SERIES:
        if (signal, confidence_bin, scope) in drop:
            continue
        curves = {
            image_id: [
                score(image_id, severity, confidence_bin, signal, scope)
                for severity in severities
            ]
            for image_id in image_ids
        }
        trends = {
            image_id: complete_trend_metrics(severities, curve)
            for image_id, curve in curves.items()
        }
        for aggregation in AGGREGATIONS:
            for image_id in image_ids:
                trend = trends[image_id]
                for severity, value in zip(severities, curves[image_id]):
                    rows.append({
                        "image_id": image_id, "severity": severity, "signal": signal,
                        "bucket_scheme": scheme, "confidence_bin": confidence_bin,
                        "membership_mode": membership_mode, "padding_mode": padding_mode,
                        "aggregation": aggregation, "score_scope": scope,
                        "source_partition": partition,
                        "score": value,
                        "selected_count": SELECTED_COUNT[scheme],
                        "clean_overlap": clean_overlap_for(
                            confidence_bin, severity, membership_mode
                        ),
                        "fully_measured": trend["fully_measured"],
                        "signed_spearman": trend["signed_spearman"],
                        "absolute_spearman": trend["absolute_spearman"],
                        "direction": trend["direction"],
                    })
    rows.extend(dict(row) for row in extra_rows)
    with (directory / "per_scene.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PER_SCENE_HEADER))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "run": {
            "source_partition": partition,
            "image_count": len(image_ids),
            "severities": list(severities),
        },
        "validation": {"per_scene_row_count": len(rows)},
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return directory
