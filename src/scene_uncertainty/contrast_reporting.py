"""The published bundle: nine files that appear together, or nothing at all.

The atomic-publication mechanics are `corruption_reporting.write_corruption_report`'s, and they
are copied rather than reinvented because they are already reviewed and already documented:
refuse an existing output before computing anything, build into a staging directory beside the
destination so `os.replace` stays on one filesystem, `chmod` off `mkdtemp`'s `0o700`, check the
file set before the rename rather than after it, and clean up on `BaseException` so an interrupt
leaves no more behind than an error does.

**One deliberate difference from the sibling.** `corruption_reporting` imports its plot module
inside the writing function, because `corruption_plots` imports it back and a module-scope import
would be a cycle. `contrast_plots` imports nothing from here, so `write_contrast_plots` is a
module-scope import -- and it has to be a module global anyway, because the staging-cleanup tests
patch that name. A function-local import would rebind it after the patch and the tests would
quietly stop testing anything.

**Two rules about what may be dropped.** A CSV cell cannot hold a dictionary, so every
per-severity map is published twice: nested in `summary.json`, which the figures and any
programmatic reader use, and flattened into scalar columns here. The flattening is declared once
as a mapping and applied by `_project`, so the column set is derived from the declaration rather
than from whatever a row happens to hold. And an *undeclared* nested field is refused rather than
dropped. The spec's rule is "drop every dictionary-valued key, after expanding the declared
ones"; this is the stricter reading, because a nested field added to a candidate later would
otherwise vanish from the table with nothing anywhere saying so -- the reader gets a
complete-looking CSV whose only trace of the missing measurement is its absence.

**What the report may not say.** No sentence here describes a score as a probability that an
image is corrupted, and none compares magnitudes across the two signals -- a persistence distance
and a `1 - confidence` share no unit, so every cross-signal comparison is made on AUROC. The
word `calibrated` does not appear anywhere in the rendered text, including in the sentence
denying the claim: the spec asks the report to state that the result is not a calibrated
probability, and the test that enforces the denial forbids the word outright, so section 9 denies
it in other words. A reader skimming for "calibrated" must not find it in a report that is
telling them there is no such thing here.

**Every number comes from the summary.** `render_contrast_report` reads its argument and nothing
else, so any claim in `easy-report.md` can be checked against the `summary.json` beside it, and
a summary read back off disk renders the identical report. That round trip turns integer keys
into strings, which is why the per-severity lookups here accept either.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .contrast_analysis import (
    CONTRAST_ROW_FIELDS,
    DEPLOYABLE_SIGNAL,
    RESIDUAL_METHOD,
)
from .contrast_controls import BOOTSTRAP_PERCENTILES, CORRUPTED_SEVERITIES
from .contrast_inputs import ARMS, EXPECTED_SEVERITIES
from .contrast_plots import PLOT_FILENAMES, write_contrast_plots

DATA_FILES = (
    "per_scene_contrasts.csv",
    "anchor_diagnostics.csv",
    "candidate_metrics.csv",
    "summary.json",
    "easy-report.md",
)

EXPECTED_FILES = frozenset({*DATA_FILES, *PLOT_FILENAMES.values()})
"""Every file a published bundle holds, and the only files it may hold.

Checked against the staging directory *before* the rename, so a run that produced eight files or
ten never becomes a readable result. Both directions matter: a missing figure is a bundle whose
report describes a picture that is not there, and an extra file is a table nothing in
`summary.json` accounts for. The four figure names come from `contrast_plots`, which is where
they are declared, so the two cannot drift apart.
"""

SEVERITY_MAP_COLUMNS = {
    "auroc_by_severity": "auroc_severity",
    "twin_auroc_by_severity": "twin_auroc_severity",
    "responsive_control_auroc_difference_by_severity":
        "responsive_control_auroc_difference_severity",
    "reference_control_auroc_difference_by_severity":
        "reference_control_auroc_difference_severity",
    "twin_auroc_difference_by_severity": "twin_auroc_difference_severity",
}
"""The candidate's per-corrupted-severity maps, and the column prefix each becomes."""

SEVERITY_STATISTIC_COLUMNS = {
    "count": "score_count", "mean": "score_mean", "variance": "score_variance",
    "median": "score_median", "q25": "score_q25", "q75": "score_q75",
}
"""`severity_statistics` is a map of maps, so it flattens twice: one column per statistic per
severity. `score_` prefixes them because `count` and `mean` alone would sit in a candidate row
next to the coverage counts and read as those."""

BOOTSTRAP_COLUMNS = ("macro_difference", "low", "high", "verdict")
"""The four fields of a bootstrap that vary per candidate.

`seed` and `samples` are deliberately not columns: they are the same for every candidate in a
run, so a column would repeat one number down the table while inviting a reader to check whether
it varies. `summary.json` records them once, under `bootstrap`, read from the bootstraps that
were actually run rather than from the module constant.
"""

BOOTSTRAP_FIELDS = (
    "responsive_control_bootstrap",
    "reference_control_bootstrap",
    "twin_bootstrap",
)
"""All three, including the reference control's -- which the first draft of this table did not
have, because the reference bootstrap was added after it was written."""

DIAGNOSTIC_RELATIONSHIP_DROPPED = ("fold_lines",)
"""Five two-element lines per configuration: a nested value with no scalar reading, kept whole in
`summary.json`. Named here rather than skipped silently, so the strict rule below still holds."""

DIAGNOSTIC_DRIFT_DROPPED = ("by_image",)
"""Three numbers per image per configuration. At the declared roster that is 750 columns in a
21-row table, so the per-image detail stays in `summary.json` where a reader can index it."""

SECTION_TITLES = (
    "## Is the low-confidence range steady enough to be an anchor?",
    "## Does the anchor explain how scenes differ?",
    "## Does any contrast beat both of the ranges it is built from?",
    "## Does any contrast beat the detector's own confidence?",
    "## How well does it separate each blur level?",
    "## How much of this survives resampling?",
    "## Anchored and differential are two separate verdicts",
    "## What a deployment would store",
    "## What this result is not",
)
"""The spec's nine questions, in its order.

Every one of them is rendered on every run. A section omitted for want of a winner reads as a
question that was never asked, which is the opposite of what an empty result means: the question
was asked and the answer was no.
"""


def _umask_directory_mode(inside: Path) -> int:
    """The mode a plain `mkdir` would give a directory here, without touching the process umask.

    `tempfile.mkdtemp` creates its directory `0o700`. That is right for a temporary nobody else
    should read and wrong for a published result: `os.replace` carries the mode onto the bundle,
    and an operator who is not the owner is left unable to list a directory whose nine files the
    umask made group-readable.

    Read by creating a directory and asking what mode it came out with, rather than through
    `os.umask`, which is a setter: reading it means setting it and setting it back, and any
    concurrent thread creating a file in that window gets the wrong permissions instead.
    """
    probe = inside / ".mode-probe"
    probe.mkdir()
    mode = probe.stat().st_mode & 0o777
    probe.rmdir()
    return mode


def _put(columns: dict, name: str, value) -> None:
    """One column, once. A second write under one name is an error, not an overwrite.

    The flattened names are built from field names that live in four different modules, and two
    of them colliding would silently publish one measurement under another's heading. Today
    nothing collides -- the drift fields are `signed_q25` and `absolute_q25` where the spread
    fields are `q25` -- but "today nothing collides" is a property no test asserts and a new
    field could end.
    """
    if name in columns:
        raise RuntimeError(
            f"two values claim the column {name!r}; the flattening rules collide and one "
            "measurement would be published under the other's heading"
        )
    columns[name] = value


def _at(mapping, severity):
    """One severity out of a per-severity map, whichever way its keys survived JSON.

    `json.dumps` turns integer keys into strings, so a summary read back off disk indexes at
    `"3"` where the summary in memory indexes at `3`. Both are the same measurement and the
    report must render identically from either.
    """
    if mapping is None:
        return None
    if severity in mapping:
        return mapping[severity]
    return mapping.get(str(severity))


def _candidate_projection(candidate: dict) -> dict:
    """One candidate row with every declared map expanded and every other dictionary refused."""
    columns: dict = {}
    for name, value in candidate.items():
        if name in SEVERITY_MAP_COLUMNS:
            prefix = SEVERITY_MAP_COLUMNS[name]
            # `{prefix}_{severity}`, because every prefix in the table already ends in
            # `_severity`. The statistic columns below are the other way round -- their
            # prefixes name the statistic -- so the two rules genuinely differ and neither is
            # the other's typo.
            for severity in CORRUPTED_SEVERITIES:
                _put(columns, f"{prefix}_{severity}", _at(value, severity))
        elif name == "severity_statistics":
            for severity in EXPECTED_SEVERITIES:
                entry = _at(value, severity) or {}
                for field, prefix in SEVERITY_STATISTIC_COLUMNS.items():
                    _put(columns, f"{prefix}_severity_{severity}", entry.get(field))
        elif name in BOOTSTRAP_FIELDS:
            for field in BOOTSTRAP_COLUMNS:
                _put(columns, f"{name}_{field}", (value or {}).get(field))
        elif isinstance(value, dict):
            raise RuntimeError(
                f"candidate field {name!r} is a nested map with no declared columns; add it to "
                "SEVERITY_MAP_COLUMNS or name it as dropped, rather than letting it vanish "
                "from candidate_metrics.csv"
            )
        else:
            _put(columns, name, value)
    return columns


def _diagnostic_projection(entry: dict) -> dict:
    """One anchor-diagnostics row, flattened the same way and under the same strict rule."""
    columns: dict = {}
    for name, value in entry.items():
        if name == "drift":
            for severity, statistics in value["by_severity"].items():
                for field, number in statistics.items():
                    _put(columns, f"{field}_severity_{severity}", number)
            # The strict rule, applied to the one nested value with no columns: anything in
            # `drift` other than `by_severity` and the declared drop is a measurement nobody
            # decided about, and it would otherwise leave the table silently.
            undeclared = set(value) - {"by_severity", *DIAGNOSTIC_DRIFT_DROPPED}
            if undeclared:
                raise RuntimeError(
                    f"anchor diagnostic drift holds undeclared nested fields "
                    f"{sorted(undeclared)}; name them as columns or as dropped"
                )
        elif name == "spread":
            for severity, statistics in value.items():
                for field, number in statistics.items():
                    _put(columns, f"{field}_severity_{severity}", number)
        elif name == "relationship":
            for field, number in value.items():
                if field in DIAGNOSTIC_RELATIONSHIP_DROPPED:
                    continue
                _put(columns, f"relationship_{field}", number)
        elif isinstance(value, dict):
            raise RuntimeError(
                f"anchor diagnostic field {name!r} is a nested map with no declared columns"
            )
        else:
            _put(columns, name, value)
    return columns


def _fold_assignment(rows: list[dict]) -> dict:
    """Which fold each image was held out in, read from the rows rather than recomputed.

    Recomputing it from `assign_folds` would record the assignment the run *should* have used;
    reading it from the rows records the one it did. The two agree whenever the pipeline built
    the rows, which is exactly why the difference is invisible until it matters.
    """
    assignment: dict[int, int] = {}
    for row in rows:
        image_id, fold = row["image_id"], row["fold"]
        if assignment.setdefault(image_id, fold) != fold:
            raise RuntimeError(
                f"image {image_id} is in fold {assignment[image_id]} on one row and {fold} on "
                "another; the cross-fitting is not one partition and its residuals are not "
                "comparable"
            )
    return assignment


def _bootstrap_settings(candidates: list[dict]) -> dict:
    """The seed and sample count the run actually resampled with.

    Read from the bootstraps rather than from `BOOTSTRAP_SAMPLES`, because the constant is what
    a default run uses and the summary has to describe *this* run: a caller that passes
    `samples` -- Task 9's integration tests do, so that they finish -- would otherwise publish a
    sample count nobody used. Two different settings in one bundle is refused rather than
    averaged or picked from, because the intervals would not be comparable and a single recorded
    number would misdescribe whichever half it did not come from.
    """
    observed = {
        (entry["seed"], entry["samples"])
        for candidate in candidates
        for field in BOOTSTRAP_FIELDS
        for entry in (candidate.get(field),)
        if entry is not None
    }
    if len(observed) > 1:
        raise RuntimeError(
            f"the run holds more than one bootstrap setting {sorted(observed)}; intervals drawn "
            "with different sample counts are not comparable with each other"
        )
    seed, samples = observed.pop() if observed else (None, None)
    return {
        "seed": seed,
        "samples": samples,
        "percentiles": list(BOOTSTRAP_PERCENTILES),
    }


def build_contrast_summary(
    *,
    inputs,
    rows: list[dict],
    fits: dict,
    diagnostics: list[dict],
    candidates: list[dict],
    controls: list[dict],
    ranking: list[dict],
    figure_spans: dict,
) -> dict:
    """Everything the bundle publishes, as one JSON-serialisable dictionary.

    `figure_spans` is recorded exactly as `write_contrast_plots` returned it and is never
    recomputed here. A re-derived span is not a record of what the figure was drawn on, and the
    two can diverge with nothing to say so -- `summary.json` would then claim a range the PNG
    beside it does not have. That the two agree on every real bundle is what makes the
    difference invisible without a fixture built to separate them.

    `ranking` is serialised as candidate keys, not as candidate dictionaries. Serialising the
    winners whole would give each of them a second copy in the same file, and two copies of one
    candidate can disagree after any later edit to either.
    """
    return {
        "provenance": dict(inputs.provenance),
        "arms": [
            {
                "name": arm.name,
                "pair_name": arm.pair_name,
                "family": arm.family,
                "bucket_scheme": arm.bucket_scheme,
                "score_scope": arm.score_scope,
                "reference_bin": arm.reference_bin,
                "responsive_bin": arm.responsive_bin,
                "declared_before_data": arm.declared_before_data,
            }
            for arm in ARMS
        ],
        "fold_assignment": _fold_assignment(rows),
        "fits": [
            {"arm": key[0], "signal": key[1], "aggregation": key[2], **value}
            for key, value in fits.items()
        ],
        "anchor_diagnostics": diagnostics,
        "candidates": candidates,
        "reference_controls": controls,
        "ranking": [
            {
                "arm": candidate["arm"],
                "signal": candidate["signal"],
                "aggregation": candidate["aggregation"],
                "method": candidate["method"],
            }
            for candidate in ranking
        ],
        "figure_spans": figure_spans,
        "row_counts": {
            "per_scene_contrasts": len(rows),
            "anchor_diagnostics": len(diagnostics),
            "candidate_metrics": len(candidates),
            "reference_controls": len(controls),
            "ranked": len(ranking),
        },
        "bootstrap": _bootstrap_settings(candidates),
    }


# --- the plain-language report -------------------------------------------------------------------


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _plain(value, digits: int = 3) -> str:
    """A number as the report prints it, or the words for one that was never measured."""
    return "not measured" if value is None else f"{float(value):.{digits}f}"


def _signed(value, digits: int = 3) -> str:
    return "not measured" if value is None else f"{float(value):+.{digits}f}"


def _reported(summary: dict) -> list[dict]:
    """Every deployable-signal candidate that produced a macro AUROC.

    Not `ranking`, and the difference is the whole of what sections 3 to 5 report on. `ranking`
    names the winners; it is the answer to one question and not the set of results. A report
    whose control comparisons vanished on a run where nothing cleared the roster gate would hide
    exactly the numbers a reader needs in order to see why nothing did.
    """
    return [
        candidate for candidate in summary["candidates"]
        if candidate["signal"] == DEPLOYABLE_SIGNAL
        and candidate.get("macro_auroc") is not None
    ]


def _best(candidates: list[dict]) -> dict | None:
    return max(candidates, key=lambda item: item["macro_auroc"], default=None)


def _name(candidate: dict) -> str:
    return (
        f"`{candidate['arm']}` / {candidate['aggregation']} / {candidate['method']}"
    )


def _preamble(summary: dict) -> list[str]:
    provenance = summary["provenance"]
    counts = summary["row_counts"]
    return [
        "# Within-image corruption contrast",
        "",
        "Every number below is scored from one image at one moment, by comparing two "
        "confidence-percentile ranges measured on that same image. Nothing here reads a "
        "comparison bank, a neighbour set, or any image other than the one being scored.",
        "",
        f"Source: `{provenance['source']}`, {_count(provenance['image_count'], 'image')} of the "
        f"`{provenance['source_partition']}` partition, severities "
        f"{provenance['severities'][0]} to {provenance['severities'][-1]}.",
        f"Tables: {_count(counts['per_scene_contrasts'], 'per-scene row')}, "
        f"{_count(counts['candidate_metrics'], 'candidate')}, "
        f"{_count(counts['anchor_diagnostics'], 'anchor diagnostic')}.",
        "",
    ]


def _anchor_section(summary: dict) -> list[str]:
    """Section 1. Does the reference range hold still enough to be worth subtracting?"""
    entries = [
        entry for entry in summary["anchor_diagnostics"]
        if entry["signal"] == DEPLOYABLE_SIGNAL
    ]
    ratios = [
        (_at(entry["spread"], severity) or {}).get("stability_to_spread")
        for entry in entries
        for severity in CORRUPTED_SEVERITIES
    ]
    measured = [value for value in ratios if value is not None]
    steady = [value for value in measured if value < 1.0]
    lines = [
        SECTION_TITLES[0],
        "",
        "The anchor is the low-confidence reference range. Subtracting it only helps if it "
        "moves less under blur than it differs between scenes -- otherwise subtracting it "
        "removes more signal than noise. The ratio below is the median distance the reference "
        "drifts from its own clean value, divided by how far the reference spreads across "
        "scenes when they are all clean. Smaller is steadier; above 1.0 the anchor moves more "
        "than the scenes differ.",
        "",
    ]
    if not measured:
        lines += [
            "No configuration produced this ratio. That happens when the clean reference is "
            "identical across every scene, which leaves no scene-specific baseline to explain "
            "and no denominator to divide by.",
            "",
        ]
        return lines
    lines += [
        f"Across {_count(len(entries), 'persistence configuration')} and "
        f"{_count(len(CORRUPTED_SEVERITIES), 'blur level')}, {len(steady)} of "
        f"{len(measured)} measured ratios are below 1.0.",
        f"Steadiest {_plain(min(measured))}, least steady {_plain(max(measured))}, "
        f"middle of the range {_plain(sorted(measured)[len(measured) // 2])}.",
        "",
    ]
    return lines


def _relationship_section(summary: dict) -> list[str]:
    """Section 2. Does the anchor carry scene-specific information, or the group average?"""
    entries = [
        entry for entry in summary["anchor_diagnostics"]
        if entry["signal"] == DEPLOYABLE_SIGNAL
    ]
    predictive = [entry for entry in entries if entry["relationship"]["predictive"]]
    lines = [
        SECTION_TITLES[1],
        "",
        "A line is fitted from the reference range to the responsive range on clean images, "
        "and then judged on images it never saw. It earns the word predictive only by beating "
        "a fold-specific constant that ignores the reference entirely and always predicts the "
        "other folds' median. A line that cannot beat that constant is not explaining "
        "scene-specific baseline; it is reproducing the group average with extra steps, and an "
        "in-sample correlation can sit on top of exactly that.",
        "",
        f"{len(predictive)} of {_count(len(entries), 'configuration')} beat the constant.",
        "",
    ]
    for entry in entries:
        relationship = entry["relationship"]
        lines.append(
            f"- `{entry['arm']}` / {entry['aggregation']}: "
            f"cross-fitted error {_plain(relationship['crossfit_median_absolute_error'])} "
            f"against the constant's "
            f"{_plain(relationship['constant_median_absolute_error'])}"
            f" -- {'beats it' if relationship['predictive'] else 'does not beat it'}"
            f" (Spearman {_signed(relationship['spearman'])})."
        )
    lines.append("")
    return lines


def _beats_both_section(summary: dict) -> list[str]:
    """Section 3. Is the contrast worth more than either of the two ranges it is built from?"""
    reported = _reported(summary)
    winners = [item for item in reported if item.get("beats_both_inputs")]
    lines = [
        SECTION_TITLES[2],
        "",
        "A contrast is built from two raw ranges, so it has to clear both of them to have "
        "added anything. The matched responsive range and the matched reference range are the "
        "two controls, scored on the same images, at the same blur levels, in each control's "
        "own direction.",
        "",
        f"{len(winners)} of {_count(len(reported), 'measured candidate')} beat both of their "
        "inputs.",
        "",
    ]
    best = _best(reported)
    if best is None:
        lines += [
            "No candidate produced a macro AUROC, so there is nothing to compare against "
            "either control.",
            "",
        ]
        return lines
    lines += [
        f"Strongest measured candidate: {_name(best)}, macro AUROC "
        f"{_plain(best['macro_auroc'])}.",
        f"Against its responsive range: "
        f"{_signed(best.get('responsive_control_macro_difference'))}. "
        f"Against its reference range: "
        f"{_signed(best.get('reference_control_macro_difference'))}.",
        "",
    ]
    return lines


def _twin_section(summary: dict) -> list[str]:
    """Section 4. Or is this the detector's own confidence, measured a second way?"""
    reported = _reported(summary)
    redundant = [item for item in reported if item.get("confidence_redundant")]
    lines = [
        SECTION_TITLES[3],
        "",
        "Each candidate has a confidence-only twin: the same contrast, built from the "
        "detector's own confidence in the same two percentile ranges instead of from "
        "persistence. If the twin does as well, the persistence machinery has rediscovered "
        "something the detector already reports for free.",
        "",
        f"{len(redundant)} of {_count(len(reported), 'measured candidate')} fail this control.",
        "",
    ]
    if redundant:
        lines += [
            "Failing this control means the candidate adds nothing over the detector's own "
            "confidence. Named plainly because it is the finding most easily mistaken for a "
            "success: the AUROC can be high and the contrast still be redundant.",
            "",
        ]
        for candidate in redundant[:5]:
            lines.append(
                f"- {_name(candidate)}: macro {_plain(candidate['macro_auroc'])} against its "
                f"twin's {_plain(candidate.get('twin_macro_auroc'))}."
            )
        if len(redundant) > 5:
            lines.append(f"- and {_count(len(redundant) - 5, 'other')}.")
        lines.append("")
    else:
        lines += [
            "Every measured candidate is ahead of its confidence-only twin by more than the "
            "twin's own measurement, so none of them is the detector's confidence in "
            "disguise.",
            "",
        ]
    return lines


def _severity_section(summary: dict) -> list[str]:
    """Section 5. Mild blur first, because that is the regime nothing has separated yet."""
    reported = _reported(summary)
    best = _best(reported)
    lines = [
        SECTION_TITLES[4],
        "",
        "Severities 1 and 2 come first because that is the regime the completed deployment "
        "analysis could not separate. A macro gain driven entirely by severities 4 and 5 "
        "repeats a result already in hand, and is not what this experiment was run to find.",
        "",
    ]
    if best is None:
        lines += [
            "No candidate produced per-severity AUROCs, so there is nothing to read at any "
            "blur level.",
            "",
        ]
        return lines
    lines.append(f"Strongest measured candidate: {_name(best)}.")
    lines.append("")
    for severity in (1, 2, 3, 4, 5):
        value = _at(best.get("auroc_by_severity"), severity)
        twin = _at(best.get("twin_auroc_by_severity"), severity)
        lines.append(
            f"- severity {severity}: {_plain(value)}"
            f" (confidence twin {_plain(twin)})."
        )
    lines += [
        "",
        "0.5 is where a score that cannot tell clean from corrupted lands. A value below it is "
        "reported as measured rather than flipped: the direction was locked from the tuning "
        "trend before any of these were scored.",
        "",
    ]
    return lines


def _bootstrap_section(summary: dict) -> list[str]:
    """Section 6. How much of the gap survives resampling the images?"""
    reported = _reported(summary)
    settings = summary["bootstrap"]
    verdicts: dict[str, int] = {}
    for candidate in reported:
        entry = candidate.get("responsive_control_bootstrap")
        if entry is not None:
            verdicts[entry["verdict"]] = verdicts.get(entry["verdict"], 0) + 1
    lines = [
        SECTION_TITLES[5],
        "",
        "Each comparison is resampled by drawing images with replacement and rescoring both "
        "methods on the same draw, so the interval measures the difference between the methods "
        "rather than which scenes were sampled.",
        "",
        f"Settings: {_count(settings['samples'] or 0, 'resample')}, seed {settings['seed']}, "
        f"interval {settings['percentiles'][0]} to {settings['percentiles'][1]} per cent.",
        "",
    ]
    if not verdicts:
        lines += ["No comparison was resampled, so there is no interval to report.", ""]
        return lines
    for verdict, count in sorted(verdicts.items()):
        lines.append(f"- {verdict}: {_count(count, 'candidate')}.")
    lines += [
        "",
        "Neither label is a held-out claim. The arms, the score methods and the scene "
        "summaries were all chosen using this same tuning programme, so an interval that "
        "excludes zero says the gap is larger than resampling these images would explain -- "
        "not that it would reappear on images nobody has looked at.",
        "",
    ]
    return lines


def _families_section(summary: dict) -> list[str]:
    """Section 7. Two arms were declared in advance and two were not, and that is not a detail."""
    arms = {arm["name"]: arm for arm in summary["arms"]}
    anchored = [arm for arm in summary["arms"] if arm["family"] == "anchored"]
    differential = [arm for arm in summary["arms"] if arm["family"] == "differential"]
    reported = _reported(summary)
    ranked = summary["ranking"]
    lines = [
        SECTION_TITLES[6],
        "",
        f"{_count(len(anchored), 'anchored arm')} "
        f"({', '.join('`' + arm['name'] + '`' for arm in anchored)}) were declared before any "
        "tuning number was read. Their results are hypotheses tested against controls.",
        "",
        f"{_count(len(differential), 'differential arm')} "
        f"({', '.join('`' + arm['name'] + '`' for arm in differential)}) were chosen after "
        "reading tuning results -- the completed deployment analysis showed the 90-100 per cent "
        "decile moving hardest, and against the 50-60 per cent range. Any number from those "
        "two arms is a selection estimate. It may motivate a held-out test. It may not stand "
        "in for one, and it may not be quoted as performance.",
        "",
        "The distinction is a property of the design and is stated whether or not a "
        "differential arm won anything here.",
        "",
    ]
    for family, members in (("anchored", anchored), ("differential", differential)):
        names = {arm["name"] for arm in members}
        measured = [item for item in reported if item["arm"] in names]
        best = _best(measured)
        winners = [item for item in ranked if item["arm"] in names]
        lines.append(
            f"- {family}: {_count(len(measured), 'measured candidate')}, "
            f"{_count(len(winners), 'ranked candidate')}, "
            + (
                f"strongest macro {_plain(best['macro_auroc'])} at {_name(best)}."
                if best is not None
                else "none produced a macro AUROC."
            )
        )
    lines.append("")
    if any(arms[item["arm"]]["family"] == "differential" for item in ranked):
        lines += [
            "A differential arm is among the ranked results above. It was chosen after reading "
            "tuning data and needs held-out confirmation before it is described as anything "
            "other than a candidate for one.",
            "",
        ]
    return lines


def _deployment_section(summary: dict) -> list[str]:
    """Section 8. The line a deployment would store, and why it is not the line that was scored."""
    ranked = summary["ranking"]
    lines = [
        SECTION_TITLES[7],
        "",
    ]
    if not ranked:
        lines += [
            "Nothing is ranked, so there is no line to store. No candidate cleared the "
            "eligibility gate: a full tuning roster, complete coverage, a locked direction, a "
            "macro AUROC, and a computable confidence twin.",
            "",
            "The clean-relationship lines are still published, per configuration, in "
            "`summary.json` under `fits` and in `anchor_diagnostics.csv`. They are what a "
            "deployment would store if a candidate did clear the gate.",
            "",
        ]
        return lines
    winner = ranked[0]
    arm = next(
        (item for item in summary["arms"] if item["name"] == winner["arm"]), {}
    )
    fit = next(
        (
            item for item in summary["fits"]
            if (item["arm"], item["signal"], item["aggregation"])
            == (winner["arm"], winner["signal"], winner["aggregation"])
        ),
        None,
    )
    lines += [
        f"Top-ranked: `{winner['arm']}` / {winner['aggregation']} / {winner['method']}.",
        "",
        "Whatever the method, a deployment stores the same three things before any of them: "
        f"the two percentile ranges (`{arm.get('reference_bin')}` as the reference and "
        f"`{arm.get('responsive_bin')}` as the responsive), the score scope "
        f"`{arm.get('score_scope')}`, and the scene summary `{winner['aggregation']}`. Those "
        "are what identify the measurement; the score method is how the two ranges are "
        "combined once they are in hand.",
        "",
    ]
    line = (fit or {}).get("final_line")
    if winner["method"] != RESIDUAL_METHOD:
        # A fitted line is published for this configuration whatever won, but only
        # `clean_residual` consumes one -- the other three methods are functions of the two raw
        # ranges alone. Offering the line here as "what a deployment would store" would hand an
        # operator a number their chosen method has no place to put, and imply the two were
        # connected.
        lines += [
            f"`{winner['method']}` uses no fitted line. It is computed from the two raw ranges "
            "directly, so there is no slope or offset for a deployment of this candidate to "
            "store.",
            "",
            f"The clean lines are published regardless, per configuration, in `summary.json` "
            f"under `fits` and in `anchor_diagnostics.csv`. They are what a `{RESIDUAL_METHOD}` "
            "candidate would need, and every published one is cross-fitted -- see below.",
            "",
        ]
    elif line is None:
        lines += [
            f"This candidate is a `{RESIDUAL_METHOD}`, which needs a fitted clean line, and "
            "its configuration has none. That is why its residual is unavailable rather than "
            "merely poor.",
            "",
        ]
        return lines
    else:
        lines += [
            f"Clean line for that configuration: slope {_plain(line[0])}, offset "
            f"{_plain(line[1])}. Fitted on every clean image, and it is the line a deployment "
            "of this candidate would store.",
            "",
        ]
    lines += [
        "The published line is *not* the line the numbers above were scored with. Every "
        f"published `{RESIDUAL_METHOD}` is cross-fitted: each image's expected clean value "
        "comes from a line fitted on the other four folds, so no image helped construct the "
        "value it is then measured against. Scoring with the final line instead would make "
        "every residual smaller than the published one, by exactly the amount the image "
        "contributed to the fit.",
        "",
    ]
    return lines


def _limits_section(summary: dict) -> list[str]:
    """Section 9. The two readings this report has to rule out in so many words."""
    return [
        SECTION_TITLES[8],
        "",
        "No held-out images were used. Every number in this report comes from the tuning "
        "partition, and the folds mentioned above are folds *within* that partition, held out "
        "from each other and never from the separate test partition -- which this command does "
        "not read.",
        "",
        "These scores are not probabilities and were never fitted to any. A macro AUROC of "
        "0.7 says that a corrupted image outranks a clean one seven times in ten under this "
        "score; it does not say that any image is 70 per cent corrupted, and no threshold here "
        "has been mapped to a rate of anything.",
        "",
        "The two signals are compared on AUROC and on rank, never on how large their scores "
        "are. A persistence distance and a `1 - confidence` share no unit, so their magnitudes "
        "are not comparable and no sentence above compares them.",
        "",
        f"Figure spans are recorded in `summary.json` under `figure_spans`. Only "
        "`auroc_by_blur_severity.png` pins its panels, to 0.0 and 1.0; the other three "
        "autoscale, and their recorded pair is the range of the data drawn rather than a limit "
        "imposed on any panel.",
        "",
    ]


def render_contrast_report(summary: dict) -> str:
    """The whole report, built from `summary` and from nothing else.

    Which is the claim `summary.json` beside it rests on: every sentence carrying a number can
    be checked against the file next to it, two runs over the same rows produce the same text,
    and a summary read back off disk renders the identical report. Nothing is quoted from a
    previous run and nothing is rounded twice.

    Sections 3, 4 and 5 read `_reported`, not `ranking`. An empty ranking is a result -- it is
    the answer "no candidate cleared the gate" -- and it must not take the control comparisons
    down with it, because those are the numbers that say why.
    """
    lines = [
        *_preamble(summary),
        *_anchor_section(summary),
        *_relationship_section(summary),
        *_beats_both_section(summary),
        *_twin_section(summary),
        *_severity_section(summary),
        *_bootstrap_section(summary),
        *_families_section(summary),
        *_deployment_section(summary),
        *_limits_section(summary),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_contrast_report(
    output_value: str | Path,
    *,
    inputs,
    rows: list[dict],
    fits: dict,
    diagnostics: list[dict],
    candidates: list[dict],
    controls: list[dict],
    ranking: list[dict],
) -> dict:
    """Publish the run as one directory of nine files, or leave nothing behind at all.

    The directory appears in one step. Every table, figure, summary and paragraph is written
    into a staging directory beside the destination, the file set is checked against
    `EXPECTED_FILES`, and only then is the staging directory renamed into place -- so a reader
    never meets a bundle whose report describes a figure that is missing, and a failed run
    leaves neither the output directory nor the staging directory behind.

    The checks are in this order for a reason. An existing output directory is refused before
    anything is computed, because merging a new run into an old one produces a directory that is
    internally inconsistent and says nothing about it. The file-set check comes before the
    rename rather than after it, because a check that runs after publication has already
    published. And the cleanup catches `BaseException`, not `Exception`: a `KeyboardInterrupt`
    part-way through leaves the same half-written staging directory a `ValueError` does, and on
    a run an operator is watching it is the likelier of the two.

    What this does not claim: it is not crash-proof. A process killed outright runs no cleanup,
    so it leaves the staging directory on disk under its `.<name>.<random>` name -- rubbish
    beside the destination, and never something a reader could mistake for a finished bundle,
    which is the property being protected.
    """
    output = Path(output_value)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        os.chmod(staging, _umask_directory_mode(staging))
        # `CONTRAST_ROW_FIELDS` is the documented column list, so it is what produces the
        # columns rather than a second description of what a row happens to hold.
        pd.DataFrame(rows, columns=list(CONTRAST_ROW_FIELDS)).to_csv(
            staging / "per_scene_contrasts.csv", index=False
        )
        pd.DataFrame(
            [_diagnostic_projection(entry) for entry in diagnostics]
        ).to_csv(staging / "anchor_diagnostics.csv", index=False)
        pd.DataFrame(
            [_candidate_projection(candidate) for candidate in candidates]
        ).to_csv(staging / "candidate_metrics.csv", index=False)
        figure_spans = write_contrast_plots(
            staging, candidates=candidates, controls=controls, rows=rows, fits=fits
        )
        summary = build_contrast_summary(
            inputs=inputs, rows=rows, fits=fits, diagnostics=diagnostics,
            candidates=candidates, controls=controls, ranking=ranking,
            figure_spans=figure_spans,
        )
        # `allow_nan=False`: `json.dumps` would otherwise write the bare token `NaN`, which no
        # strict JSON reader accepts, and the file would fail somewhere far from this run.
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "easy-report.md").write_text(
            render_contrast_report(summary), encoding="utf-8"
        )
        actual = {path.name for path in staging.iterdir()}
        if actual != EXPECTED_FILES:
            raise RuntimeError(
                f"report bundle contains {sorted(actual)}, expected {sorted(EXPECTED_FILES)}"
            )
        os.replace(staging, output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
