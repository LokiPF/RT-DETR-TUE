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
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

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

CONFIDENCE_LEVEL = {
    "decile_00_10": 0.05, "quintile_00_20": 0.10,
    "decile_50_60": 0.55, "quintile_40_60": 0.50,
    "decile_90_100": 0.95,
}
"""Roughly the mean confidence of the queries a bin actually holds, per bin.

The confidence control is not one number repeated: a decile bucket is *defined* by the
confidence of its members, so the 90--100 percent bucket's control sits near 0.95 and the
0--10 percent bucket's near 0.05. The two schemes are given distinguishable levels for the same
reason the persistence terms are -- `decile_50_60` at 0.55 against `quintile_40_60` at 0.50 --
so a consumer that read the wrong scheme's twin would not find identical numbers.
"""


def default_score(
    image_id: int, severity: int, confidence_bin: str, signal: str, scope: str
) -> float:
    """A score that differs along every axis of the loader's key except `aggregation`.

    Every term is needed. Without the `image_id` term two images share a curve and a
    between-image spread test measures nothing. Without the `severity` term the trend is flat
    and every orientation test passes vacuously. Without the bin term the reference and
    responsive series are identical and every contrast is exactly zero -- which is why the
    confidence branch carries `CONFIDENCE_LEVEL` rather than returning early on `signal` alone:
    a bin-blind control is a control that reads `0.0` for every image at every severity, and the
    confidence twin is the comparison the whole arm table is built around. Without the `scope`
    term the `layer_2` and `combined` differential arms are byte-identical, and a mutation that
    read the wrong scope would pass every test in the suite.

    `aggregation` is the one exception, and it is a gap rather than a decision: three
    aggregations of one selection are three summaries of one population, and this callback is
    not handed the aggregation to vary on. The signature is five positional arguments because
    every later task's wrapper delegates to it with exactly those five, so a test that needs the
    three aggregations to differ has to build its rows some other way -- and a consumer that
    read `q90` where it meant `mean` would not be caught here.
    """
    if signal == "confidence":
        return round(CONFIDENCE_LEVEL[confidence_bin] + 0.001 * image_id - 0.002 * severity, 6)
    base = 1.0 + 0.01 * image_id + (0.5 if scope == "combined" else 0.0)
    lift = {"decile_00_10": 0.0, "quintile_00_20": 0.0,
            "decile_50_60": 0.30, "quintile_40_60": 0.30,
            "decile_90_100": 0.60}[confidence_bin]
    slope = {"decile_00_10": 0.00, "quintile_00_20": 0.00,
             "decile_50_60": 0.05, "quintile_40_60": 0.05,
             "decile_90_100": -0.04}[confidence_bin]
    return round(base + lift + slope * severity, 6)


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
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_ids = [int(value) for value in image_ids]
    severities = [int(value) for value in severities]
    rows = []
    for signal, scheme, confidence_bin, scope in SERIES:
        if (signal, confidence_bin, scope) in drop:
            continue
        for aggregation in AGGREGATIONS:
            for image_id in image_ids:
                for severity in severities:
                    rows.append({
                        "image_id": image_id, "severity": severity, "signal": signal,
                        "bucket_scheme": scheme, "confidence_bin": confidence_bin,
                        "membership_mode": membership_mode, "padding_mode": padding_mode,
                        "aggregation": aggregation, "score_scope": scope,
                        "source_partition": partition,
                        "score": score(image_id, severity, confidence_bin, signal, scope),
                        "selected_count": 30, "clean_overlap": 0.2,
                        "fully_measured": True, "signed_spearman": 0.6,
                        "absolute_spearman": 0.6, "direction": "increasing",
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
