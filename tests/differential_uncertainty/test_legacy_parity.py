import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from differential_uncertainty.evaluation import summarize_series


ARCHIVE = Path(
    "/home/yuchen/YuchenZ/UE/within-image-contrast-run-record/"
    "within_image_contrast/per_scene_contrasts.csv"
)


@lru_cache(maxsize=2)
def _archived_rows(signal):
    with ARCHIVE.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    return tuple(
        {
            "image_id": row["image_id"],
            "severity": int(row["severity"]),
            "score": float(row["score"]),
        }
        for row in source
        if row["arm"] == "decile_90_100__50_60"
        and row["aggregation"] == "mean"
        and row["method"] == "relative_gap"
        and row["signal"] == signal
    )


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="archived completed run unavailable")
def test_archived_relative_gap_rows_are_the_exact_paired_complete_roster():
    persistence = _archived_rows("persistence")
    confidence = _archived_rows("confidence")
    assert len(persistence) == len(confidence) == 250 * 6

    for rows in (persistence, confidence):
        pairs = Counter((row["image_id"], row["severity"]) for row in rows)
        assert set(pairs.values()) == {1}
        image_ids = {row["image_id"] for row in rows}
        assert len(image_ids) == 250
        assert {
            row["severity"] for row in rows
        } == {0, 1, 2, 3, 4, 5}
        assert all(
            {row["severity"] for row in rows if row["image_id"] == image_id}
            == {0, 1, 2, 3, 4, 5}
            for image_id in image_ids
        )

    assert {row["image_id"] for row in persistence} == {
        row["image_id"] for row in confidence
    }


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="archived completed run unavailable")
def test_persistence_relative_gap_reproduces_the_completed_run():
    result = summarize_series(_archived_rows("persistence"), "score", orientation=1)
    assert result["median_signed_spearman"] == pytest.approx(
        0.8285714285714287
    )
    assert result["auroc_by_severity"] == pytest.approx(
        {
            1: 0.510112,
            2: 0.565776,
            3: 0.6668,
            4: 0.893072,
            5: 0.961344,
        }
    )
    assert result["macro_auroc"] == pytest.approx(0.7194208)


@pytest.mark.skipif(not ARCHIVE.is_file(), reason="archived completed run unavailable")
def test_matched_confidence_relative_gap_reproduces_the_completed_run():
    result = summarize_series(_archived_rows("confidence"), "score", orientation=-1)
    assert result["median_signed_spearman"] == pytest.approx(
        -0.9142857142857144
    )
    assert result["auroc_by_severity"] == pytest.approx(
        {
            1: 0.521424,
            2: 0.56312,
            3: 0.633056,
            4: 0.78968,
            5: 0.881872,
        }
    )
    assert result["macro_auroc"] == pytest.approx(0.6778304)
