import json
import math
import warnings
from pathlib import Path

import pytest

from src.scene_uncertainty import reporting
from src.scene_uncertainty.reporting import (
    find_duplicate_result_key,
    read_result_csv,
    write_report,
    write_result_csv,
)


def make_rows():
    rows = []
    for severity in range(6):
        rows.append({
            "image_id": 1,
            "severity": severity,
            "raw_score": float(severity),
            "layer_scores": {0: severity + 0.1, 1: severity + 0.2, 2: severity + 0.3},
            "valid": True,
            "source_partition": "tuning",
            "policy": "all",
            "aggregation": "mean",
            "selected_count": 2,
            "selected_query_ids": [0, 1],
            "matched_predictions": {10: 2 if severity < 3 else 5},
        })
    return rows


def make_trend_rows(policy, scores):
    """One image swept over severities 0..len(scores)-1, unscored wherever `scores` is nan."""
    rows = []
    for severity, score in enumerate(scores):
        scored = not math.isnan(score)
        rows.append({
            "image_id": 1,
            "severity": severity,
            "raw_score": score,
            "layer_scores": {0: score} if scored else {},
            "valid": scored,
            "source_partition": "tuning",
            "policy": policy,
            "aggregation": "mean",
            "selected_count": 2 if scored else 0,
            "selected_query_ids": [0, 1] if scored else [],
            "matched_predictions": {10: 2},
        })
    return rows


def test_structured_csv_fields_round_trip(tmp_path: Path):
    path = tmp_path / "results.csv"
    write_result_csv(make_rows(), path)
    loaded = read_result_csv(path)
    assert loaded[0]["selected_query_ids"] == [0, 1]
    assert loaded[0]["layer_scores"] == {"0": 0.1, "1": 0.2, "2": 0.3}
    assert loaded[0]["matched_predictions"] == {"10": 2}


def test_write_report_creates_raw_relative_and_switch_outputs(tmp_path: Path):
    write_report(make_rows(), tmp_path, run_metadata={"feature_cache_id": "cache", "bank_id": "bank"})
    for name in ("per_scene.csv", "summary.json", "run_metadata.json", "raw_trend.png", "relative_trend.png", "class_switch.png"):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    combined = next(group for group in summary["groups"] if group["score_scope"] == "combined")
    assert combined["median_spearman"] == 1.0
    assert combined["class_switch_step_count"] == 1
    assert combined["no_switch_step_count"] == 4
    assert summary["run_metadata"]["feature_cache_id"] == "cache"


def test_collapsed_config_is_distinguishable_from_a_surviving_one(tmp_path: Path):
    """A policy that stops selecting under blur must not publish as a perfect trend.

    Both configs below report `spearman == 1.0`, because the trend statistics only ever
    describe the severities that produced a score. The severity counts are what separate
    "rose across the whole sweep" from "rose over the half of the sweep it survived".
    """
    rows = (
        make_trend_rows("collapsing", [1.0, 2.0, 3.0, float("nan"), float("nan"), float("nan")])
        + make_trend_rows("surviving", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )
    write_report(rows, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    groups = {(group["policy"], group["score_scope"]): group for group in summary["groups"]}

    collapsing = groups[("collapsing", "combined")]
    surviving = groups[("surviving", "combined")]
    assert collapsing["median_spearman"] == surviving["median_spearman"] == 1.0
    assert collapsing["endpoint_increase_rate"] == surviving["endpoint_increase_rate"] == 1.0

    assert collapsing["scored_severity_count"] == 3
    assert collapsing["total_severity_count"] == 6
    assert collapsing["images_with_unscored_severities"] == 1
    assert surviving["scored_severity_count"] == 6
    assert surviving["total_severity_count"] == 6
    assert surviving["images_with_unscored_severities"] == 0

    assert collapsing["empty_selection_frequency"] == 0.5
    assert surviving["empty_selection_frequency"] == 0.0
    assert collapsing["median_selected_count_by_severity"]["3"] == 0.0

    # The per-layer scope collapses with the combined scope, so both of the collapsing
    # config's groups are flagged and neither of the surviving config's groups are.
    assert summary["diagnostics"]["group_count"] == 4
    assert summary["diagnostics"]["groups_with_unscored_severities"] == 2


def test_csv_round_trip_preserves_row_order_columns_and_unscored_rows(tmp_path: Path):
    rows = [
        {
            "image_id": 4,
            "severity": 5,
            # Not exactly representable: pandas' default float converter reads this back
            # as 0.3, and every published statistic in Task 12 is derived from what this
            # reader returns rather than from what the scorer produced.
            "raw_score": 0.1 + 0.2,
            "layer_scores": {0: 1.5},
            "clean_scaled_layer_scores": {0: -1.25},
            "valid": True,
            "source_partition": "tuning",
            "policy": "threshold_0.5",
            "aggregation": "mean",
            "selected_count": 1,
            "selected_query_ids": [7],
            "matched_predictions": {10: 2},
        },
        {
            "image_id": 4,
            "severity": 0,
            "raw_score": float("nan"),
            "layer_scores": {},
            "clean_scaled_layer_scores": {},
            "valid": False,
            "source_partition": "tuning",
            "policy": "threshold_0.5",
            "aggregation": "mean",
            "selected_count": 0,
            "selected_query_ids": [],
            "matched_predictions": {},
        },
    ]
    path = tmp_path / "results.csv"
    write_result_csv(rows, path)
    loaded = read_result_csv(path)

    assert [row["severity"] for row in loaded] == [5, 0]
    assert set(loaded[0]) == set(rows[0])
    assert loaded[0]["raw_score"] == 0.1 + 0.2
    assert loaded[0]["valid"] is True
    assert loaded[0]["clean_scaled_layer_scores"] == {"0": -1.25}
    assert math.isnan(loaded[1]["raw_score"])
    assert loaded[1]["valid"] is False
    assert loaded[1]["layer_scores"] == {}
    assert loaded[1]["selected_query_ids"] == []
    assert loaded[1]["matched_predictions"] == {}


def test_writers_leave_no_temporary_files_behind(tmp_path: Path):
    write_report(make_rows(), tmp_path, run_metadata={"bank_id": "bank"})
    assert sorted(path.name for path in tmp_path.glob("*.tmp")) == []


def test_plots_render_to_png_without_a_display(tmp_path: Path):
    write_report(make_rows(), tmp_path)
    for name in ("raw_trend.png", "relative_trend.png", "class_switch.png"):
        content = (tmp_path / name).read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(content) > 1000


def test_run_metadata_is_written_as_its_own_document(tmp_path: Path):
    metadata = {"feature_cache_id": "cache", "bank_id": "bank", "checkpoint_sha256": "abc"}
    write_report(make_rows(), tmp_path, run_metadata=metadata)
    assert json.loads((tmp_path / "run_metadata.json").read_text()) == metadata


def test_summary_reports_the_combined_scope_and_one_group_per_layer(tmp_path: Path):
    write_report(make_rows(), tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert sorted(group["score_scope"] for group in summary["groups"]) == [
        "combined", "layer_0", "layer_1", "layer_2"
    ]
    layer = next(group for group in summary["groups"] if group["score_scope"] == "layer_1")
    assert layer["median_spearman"] == 1.0
    assert layer["image_count"] == 1


def test_undefined_statistics_are_written_as_null_not_nan(tmp_path: Path):
    rows = make_trend_rows("threshold_0.5", [float("nan")] * 6)
    # Nothing was scored, so every trend plot is empty. Rendering one must stay silent:
    # an unguarded legend call on an empty axis warns, and warnings in a report run are
    # noise a reader has to learn to ignore.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_report(rows, tmp_path)
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text
    group = json.loads(text)["groups"][0]
    assert group["median_spearman"] is None
    assert group["mean_adjacent_monotonicity"] is None
    assert group["mean_violation_magnitude"] is None
    # `monotonicity_metrics` reports `endpoint_increase: False` -- not `nan` -- when fewer
    # than two severities survived, so this is the one trend statistic a plain mean cannot
    # drop. Averaged over the images that were actually measured, it goes null with the
    # other three instead of publishing "no image rose" about zero measurements.
    assert group["endpoint_increase_rate"] is None
    assert group["mean_adjacent_query_overlap"] is None
    assert group["class_switch_monotonicity"] is None
    assert group["image_count"] == 1
    assert group["scored_image_count"] == 0
    assert group["empty_selection_frequency"] == 1.0
    assert group["scored_severity_count"] == 0
    assert group["total_severity_count"] == 6


def test_a_single_severity_produces_no_adjacent_steps(tmp_path: Path):
    write_report(make_trend_rows("all", [1.0]), tmp_path)
    group = json.loads((tmp_path / "summary.json").read_text())["groups"][0]
    assert group["class_switch_step_count"] == 0
    assert group["no_switch_step_count"] == 0
    assert group["median_spearman"] is None
    assert group["scored_severity_count"] == 1
    assert group["total_severity_count"] == 1


def test_trend_statistics_report_the_images_they_were_measured_over(tmp_path: Path):
    """One image rose across the whole sweep; three collapsed after severity 0.

    Every trend statistic here is an n=1 result. `image_count` says 4 because four images
    were swept, so `scored_image_count` has to say 1, or a reader takes the median and the
    endpoint rate for a four-image finding.
    """
    rows = list(make_trend_rows("all", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    for image_id, collapsed in enumerate(
        [[1.0] + [float("nan")] * 5] * 3, start=2
    ):
        for row in make_trend_rows("all", collapsed):
            rows.append({**row, "image_id": image_id})
    write_report(rows, tmp_path)
    group = next(
        group for group in json.loads((tmp_path / "summary.json").read_text())["groups"]
        if group["score_scope"] == "combined"
    )
    assert group["image_count"] == 4
    assert group["scored_image_count"] == 1
    assert group["median_spearman"] == 1.0
    # Not 0.25: the three collapsed images never produced an endpoint to compare.
    assert group["endpoint_increase_rate"] == 1.0
    assert group["images_with_unscored_severities"] == 3


def test_empty_input_round_trips_as_no_rows_and_is_refused_by_write_report(tmp_path: Path):
    path = tmp_path / "results.csv"
    write_result_csv([], path)
    assert read_result_csv(path) == []
    with pytest.raises(ValueError, match="at least one"):
        write_report([], tmp_path / "report")


def test_write_report_refuses_a_duplicated_result_key(tmp_path: Path):
    """Concatenating two result CSVs is an obvious operator move and silently inflates.

    Every row is joined against its own severity-0 row, so a key present twice produces
    four severity-0 pairings instead of one and the adjacent-step diagnostics grow
    quadratically: with the guard removed, `write_report(make_rows() * 2)` publishes
    `no_switch_step_count` 22 instead of 4 (doubling would be 8) and
    `scored_severity_count` 12 instead of 6, while `median_spearman` still reads a
    confident 1.0. Nothing downstream can tell that apart from a real run, so it is
    refused here, by name.
    """
    rows = make_rows()
    with pytest.raises(ValueError, match=r"image_id=1, severity=0, policy=all"):
        write_report(rows + rows, tmp_path)
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "per_scene.csv").exists()


def test_the_duplicate_check_reports_the_first_repeated_key():
    rows = make_rows()
    assert find_duplicate_result_key(rows) is None
    repeated = dict(rows[3])
    assert find_duplicate_result_key([*rows, repeated]) == (1, 3, "all", "mean", "tuning")


def test_rows_that_differ_in_any_key_column_are_not_duplicates():
    """The key is all five columns; two policies over one image are a normal report."""
    rows = make_rows() + [{**row, "policy": "top10"} for row in make_rows()]
    assert find_duplicate_result_key(rows) is None


def test_the_trend_plots_say_that_their_curves_are_in_two_units(tmp_path: Path, monkeypatch):
    """`raw_trend.png` draws `layer_*` and `combined` on one axis in two different units.

    `combined` is `raw_score`, the mean of the per-layer scores after each was centred and
    divided by its clean-distance scale; the `layer_*` curves are the same scores before
    that standardisation. The statistics in `summary.json` are unaffected -- they are all
    invariant under a positive affine map -- but a reader comparing two curves by eye is
    not, and these PNGs travel on their own. The label is the fix, so it is asserted.
    """
    drawn = []
    real_close = reporting.plt.close

    def spy(figure):
        axis = figure.axes[0]
        drawn.append((axis.get_ylabel(), axis.get_title()))
        real_close(figure)

    monkeypatch.setattr(reporting.plt, "close", spy)
    write_report(make_rows(), tmp_path)

    raw_label, raw_note = drawn[0]
    relative_label, relative_note = drawn[1]
    assert "mixed units" in raw_label.lower()
    assert "mixed units" in relative_label.lower()
    for note in (raw_note, relative_note):
        assert "unscaled" in note.lower()
        assert "standardised" in note.lower()
        assert "layer_" in note and "combined" in note
