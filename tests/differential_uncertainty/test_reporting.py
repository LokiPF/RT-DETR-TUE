import copy
import json

import pandas as pd
import pytest

from differential_uncertainty.config import ExperimentConfig
from differential_uncertainty.evaluation import evaluate_rows
from differential_uncertainty.reporting import REPORT_FILES, write_report


def _rows():
    rows = []
    for image_id, offset in (("a", 0.0), ("b", 0.2), ("c", -0.1)):
        for severity in range(6):
            persistence_reference = 2.0 - 0.1 * severity + offset
            persistence_responsive = 1.0 + 0.2 * severity + offset
            confidence_reference = 0.2 + 0.05 * severity
            confidence_responsive = 0.8 - 0.05 * severity
            rows.append({
                "image_id": image_id, "severity": severity,
                "padded_count": 0, "valid_count": 20,
                "reference_count": 2, "responsive_count": 2,
                "persistence_reference": persistence_reference,
                "persistence_responsive": persistence_responsive,
                "persistence_relative_gap": 2 * (persistence_responsive - persistence_reference)
                / (persistence_responsive + persistence_reference),
                "confidence_reference": confidence_reference,
                "confidence_responsive": confidence_responsive,
                "confidence_relative_gap": 2 * (confidence_responsive - confidence_reference)
                / (confidence_responsive + confidence_reference),
            })
    return rows


def _inputs():
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=5, query_count=20, persistence_dim=7,
        bootstrap_samples=100,
    )
    rows = _rows()
    evaluation = evaluate_rows(rows, config)
    provenance = {
        "checkpoint_sha256": "abc",
        "config": config.scientific_dict(),
        "corruption": {
            "name": "gaussian_blur",
            "severities": [
                {"level": level, "parameter": radius}
                for level, radius in enumerate(config.blur_radii)
            ],
        },
    }
    return rows, evaluation, provenance


def test_report_writes_the_exact_dedicated_bundle(tmp_path):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)
    actual = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    assert actual == sorted(REPORT_FILES)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["evaluation"]["series"]["persistence_relative_gap"]["orientation"] == 1
    text = (output / "report.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "what was tested", "relative gap", "nearest clean", "sigmoid", "spearman",
        "3.5 / 4 = 0.875", "not a probability", "adjacent consistency",
        "paired bootstrap", "does not measure map",
    ):
        assert phrase in text


def test_csv_and_json_numbers_reconcile_with_the_supplied_results(tmp_path):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)

    score_frame = pd.read_csv(output / "per-image-scores.csv")
    expected_scores = pd.DataFrame(rows).sort_values(
        ["image_id", "severity"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        score_frame, expected_scores, check_exact=False, rtol=1e-12, atol=1e-12
    )

    metric_frame = pd.read_csv(output / "metrics.csv").set_index("series")
    assert set(metric_frame.index) == set(evaluation["series"])
    for name, summary in evaluation["series"].items():
        for key, expected in summary.items():
            if key in {"auroc_by_severity", "severity_statistics", "field"}:
                continue
            assert metric_frame.loc[name, key] == pytest.approx(expected)
        for severity, expected in summary["auroc_by_severity"].items():
            assert metric_frame.loc[name, f"auroc_severity_{severity}"] == pytest.approx(expected)

    bootstrap = pd.read_csv(output / "bootstrap-comparisons.csv")
    expected_bootstrap = pd.DataFrame(evaluation["bootstrap_comparisons"])
    pd.testing.assert_frame_equal(
        bootstrap, expected_bootstrap, check_exact=False, rtol=1e-12, atol=1e-12
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary == json.loads(json.dumps({
        "schema_version": 1,
        "provenance": provenance,
        "evaluation": evaluation,
    }))


def test_report_explains_calculations_limits_and_uses_concrete_run_values(tmp_path):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)

    text = (output / "report.md").read_text(encoding="utf-8")
    lower = text.lower()
    example = sorted(rows, key=lambda row: (row["image_id"], row["severity"]))[0]
    primary = evaluation["series"]["persistence_relative_gap"]
    confidence = evaluation["series"]["confidence_relative_gap"]
    for concrete in (
        f"{example['persistence_reference']:.6f}",
        f"{example['persistence_responsive']:.6f}",
        f"{example['persistence_relative_gap']:.6f}",
        f"{example['confidence_reference']:.6f}",
        f"{example['confidence_responsive']:.6f}",
        f"{example['confidence_relative_gap']:.6f}",
        f"{primary['median_signed_spearman']:+.3f}",
        f"{primary['oriented_adjacent_consistency']:.1%}",
    ):
        assert concrete in text
    for severity in range(1, 6):
        assert (
            f"| {severity} | {primary['auroc_by_severity'][severity]:.3f} | "
            f"{confidence['auroc_by_severity'][severity]:.3f} |"
        ) in text
    for phrase in (
        "level 0 is the clean image",
        "same query ids",
        "1 - 0.90 = 0.10",
        "maximum sigmoid",
        "not softmax",
        "scale independent",
        "did not prove that it was better than a raw subtraction",
        "each severity",
        "clean scores from all images",
        "corrupted scores from all images",
        "tie worth half",
        "0.5 is chance",
        "strongest corruption",
        "all six levels",
        "does not prove a universal effect",
        "detection accuracy",
        "calibration",
        "object labels",
        "padded",
    ):
        assert phrase in lower


def test_existing_identical_bundle_is_a_safe_resumable_noop(tmp_path):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)
    before = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }

    write_report(
        output, list(reversed(rows)), copy.deepcopy(evaluation),
        copy.deepcopy(provenance),
    )

    after = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*") if path.is_file()
    }
    assert after == before
    assert not list(tmp_path.glob(".report.staging-*"))


@pytest.mark.parametrize("damage", ["missing", "corrupt", "extra"])
def test_existing_incomplete_or_corrupt_bundle_is_rejected_without_overwrite(
    tmp_path, damage
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)
    original_summary = (output / "summary.json").read_bytes()
    if damage == "missing":
        (output / "metrics.csv").unlink()
    elif damage == "corrupt":
        (output / "metrics.csv").write_text("broken\n", encoding="utf-8")
    else:
        (output / "unexpected.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="existing report bundle differs"):
        write_report(output, rows, evaluation, provenance)

    assert (output / "summary.json").read_bytes() == original_summary
    assert not list(tmp_path.glob(".report.staging-*"))


def test_existing_matching_provenance_but_different_requested_results_is_rejected(
    tmp_path,
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)
    different_rows = copy.deepcopy(rows)
    for row in different_rows:
        row["confidence_reference"] += 0.01
        row["confidence_relative_gap"] = 2 * (
            row["confidence_responsive"] - row["confidence_reference"]
        ) / (row["confidence_responsive"] + row["confidence_reference"])
    different_evaluation = evaluate_rows(
        different_rows,
        ExperimentConfig.for_tests(
            bank_capacity=20, k=5, query_count=20, persistence_dim=7,
            bootstrap_samples=100,
        ),
    )

    with pytest.raises(ValueError, match="existing report bundle differs"):
        write_report(output, different_rows, different_evaluation, provenance)


def test_existing_different_provenance_is_rejected_without_overwrite(tmp_path):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)
    before = (output / "summary.json").read_bytes()
    different = copy.deepcopy(provenance)
    different["checkpoint_sha256"] = "different"

    with pytest.raises(ValueError, match="existing report bundle differs"):
        write_report(output, rows, evaluation, different)

    assert (output / "summary.json").read_bytes() == before


class _StopNow(BaseException):
    pass


def test_staging_is_same_parent_and_is_cleaned_after_baseexception(
    tmp_path, monkeypatch
):
    import differential_uncertainty.reporting as reporting

    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"

    def stop(frame, evaluated, directory):
        assert directory.parent.parent == output.parent
        assert directory.parent.name.startswith(".report.staging-")
        raise _StopNow("interrupt")

    monkeypatch.setattr(reporting, "_write_figures", stop)
    with pytest.raises(_StopNow, match="interrupt"):
        write_report(output, rows, evaluation, provenance)

    assert not output.exists()
    assert not list(tmp_path.glob(".report.staging-*"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows, evaluation, provenance: rows.__setitem__(
            0, {**rows[0], "persistence_relative_gap": float("nan")}
        ), "finite"),
        (
            lambda rows, evaluation, provenance: rows.pop(),
            "severities 0 through 5",
        ),
        (
            lambda rows, evaluation, provenance: evaluation["series"]
            ["persistence_relative_gap"].__setitem__("macro_auroc", float("nan")),
            "finite",
        ),
        (
            lambda rows, evaluation, provenance:
            provenance.pop("checkpoint_sha256"),
            "checkpoint_sha256",
        ),
        (
            lambda rows, evaluation, provenance: provenance["corruption"]
            ["severities"].pop(),
            "levels 0 through 5",
        ),
    ],
)
def test_invalid_rows_evaluation_or_provenance_never_publish(
    tmp_path, mutate, message
):
    rows, evaluation, provenance = _inputs()
    mutate(rows, evaluation, provenance)

    with pytest.raises(ValueError, match=message):
        write_report(tmp_path / "report", rows, evaluation, provenance)

    assert not (tmp_path / "report").exists()
    assert not list(tmp_path.glob(".report.staging-*"))


def test_evaluation_must_describe_the_supplied_rows(tmp_path):
    rows, evaluation, provenance = _inputs()
    evaluation["series"]["persistence_relative_gap"]["macro_auroc"] += 0.01

    with pytest.raises(ValueError, match="does not match supplied rows"):
        write_report(tmp_path / "report", rows, evaluation, provenance)
