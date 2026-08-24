import copy
import dis
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from differential_uncertainty import reporting
from differential_uncertainty.artifacts import _open_regular_file
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
        assert directory.parent.resolve().parent == output.parent
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


def test_series_orientations_must_match_the_fixed_provenance_mapping(tmp_path):
    rows, evaluation, provenance = _inputs()
    provenance["config"]["orientations"]["persistence_relative_gap"] = -1

    with pytest.raises(ValueError, match="fixed orientation"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


@pytest.mark.parametrize(
    ("field", "delta"),
    [("ci_low", -0.001), ("ci_high", 0.001)],
)
def test_bootstrap_interval_must_reconcile_with_rows_seed_and_samples(
    tmp_path, field, delta
):
    rows, evaluation, provenance = _inputs()
    evaluation["bootstrap_comparisons"][0][field] += delta

    with pytest.raises(ValueError, match="bootstrap comparison.*does not match"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


def test_bootstrap_seed_must_reconcile_with_provenance(tmp_path):
    rows, evaluation, provenance = _inputs()
    provenance["config"]["bootstrap_seed"] += 1

    with pytest.raises(ValueError, match="bootstrap comparison.*does not match"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows.__setitem__(
                0, {
                    **rows[0],
                    "padded_count": 11,
                    "valid_count": 9,
                },
            ),
            "at least ten valid",
        ),
        (
            lambda rows: rows.__setitem__(
                0, {
                    **rows[0],
                    "confidence_reference": -0.1,
                    "confidence_relative_gap": 2 * (
                        rows[0]["confidence_responsive"] + 0.1
                    ) / (rows[0]["confidence_responsive"] - 0.1),
                },
            ),
            r"confidence.*\[0, 1\]",
        ),
        (
            lambda rows: rows.__setitem__(
                0, {
                    **rows[0],
                    "persistence_reference": -0.1,
                    "persistence_relative_gap": 2 * (
                        rows[0]["persistence_responsive"] + 0.1
                    ) / (rows[0]["persistence_responsive"] - 0.1),
                },
            ),
            "persistence means must be non-negative",
        ),
    ],
)
def test_impossible_score_values_are_rejected_before_reporting(
    tmp_path, mutation, message
):
    rows, evaluation, provenance = _inputs()
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        write_report(tmp_path / "report", rows, evaluation, provenance)


def test_group_counts_follow_configured_tensor_split_deciles(tmp_path):
    rows, evaluation, provenance = _inputs()
    for row in rows:
        row["padded_count"] = 1
        row["valid_count"] = 19
        row["reference_count"] = 2

    with pytest.raises(ValueError, match="configured decile"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


def test_padding_and_decile_counts_are_constant_across_an_image_curve(tmp_path):
    rows, evaluation, provenance = _inputs()
    changed = rows[1]
    changed["padded_count"] = 2
    changed["valid_count"] = 18
    changed["reference_count"] = 1

    with pytest.raises(ValueError, match="same counts across all six"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.pop("class_count"),
        lambda config: config.__setitem__("unexpected", 1),
        lambda config: config.__setitem__("image_size", [640]),
        lambda config: config.__setitem__("persistence_dim", 0),
        lambda config: config.__setitem__("bank_capacity", 4),
        lambda config: config.__setitem__("bank_seed", -1),
        lambda config: config.__setitem__("bank_chunk_size", 0),
        lambda config: config.__setitem__("reference_decile", 10),
        lambda config: config.__setitem__("responsive_decile", 9),
        lambda config: config.__setitem__("default_blur_radii", [0.0, 1.0]),
        lambda config: config.__setitem__("feature_normalization", "unit"),
    ],
)
def test_provenance_requires_the_full_valid_scientific_config(
    tmp_path, mutation
):
    rows, evaluation, provenance = _inputs()
    mutation(provenance["config"])

    with pytest.raises(ValueError, match="scientific config"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


@pytest.mark.parametrize(
    "unsafe_id",
    ["=2+3", "+cmd", "-2", "@SUM(A1:A2)", "  =2+3"],
)
def test_csv_image_ids_are_formula_safe(tmp_path, unsafe_id):
    rows, _, provenance = _inputs()
    for row in rows:
        if row["image_id"] == "a":
            row["image_id"] = unsafe_id
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=5, query_count=20, persistence_dim=7,
        bootstrap_samples=100,
    )
    evaluation = evaluate_rows(rows, config)
    output = tmp_path / "report"

    write_report(output, rows, evaluation, provenance)

    observed = pd.read_csv(
        output / "per-image-scores.csv", dtype={"image_id": str}
    )
    assert f"'{unsafe_id}" in set(observed["image_id"])


def test_markdown_escapes_dynamic_image_and_provenance_text(tmp_path):
    rows, _, provenance = _inputs()
    dangerous = "`</code><script>alert(1)</script>**"
    for row in rows:
        if row["image_id"] == "a":
            row["image_id"] = dangerous
    provenance["checkpoint_sha256"] = "`<script>checkpoint</script>"
    provenance["corruption"]["name"] = "<img src=x onerror=alert(1)>_blur"
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=5, query_count=20, persistence_dim=7,
        bootstrap_samples=100,
    )
    evaluation = evaluate_rows(rows, config)

    write_report(tmp_path / "report", rows, evaluation, provenance)

    text = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "<script>" not in text
    assert "</code>" not in text
    assert "<img " not in text
    assert "&lt;script&gt;" in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("image_id", "a\nb"),
        ("image_id", "a" * 257),
        ("checkpoint_sha256", "x\u0000y"),
        ("checkpoint_sha256", "x" * 257),
        ("corruption_name", "x\ny"),
        ("corruption_name", "x" * 129),
    ],
)
def test_dynamic_text_rejects_controls_and_excessive_lengths(
    tmp_path, target, value
):
    rows, evaluation, provenance = _inputs()
    if target == "image_id":
        for row in rows:
            if row["image_id"] == "a":
                row["image_id"] = value
        config = ExperimentConfig.for_tests(
            bank_capacity=20, k=5, query_count=20, persistence_dim=7,
            bootstrap_samples=100,
        )
        evaluation = evaluate_rows(rows, config)
    elif target == "checkpoint_sha256":
        provenance["checkpoint_sha256"] = value
    else:
        provenance["corruption"]["name"] = value

    with pytest.raises(ValueError, match="text"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


def _reverse_mappings(value):
    if isinstance(value, dict):
        return {
            key: _reverse_mappings(child)
            for key, child in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mappings(child) for child in value]
    return value


def test_json_native_reordered_evaluation_renders_canonical_bundle(tmp_path):
    rows, evaluation, provenance = _inputs()
    baseline = tmp_path / "baseline"
    reordered = tmp_path / "reordered"
    write_report(baseline, rows, evaluation, provenance)

    json_native = json.loads(json.dumps(evaluation))
    write_report(
        reordered,
        list(reversed(rows)),
        _reverse_mappings(json_native),
        _reverse_mappings(provenance),
    )

    for relative in REPORT_FILES:
        assert (reordered / relative).read_bytes() == (
            baseline / relative
        ).read_bytes()

    write_report(
        baseline,
        rows,
        _reverse_mappings(json_native),
        _reverse_mappings(provenance),
    )


def test_existing_bundle_comparison_never_uses_path_read_bytes(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)

    def forbidden_read(_self):
        raise AssertionError("comparison used an unpinned path read")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    write_report(output, rows, evaluation, provenance)


def test_existing_bundle_rejects_a_raced_file_symlink_to_fifo(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    fifo = tmp_path / "fifo"
    write_report(output, rows, evaluation, provenance)
    os.mkfifo(fifo)
    triggered = False

    def swap_then_open(path, *, directory_fd=None, error_message):
        nonlocal triggered
        if not triggered and os.fspath(path) == "summary.json":
            triggered = True
            os.rename(
                "summary.json", "summary.original",
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
            )
            os.symlink(fifo, "summary.json", dir_fd=directory_fd)
        return _open_regular_file(
            path,
            directory_fd=directory_fd,
            error_message=error_message,
        )

    monkeypatch.setattr(
        reporting, "_open_regular_file", swap_then_open, raising=False
    )
    with pytest.raises(ValueError, match="existing report bundle differs"):
        write_report(output, rows, evaluation, provenance)
    assert triggered


def test_existing_bundle_rejects_file_replacement_after_safe_open(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)
    triggered = False

    def replace_after_open(path, *, directory_fd=None, error_message):
        nonlocal triggered
        handle = _open_regular_file(
            path,
            directory_fd=directory_fd,
            error_message=error_message,
        )
        if not triggered and os.fspath(path) == "metrics.csv":
            triggered = True
            os.rename(
                "metrics.csv", "metrics.original",
                src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
            )
            shutil.copyfile(
                f"/proc/self/fd/{directory_fd}/metrics.original",
                f"/proc/self/fd/{directory_fd}/metrics.csv",
            )
        return handle

    monkeypatch.setattr(reporting, "_open_regular_file", replace_after_open)
    with pytest.raises(ValueError, match="existing report bundle differs"):
        write_report(output, rows, evaluation, provenance)
    assert triggered


@pytest.mark.parametrize("replace_figures", [False, True])
def test_existing_bundle_rejects_raced_directory_identity(
    tmp_path, monkeypatch, replace_figures
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    moved = tmp_path / ("moved-figures" if replace_figures else "moved-root")
    write_report(output, rows, evaluation, provenance)
    triggered = False

    def replace_then_open(path, *, directory_fd=None, error_message):
        nonlocal triggered
        name = os.fspath(path)
        should_replace = (
            replace_figures and name.endswith(".png")
        ) or (not replace_figures and name == "per-image-scores.csv")
        if not triggered and should_replace:
            triggered = True
            target = output / "figures" if replace_figures else output
            target.rename(moved)
            shutil.copytree(moved, target)
        return _open_regular_file(
            path,
            directory_fd=directory_fd,
            error_message=error_message,
        )

    monkeypatch.setattr(
        reporting, "_open_regular_file", replace_then_open, raising=False
    )
    with pytest.raises(ValueError, match="existing report bundle differs"):
        write_report(output, rows, evaluation, provenance)
    assert triggered


def test_staging_call_store_interrupt_cannot_strand_directory(tmp_path):
    rows, evaluation, provenance = _inputs()
    target_code = reporting._StagingOwner.__enter__.__code__
    instructions = list(dis.get_instructions(target_code))
    store_offset = next(
        instruction.offset
        for instruction in instructions
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "_temporary"
    )

    def trace(frame, event, _arg):
        if frame.f_code is target_code:
            frame.f_trace_opcodes = True
            if event == "opcode" and frame.f_lasti == store_offset:
                raise KeyboardInterrupt("injected after staging allocation")
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="staging allocation"):
            write_report(tmp_path / "report", rows, evaluation, provenance)
    finally:
        sys.settrace(previous)
    assert not list(tmp_path.glob(".report.staging-*"))


def test_replaced_staging_path_is_not_published_or_destructively_cleaned(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    original_write = reporting._write_bundle
    replacement = None
    moved = None

    def replace_staging(*args, **kwargs):
        nonlocal replacement, moved
        original_write(*args, **kwargs)
        replacement = Path(args[0]).resolve(strict=True)
        moved = replacement.with_name(f"{replacement.name}.moved")
        replacement.rename(moved)
        replacement.mkdir()
        (replacement / "sentinel").write_text("replacement")

    monkeypatch.setattr(reporting, "_write_bundle", replace_staging)
    with pytest.raises(ValueError, match="staging.*changed"):
        write_report(tmp_path / "report", rows, evaluation, provenance)
    assert not (tmp_path / "report").exists()
    assert replacement is not None and (replacement / "sentinel").is_file()
    assert moved is not None and not moved.exists()


def test_replaced_parent_path_fails_closed_and_cleans_pinned_staging(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    parent = tmp_path / "parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-parent"
    original_write = reporting._write_bundle

    def replace_parent(*args, **kwargs):
        original_write(*args, **kwargs)
        parent.rename(moved_parent)
        parent.mkdir()

    monkeypatch.setattr(reporting, "_write_bundle", replace_parent)
    with pytest.raises(ValueError, match="parent.*changed"):
        write_report(parent / "report", rows, evaluation, provenance)
    assert not (parent / "report").exists()
    assert not list(moved_parent.glob(".report.staging-*"))


def test_cleanup_failure_does_not_mask_original_exception(tmp_path, monkeypatch):
    rows, evaluation, provenance = _inputs()

    class OriginalFailure(BaseException):
        pass

    def fail_write(*_args, **_kwargs):
        raise OriginalFailure("original")

    def fail_cleanup(*_args, **_kwargs):
        raise RuntimeError("cleanup")

    monkeypatch.setattr(reporting, "_write_bundle", fail_write)
    monkeypatch.setattr(reporting, "_cleanup_staging", fail_cleanup)
    with pytest.raises(OriginalFailure, match="original"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


def test_plot_failure_closes_only_figures_created_by_this_call(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    frame = reporting._validated_frame(rows, config=provenance["config"])
    caller = plt.figure()
    before = set(plt.get_fignums())

    def fail_scatter(*_args, **_kwargs):
        raise RuntimeError("plot failed")

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fail_scatter)
    try:
        with pytest.raises(RuntimeError, match="plot failed"):
            reporting._write_figures(frame, evaluation, tmp_path)
        assert set(plt.get_fignums()) == before
    finally:
        plt.close(caller)
