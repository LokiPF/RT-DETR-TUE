import copy
import dis
import json
import os
import shutil
import sys
import threading
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
            direct_confidence_mean = 0.60 - 0.04 * severity + 0.01 * offset
            direct_confidence_max = 0.85 - 0.03 * severity + 0.01 * offset
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
                "direct_confidence_mean": direct_confidence_mean,
                "direct_confidence_max": direct_confidence_max,
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
        "runtime": {
            "device": {"type": "cpu", "index": None},
            "batch_size": 2,
            "shard_size": 2,
            "libraries": {
                "python": "3.test",
                "pytorch": "2.test",
                "torchvision": "0.test",
                "numpy": "1.test",
                "scipy": "1.test",
                "pillow": "11.test",
            },
            "cuda": {"runtime": None, "cudnn": None, "gpu_name": None,
                     "compute_capability": None},
        },
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
    corruption = summary["provenance"]["corruption"]
    assert summary["evaluation"]["series"]["direct_confidence_mean"]["orientation"] == -1
    assert summary["evaluation"]["series"]["direct_confidence_max"]["orientation"] == -1
    assert set(corruption) == {"name", "severities"}
    assert corruption["name"] == "gaussian_blur"
    text = (output / "report.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "what was tested", "relative gap", "nearest clean", "sigmoid", "spearman",
        "3.5 / 4 = 0.875", "not a probability", "adjacent consistency",
        "paired bootstrap", "does not measure map",
        "direct global confidence", "all valid queries",
        "lower confidence ranks as more corrupted",
        "corruption code or hidden settings change", "new output folder",
    ):
        assert phrase in text
    assert "authoritative build sha-256" not in text
    assert "corruption implementation was" not in text


def test_report_rejects_rows_missing_direct_confidence_mean(tmp_path):
    rows, evaluation, provenance = _inputs()
    del rows[0]["direct_confidence_mean"]
    with pytest.raises(ValueError, match="missing=.*direct_confidence_mean"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


def test_report_writes_through_an_active_pinned_parent_lease(tmp_path):
    rows, evaluation, provenance = _inputs()

    with reporting._DirectoryLease(
        tmp_path,
        message="pinned report parent changed",
    ) as parent:
        write_report(
            "report",
            rows,
            evaluation,
            provenance,
            parent=parent,
        )
        os.fstat(parent.fd)

    assert (tmp_path / "report" / "report.md").is_file()


def test_directory_lease_direct_acquisition_survives_fd_reuse(
    tmp_path, monkeypatch
):
    held_path = tmp_path / "held"
    raced_path = tmp_path / "raced"
    held_path.mkdir()
    raced_path.mkdir()
    held = reporting._DirectoryLease(held_path, message="held changed")
    held.__enter__()
    held_released = False
    try:
        reused_fd = held.fd
        direct_open = getattr(reporting, "_open_owned_directory", None)
        assert callable(direct_open)
        triggered = False

        def close_then_open(path, *, directory_fd=None):
            nonlocal held_released, triggered
            triggered = True
            held.__exit__(None, None, None)
            held_released = True
            owner = direct_open(path, directory_fd=directory_fd)
            assert owner.fd == reused_fd
            return owner

        monkeypatch.setattr(
            reporting, "_open_owned_directory", close_then_open
        )
        with reporting._DirectoryLease(
            raced_path, message="raced acquisition failed"
        ) as raced:
            assert raced.fd == reused_fd
        assert triggered
    finally:
        if not held_released:
            held.__exit__(None, None, None)


def test_directory_direct_open_call_store_interrupt_closes_fd(tmp_path):
    target_code = reporting._DirectoryLease.__enter__.__code__
    owner_stores = [
        instruction.offset
        for instruction in dis.get_instructions(target_code)
        if instruction.opname == "STORE_FAST"
        and instruction.argval == "owner"
    ]
    assert len(owner_stores) == 2
    direct_owner_store = owner_stores[1]
    before = {
        int(name)
        for name in os.listdir("/proc/self/fd")
        if name.isdigit() and Path(f"/proc/self/fd/{name}").exists()
    }
    interrupted = False

    def trace(frame, event, _argument):
        nonlocal interrupted
        if frame.f_code is target_code:
            frame.f_trace_opcodes = True
            if (
                not interrupted
                and event == "opcode"
                and frame.f_lasti == direct_owner_store
            ):
                interrupted = True
                raise KeyboardInterrupt("after direct directory open")
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match="direct directory open"):
            with reporting._DirectoryLease(
                tmp_path, message="directory changed"
            ):
                pass
    finally:
        sys.settrace(previous)
    after = {
        int(name)
        for name in os.listdir("/proc/self/fd")
        if name.isdigit() and Path(f"/proc/self/fd/{name}").exists()
    }

    assert interrupted
    assert after == before


class _DirectoryCloseInterrupted(BaseException):
    pass


def _live_file_descriptors():
    descriptors = set()
    for name in os.listdir("/proc/self/fd"):
        if not name.isdigit():
            continue
        descriptor = int(name)
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        descriptors.add(descriptor)
    return descriptors


def _fd_identity(descriptor):
    try:
        state = os.fstat(descriptor)
    except OSError:
        return None
    return state.st_dev, state.st_ino, state.st_mode


def _executed_opcode_offsets(code, action, expected=()):
    offsets = []

    def trace(frame, event, _argument):
        if frame.f_code is code:
            frame.f_trace_opcodes = True
            if event == "opcode":
                offsets.append(frame.f_lasti)
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        action()
    except expected:
        pass
    finally:
        sys.settrace(previous)
    return tuple(dict.fromkeys(offsets))


def _interrupt_at_opcode(code, target, action):
    interrupted = False

    def trace(frame, event, _argument):
        nonlocal interrupted
        if frame.f_code is code:
            frame.f_trace_opcodes = True
            if (
                not interrupted
                and event == "opcode"
                and frame.f_lasti == target
            ):
                interrupted = True
                raise _DirectoryCloseInterrupted(
                    f"directory close interrupted at opcode {target}"
                )
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(
            _DirectoryCloseInterrupted,
            match=f"opcode {target}",
        ):
            action()
    finally:
        sys.settrace(previous)
    assert interrupted


def _open_test_lease(path):
    lease = reporting._DirectoryLease(path, message="directory changed")
    lease.__enter__()
    return lease


def _assert_lease_can_open_in_another_thread(path):
    result = []

    def open_lease():
        try:
            with reporting._DirectoryLease(
                path, message="directory changed"
            ) as lease:
                os.fstat(lease.fd)
            result.append(None)
        except BaseException as error:
            result.append(error)

    thread = threading.Thread(target=open_lease, daemon=True)
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    assert result == [None]


def test_owned_directory_close_is_safe_at_every_executed_opcode(tmp_path):
    probe = _open_test_lease(tmp_path)
    code = type(probe.owner).close.__code__
    offsets = _executed_opcode_offsets(code, probe.owner.close)
    probe.__exit__(None, None, None)
    assert offsets

    baseline = _live_file_descriptors()
    for target in offsets:
        lease = _open_test_lease(tmp_path)
        owner = lease.owner
        descriptor = lease.fd
        identity = _fd_identity(descriptor)
        try:
            _interrupt_at_opcode(code, target, owner.close)
            if _fd_identity(descriptor) == identity:
                assert owner.fd == descriptor, target
            else:
                with pytest.raises(RuntimeError, match="closed"):
                    owner.fd
        finally:
            lease.__exit__(None, None, None)
            if _fd_identity(descriptor) == identity:
                os.close(descriptor)
        assert _live_file_descriptors() == baseline, target

    with reporting._DirectoryLease(
        tmp_path, message="directory changed"
    ) as subsequent:
        os.fstat(subsequent.fd)
    _assert_lease_can_open_in_another_thread(tmp_path)
    assert _live_file_descriptors() == baseline


def test_directory_lease_exit_is_safe_at_every_executed_opcode(tmp_path):
    probe = _open_test_lease(tmp_path)
    code = reporting._DirectoryLease.__exit__.__code__
    offsets = _executed_opcode_offsets(
        code, lambda: probe.__exit__(None, None, None)
    )
    assert offsets

    baseline = _live_file_descriptors()
    for target in offsets:
        lease = _open_test_lease(tmp_path)
        owner = lease.owner
        descriptor = lease.fd
        identity = _fd_identity(descriptor)
        try:
            _interrupt_at_opcode(
                code,
                target,
                lambda: lease.__exit__(None, None, None),
            )
            if _fd_identity(descriptor) == identity:
                assert lease.owner is owner, target
                assert lease.fd == descriptor, target
                assert owner.fd == descriptor, target
            else:
                with pytest.raises(RuntimeError, match="closed"):
                    owner.fd
        finally:
            lease.__exit__(None, None, None)
            if _fd_identity(descriptor) == identity:
                os.close(descriptor)
        assert _live_file_descriptors() == baseline, target

    with reporting._DirectoryLease(
        tmp_path, message="directory changed"
    ) as subsequent:
        os.fstat(subsequent.fd)
    _assert_lease_can_open_in_another_thread(tmp_path)
    assert _live_file_descriptors() == baseline


class _DirectoryCloseCallFailed(BaseException):
    pass


def test_owner_close_exception_paths_are_safe_at_every_executed_opcode(
    tmp_path, monkeypatch
):
    owned = tmp_path / "owned-exhaustive"
    unrelated = tmp_path / "unrelated-exhaustive"
    owned.mkdir()
    unrelated.mkdir()
    code = reporting._OwnedDirectoryDescriptor.close.__code__
    real_close = os.close

    def observe_before_close():
        lease = _open_test_lease(owned)

        def fail_before_close(_descriptor):
            raise _DirectoryCloseCallFailed("before close")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(reporting.os, "close", fail_before_close)
                offsets = _executed_opcode_offsets(
                    code, lease.owner.close, _DirectoryCloseCallFailed
                )
            return offsets
        finally:
            lease.__exit__(None, None, None)

    def observe_after_reuse():
        lease = _open_test_lease(owned)
        reused = None

        def close_reuse_then_fail(descriptor):
            nonlocal reused
            real_close(descriptor)
            reused = os.open(
                unrelated, os.O_RDONLY | os.O_DIRECTORY
            )
            assert reused == descriptor
            raise _DirectoryCloseCallFailed("after reuse")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(
                    reporting.os, "close", close_reuse_then_fail
                )
                offsets = _executed_opcode_offsets(
                    code, lease.owner.close, _DirectoryCloseCallFailed
                )
            return offsets
        finally:
            if reused is not None and _fd_identity(reused) is not None:
                real_close(reused)
            try:
                lease.__exit__(None, None, None)
            except OSError:
                pass

    scenario_offsets = {
        "before_close": observe_before_close(),
        "after_reuse": observe_after_reuse(),
    }
    assert all(scenario_offsets.values())

    baseline = _live_file_descriptors()
    for scenario, offsets in scenario_offsets.items():
        for target in offsets:
            lease = _open_test_lease(owned)
            owner = lease.owner
            descriptor = lease.fd
            identity = _fd_identity(descriptor)
            reused = None

            def fail_before_close(_descriptor):
                raise _DirectoryCloseCallFailed("before close")

            def close_reuse_then_fail(closing):
                nonlocal reused
                real_close(closing)
                reused = os.open(
                    unrelated, os.O_RDONLY | os.O_DIRECTORY
                )
                assert reused == closing
                raise _DirectoryCloseCallFailed("after reuse")

            failing_close = (
                fail_before_close
                if scenario == "before_close"
                else close_reuse_then_fail
            )

            def action():
                with monkeypatch.context() as patch:
                    patch.setattr(reporting.os, "close", failing_close)
                    owner.close()

            try:
                _interrupt_at_opcode(code, target, action)
                if _fd_identity(descriptor) == identity:
                    assert owner.fd == descriptor, (scenario, target)
                else:
                    with pytest.raises(RuntimeError, match="closed"):
                        owner.fd
            finally:
                if reused is not None and _fd_identity(reused) is not None:
                    real_close(reused)
                try:
                    lease.__exit__(None, None, None)
                except OSError:
                    pass
                if _fd_identity(descriptor) == identity:
                    real_close(descriptor)
            assert _live_file_descriptors() == baseline, (
                scenario,
                target,
            )
    _assert_lease_can_open_in_another_thread(owned)


@pytest.mark.parametrize("same_directory", [False, True])
def test_close_that_reuses_the_fd_then_raises_never_closes_the_reuser(
    tmp_path, monkeypatch, same_directory
):
    owned = tmp_path / "owned"
    unrelated = tmp_path / "unrelated"
    owned.mkdir()
    unrelated.mkdir()
    reuse_directory = owned if same_directory else unrelated
    lease = _open_test_lease(owned)
    owner = lease.owner
    descriptor = lease.fd
    probe = os.open(reuse_directory, os.O_RDONLY | os.O_DIRECTORY)
    reuse_identity = _fd_identity(probe)
    os.close(probe)
    real_close = os.close
    reused = None

    def close_reuse_then_interrupt(closing):
        nonlocal reused
        real_close(closing)
        reused = os.open(
            reuse_directory, os.O_RDONLY | os.O_DIRECTORY
        )
        assert reused == closing
        raise _DirectoryCloseInterrupted("closed and reused")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(reporting.os, "close", close_reuse_then_interrupt)
            with pytest.raises(
                _DirectoryCloseInterrupted, match="closed and reused"
            ):
                lease.__exit__(None, None, None)

        assert reused == descriptor
        assert _fd_identity(reused) == reuse_identity
        with pytest.raises(RuntimeError, match="closed"):
            owner.fd
        lease.__exit__(None, None, None)
        assert _fd_identity(reused) == reuse_identity
    finally:
        if reused is not None and _fd_identity(reused) is not None:
            real_close(reused)
        elif _fd_identity(descriptor) is not None:
            real_close(descriptor)


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
    assert "strongest_corruption_above_clean_rate" in metric_frame.columns
    assert "strongest_blur_above_clean_rate" not in metric_frame.columns
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
    for series in summary["evaluation"]["series"].values():
        assert "strongest_corruption_above_clean_rate" in series
        assert "strongest_blur_above_clean_rate" not in series


def test_report_explains_calculations_limits_and_uses_concrete_run_values(tmp_path):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)

    text = (output / "report.md").read_text(encoding="utf-8")
    lower = text.lower()
    example = sorted(rows, key=lambda row: (row["image_id"], row["severity"]))[0]
    primary = evaluation["series"]["persistence_relative_gap"]
    confidence = evaluation["series"]["confidence_relative_gap"]
    direct_mean = evaluation["series"]["direct_confidence_mean"]
    direct_max = evaluation["series"]["direct_confidence_max"]
    for concrete in (
        f"{example['persistence_reference']:.6f}",
        f"{example['persistence_responsive']:.6f}",
        f"{example['persistence_relative_gap']:.6f}",
        f"{example['confidence_reference']:.6f}",
        f"{example['confidence_responsive']:.6f}",
        f"{example['confidence_relative_gap']:.6f}",
        f"{example['direct_confidence_mean']:.6f}",
        f"{example['direct_confidence_max']:.6f}",
        f"{primary['median_signed_spearman']:+.3f}",
        f"{primary['oriented_adjacent_consistency']:.1%}",
    ):
        assert concrete in text
    for severity in range(1, 6):
        assert (
            f"| {severity} | {primary['auroc_by_severity'][severity]:.3f} | "
            f"{confidence['auroc_by_severity'][severity]:.3f} | "
            f"{direct_mean['auroc_by_severity'][severity]:.3f} | "
            f"{direct_max['auroc_by_severity'][severity]:.3f} |"
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
        "direct global confidence",
        "all valid queries",
        "lower confidence ranks as more corrupted",
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

    def stop(frame, evaluated, directory, config):
        assert directory.parent.resolve().parent == output.parent
        assert directory.parent.name.startswith(".report.staging-")
        assert config == provenance["config"]
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
def test_csv_unsafe_image_ids_are_rejected_without_publication(
    tmp_path, unsafe_id
):
    rows, _, provenance = _inputs()
    for row in rows:
        if row["image_id"] == "a":
            row["image_id"] = unsafe_id
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=5, query_count=20, persistence_dim=7,
        bootstrap_samples=100,
    )
    evaluation = evaluate_rows(rows, config)
    with pytest.raises(ValueError, match="image_id.*CSV safety"):
        write_report(tmp_path / "report", rows, evaluation, provenance)
    assert not (tmp_path / "report").exists()


def test_accepted_image_ids_round_trip_exactly_through_csv(tmp_path):
    rows, _, provenance = _inputs()
    accepted = "  safe,é_[sample]  "
    for row in rows:
        if row["image_id"] == "a":
            row["image_id"] = accepted
    config = ExperimentConfig.for_tests(
        bank_capacity=20, k=5, query_count=20, persistence_dim=7,
        bootstrap_samples=100,
    )
    evaluation = evaluate_rows(rows, config)

    write_report(tmp_path / "report", rows, evaluation, provenance)

    observed = pd.read_csv(
        tmp_path / "report" / "per-image-scores.csv",
        dtype={"image_id": str},
    )
    assert accepted in set(observed["image_id"])


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
            reporting._write_figures(
                frame, evaluation, tmp_path, provenance["config"]
            )
        assert set(plt.get_fignums()) == before
    finally:
        plt.close(caller)


def test_report_and_figure_labels_follow_the_validated_test_config(
    tmp_path, monkeypatch
):
    config = ExperimentConfig.for_tests(
        bank_capacity=20,
        k=3,
        query_count=20,
        persistence_layer=4,
        persistence_dim=7,
        reference_decile=1,
        responsive_decile=7,
        bootstrap_samples=100,
        blur_radii=(0.0, 0.5, 1.5, 3.0, 6.0, 9.0),
    )
    rows = _rows()
    evaluation = evaluate_rows(rows, config)
    default_provenance = _inputs()[2]
    provenance = {
        "checkpoint_sha256": "abc",
        "config": config.scientific_dict(),
        "runtime": copy.deepcopy(default_provenance["runtime"]),
        "corruption": {
            "name": "gaussian_blur",
            "severities": [
                {"level": level, "parameter": radius}
                for level, radius in enumerate(config.blur_radii)
            ],
        },
    }
    captured = {}
    original_save = reporting._save_close

    def capture_labels(fig, path):
        captured[path.name] = {
            "titles": [axis.get_title() for axis in fig.axes],
            "labels": [
                line.get_label()
                for axis in fig.axes
                for line in axis.get_lines()
            ],
        }
        original_save(fig, path)

    monkeypatch.setattr(reporting, "_save_close", capture_labels)
    output = tmp_path / "report"
    write_report(output, rows, evaluation, provenance)

    text = (output / "report.md").read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())
    assert "10-20% group, called the reference group" in normalized_text
    assert "70-80% group, called the responsive group" in normalized_text
    assert "decoder layer 4" in normalized_text
    assert "**3 nearest clean**" in normalized_text
    assert "toy five-neighbor example" in normalized_text
    assert "actual k is 3" in normalized_text
    assert "level 1 = 0.5" in normalized_text
    assert "add more corruption" not in normalized_text
    assert "earlier blur tuning" not in normalized_text
    labels = captured["reference-and-responsive-distance.png"]
    assert "Layer-4 distance" in labels["titles"]
    assert "10-20% reference" in labels["labels"]
    assert "70-80% responsive" in labels["labels"]
    ranking_labels = captured["auroc-by-corruption-severity.png"]["labels"]
    assert "Direct global confidence mean" in ranking_labels
    assert "Direct global confidence maximum" in ranking_labels


def test_gaussian_parameters_must_match_scientific_config(tmp_path):
    rows, evaluation, provenance = _inputs()
    provenance["corruption"]["severities"][3]["parameter"] = 999.0

    with pytest.raises(ValueError, match="Gaussian.*default_blur_radii"):
        write_report(tmp_path / "report", rows, evaluation, provenance)


def _write_minimal_complete_bundle(staging, *_args, **_kwargs):
    (staging / "figures").mkdir()
    for relative in REPORT_FILES:
        path = staging / relative
        path.write_bytes(relative.encode("ascii"))


def _publication_transfer_opcode_targets():
    targets = []
    publish_code = reporting._publish_no_replace.__code__
    publish_instructions = list(dis.get_instructions(publish_code))
    store_result = next(
        index
        for index, instruction in enumerate(publish_instructions)
        if instruction.opname == "STORE_FAST" and instruction.argval == "result"
    )
    publish_end = next(
        index
        for index, instruction in enumerate(
            publish_instructions[store_result:], store_result
        )
        if instruction.opname == "RETURN_VALUE"
    )
    for instruction in publish_instructions[store_result:publish_end + 1]:
        targets.append((
            publish_code,
            instruction.offset,
            f"publish-{instruction.offset}",
            True,
        ))

    write_code = reporting.write_report.__code__
    write_instructions = list(dis.get_instructions(write_code))
    start = next(
        index
        for index, instruction in enumerate(write_instructions)
        if instruction.opname == "LOAD_GLOBAL"
        and instruction.argval == "_publish_no_replace"
    )
    mark = next(
        index
        for index, instruction in enumerate(write_instructions[start:], start)
        if instruction.opname == "LOAD_METHOD"
        and instruction.argval == "mark_published"
    )
    end = next(
        index
        for index, instruction in enumerate(write_instructions[mark:], mark)
        if instruction.opname == "POP_TOP"
    )
    success_jump = next(
        index
        for index, instruction in enumerate(write_instructions[start:], start)
        if instruction.opname == "JUMP_FORWARD"
    )
    publish_call = next(
        instruction.offset
        for instruction in write_instructions[start:success_jump + 1]
        if instruction.opname == "CALL"
    )
    for instruction in (
        write_instructions[start:success_jump + 1]
        + write_instructions[mark:end + 1]
    ):
        targets.append((
            write_code,
            instruction.offset,
            f"write-{instruction.offset}",
            instruction.offset > publish_call,
        ))
    return targets


@pytest.mark.parametrize(
    ("target_code", "target_offset", "boundary", "must_survive"),
    _publication_transfer_opcode_targets(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_every_publication_transfer_interrupt_preserves_or_recovers_report(
    tmp_path, monkeypatch, target_code, target_offset, boundary, must_survive
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / boundary
    monkeypatch.setattr(
        reporting, "_write_bundle", _write_minimal_complete_bundle
    )
    raised = False

    def trace(frame, event, _arg):
        nonlocal raised
        if frame.f_code is target_code:
            frame.f_trace_opcodes = True
            if (
                not raised
                and event == "opcode"
                and frame.f_lasti == target_offset
            ):
                raised = True
                raise KeyboardInterrupt(boundary)
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(KeyboardInterrupt, match=boundary):
            write_report(output, rows, evaluation, provenance)
    finally:
        sys.settrace(previous)
    assert raised
    assert not list(tmp_path.glob(f".{boundary}.staging-*"))
    if must_survive:
        assert output.is_dir()
    write_report(output, rows, evaluation, provenance)
    assert sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    ) == sorted(REPORT_FILES)


def test_second_writer_success_survives_first_writer_post_rename_interrupt(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    published = threading.Event()
    second_done = threading.Event()
    failures = []
    original_publish = reporting._publish_no_replace
    monkeypatch.setattr(
        reporting, "_write_bundle", _write_minimal_complete_bundle
    )

    def interrupt_first_after_rename(*args):
        original_publish(*args)
        if threading.current_thread().name == "writer-a":
            published.set()
            assert second_done.wait(10)
            raise KeyboardInterrupt("after rename")

    def first_writer():
        try:
            write_report(output, rows, evaluation, provenance)
        except KeyboardInterrupt as error:
            if str(error) != "after rename":
                failures.append(error)
        except BaseException as error:
            failures.append(error)

    def second_writer():
        try:
            assert published.wait(10)
            write_report(output, rows, evaluation, provenance)
        except BaseException as error:
            failures.append(error)
        finally:
            second_done.set()

    monkeypatch.setattr(
        reporting, "_publish_no_replace", interrupt_first_after_rename
    )
    first = threading.Thread(target=first_writer, name="writer-a")
    second = threading.Thread(target=second_writer, name="writer-b")
    first.start()
    second.start()
    first.join(15)
    second.join(15)

    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert output.is_dir()
    write_report(output, rows, evaluation, provenance)
    assert output.is_dir()


def test_overlapping_plot_calls_never_close_each_others_figures(
    tmp_path, monkeypatch
):
    rows, evaluation, provenance = _inputs()
    frame = reporting._validated_frame(rows, config=provenance["config"])
    first_inside = threading.Event()
    second_attempted = threading.Event()
    second_inside = threading.Event()
    first_done = threading.Event()
    second_figure_alive = []
    failures = []

    def overlapping_impl(*_args, **_kwargs):
        figure = plt.figure()
        if threading.current_thread().name == "plot-a":
            first_inside.set()
            assert second_attempted.wait(5)
            second_inside.wait(0.3)
            raise RuntimeError("first plot fails")
        second_inside.set()
        assert first_done.wait(5)
        second_figure_alive.append(figure.number in plt.get_fignums())

    def run_first():
        try:
            reporting._write_figures(
                frame, evaluation, tmp_path / "a", provenance["config"]
            )
        except RuntimeError as error:
            if str(error) != "first plot fails":
                failures.append(error)
        except BaseException as error:
            failures.append(error)
        finally:
            first_done.set()

    def run_second():
        second_attempted.set()
        try:
            reporting._write_figures(
                frame, evaluation, tmp_path / "b", provenance["config"]
            )
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(reporting, "_write_figures_impl", overlapping_impl)
    first = threading.Thread(target=run_first, name="plot-a")
    second = threading.Thread(target=run_second, name="plot-b")
    first.start()
    assert first_inside.wait(5)
    second.start()
    first.join(10)
    second.join(10)

    assert not first.is_alive() and not second.is_alive()
    assert not failures
    assert second_figure_alive == [True]


def test_report_can_capture_the_exact_intended_bundle_without_changing_default_api(
    tmp_path,
):
    rows, evaluation, provenance = _inputs()
    output = tmp_path / "report"
    intended = {}

    result = write_report(
        output,
        rows,
        evaluation,
        provenance,
        _expected_content=intended,
    )

    assert result is None
    assert intended == {
        relative: (output / relative).read_bytes() for relative in REPORT_FILES
    }
