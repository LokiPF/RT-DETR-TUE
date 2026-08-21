import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.scene_uncertainty import cli, pipeline
from src.scene_uncertainty.artifacts import ShardWriter, iter_records, load_manifest
from src.scene_uncertainty.cli import build_parser, main
from src.scene_uncertainty.pipeline import PipelineError
from src.scene_uncertainty.runtime import checkpoint_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Small stand-ins for the real (300, 335) persistence tensors over three decoder layers.
QUERY_COUNT = 4
PERSISTENCE_DIM = 3
LAYERS = (0, 1)
SEVERITIES = tuple(range(6))


def test_cli_exposes_complete_artifact_pipeline():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    assert set(subcommands) == {
        "select",
        "extract-reference",
        "extract-blur",
        "build-bank",
        "evaluate-knn",
        "report",
        "analyze-confidence-deciles",
    }


# --------------------------------------------------------------------------------------
# parser: help and argument validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", [
    "select", "extract-reference", "extract-blur", "build-bank", "evaluate-knn", "report",
    "analyze-confidence-deciles",
])
def test_every_subcommand_documents_itself(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--output" in help_text
    assert len(help_text.splitlines()) > 5


def test_top_level_help_lists_every_subcommand(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    help_text = capsys.readouterr().out
    for command in ("select", "extract-reference", "extract-blur", "build-bank", "evaluate-knn",
                    "report", "analyze-confidence-deciles"):
        assert command in help_text


# The module docstring is the top-level epilog and the only ordered listing of the chain, so it
# is the one place where a subcommand can be added to the parser and stay invisible to a reader
# of `--help`. Spelled out rather than derived because the docstring spells it out.
SUBCOMMAND_COUNT_WORDS = {5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}


def test_the_pipeline_overview_counts_names_and_orders_every_subcommand():
    """`--help`'s overview must list what the parser accepts, how many, and in what order.

    `test_top_level_help_lists_every_subcommand` is satisfied by argparse's own generated
    listing of subparser names, so it passes whether or not the overview above it mentions the
    command at all -- and it says nothing about the count sentence, which is the part that goes
    stale silently.

    One equality rather than a membership loop, because the overview's own promise is "run in
    this order" and a loop cannot check it. The order is the whole content of the list: moving
    `analyze-confidence-deciles` to the top would tell an operator to analyse a feature cache
    and a results CSV that the commands below it have not produced yet. The equality also
    refuses a row for a command the parser does not have.
    """
    subcommands = list(build_parser()._subparsers._group_actions[0].choices)
    overview = cli.__doc__
    assert f"{SUBCOMMAND_COUNT_WORDS[len(subcommands)]} subcommands, run in this order:" in overview
    assert re.findall(r"^    (\S+) +-> ", overview, re.MULTILINE) == subcommands


def test_confidence_decile_command_has_only_cache_results_and_output():
    """Three arguments and nothing else -- in particular, no `--partition`.

    `load_decile_inputs` refuses any result manifest that was not built for the tuning
    partition, and the held-out test images are meant to stay unspendable from the command
    line. A `--partition` flag here would be the one way to spend them, so this compares the
    whole namespace rather than looking up the three arguments it expects to find.
    """
    args = build_parser().parse_args([
        "analyze-confidence-deciles",
        "--cache", "cache", "--results", "raw_k5.csv", "--output", "report",
    ])
    assert vars(args) == {
        "command": "analyze-confidence-deciles",
        "cache": "cache", "results": "raw_k5.csv", "output": "report",
    }


def test_the_confidence_decile_command_refuses_a_partition_flag(capsys):
    """What the namespace equality above cannot say, and the difference is not academic.

    `vars(args)` pins "no argument with a materialised default". An argument declared
    `default=argparse.SUPPRESS` never lands in the namespace at all, so the equality still
    holds while the parser happily accepts `--partition test` and binds it -- an inert flag
    today, and the seam a later edit widens. What actually keeps the held-out test images
    unspendable from the command line is that argparse rejects the flag outright, so that is
    what is asserted here, against the parser's behaviour rather than against the shape of its
    result.

    This is the outermost of several layers, not the only one: `_load_result_manifest` refuses
    any manifest not built for the tuning partition and `_load_distance_index` re-checks it per
    row. Both are exercised through `main` in `test_decile_integration.py`.
    """
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([
            "analyze-confidence-deciles",
            "--cache", "c", "--results", "r", "--output", "o",
            "--partition", "test",
        ])
    assert exit_info.value.code == 2
    assert "unrecognized arguments: --partition test" in capsys.readouterr().err


def test_missing_required_argument_is_refused_at_parse_time(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["build-bank", "--cache", "cache"])
    assert exit_info.value.code == 2
    assert "--output" in capsys.readouterr().err


def test_unknown_policy_is_named_at_parse_time(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([
            "evaluate-knn", "--cache", "c", "--bank", "b", "--output", "o",
            "--normalization", "raw", "--policies", "all,top99",
        ])
    assert exit_info.value.code == 2
    assert "top99" in capsys.readouterr().err


def test_unknown_aggregation_is_named_at_parse_time(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "evaluate-knn", "--cache", "c", "--bank", "b", "--output", "o",
            "--normalization", "raw", "--aggregations", "mean,geometric_mean",
        ])
    assert "geometric_mean" in capsys.readouterr().err


def test_default_policies_and_aggregations_are_validated_too():
    args = build_parser().parse_args([
        "evaluate-knn", "--cache", "c", "--bank", "b", "--output", "o", "--normalization", "raw",
    ])
    assert set(args.policies) <= set(pipeline.POLICIES)
    assert set(args.aggregations) == set(pipeline.AGGREGATIONS)


@pytest.mark.parametrize("argument,value", [("--k", "0"), ("--bank-chunk-size", "-1")])
def test_non_positive_evaluate_sizes_are_refused(argument, value, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([
            "evaluate-knn", "--cache", "c", "--bank", "b", "--output", "o",
            "--normalization", "raw", argument, value,
        ])
    assert exit_info.value.code == 2
    assert argument in capsys.readouterr().err


@pytest.mark.parametrize("argument,value", [
    ("--batch-size", "0"), ("--shard-size", "0"), ("--limit", "0"), ("--num-workers", "-1"),
])
def test_non_positive_extraction_sizes_are_refused(argument, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "extract-blur", "--config", "c", "--checkpoint", "k", "--images", "i",
            "--annotations", "a", "--selection", "s", "--output", "o", argument, value,
        ])


@pytest.mark.parametrize("command,arguments", [
    ("select", ["--train-ann", "t", "--val-ann", "v", "--output", "o"]),
    ("build-bank", ["--cache", "c", "--output", "o", "--population", "natural", "--capacity", "8"]),
])
def test_a_negative_seed_is_refused(command, arguments):
    """numpy's default_rng rejects it, several layers down, after the annotations are loaded."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, *arguments, "--seed", "-1"])


def test_zero_workers_is_accepted():
    args = build_parser().parse_args([
        "extract-blur", "--config", "c", "--checkpoint", "k", "--images", "i",
        "--annotations", "a", "--selection", "s", "--output", "o", "--num-workers", "0",
    ])
    assert args.num_workers == 0


def test_tools_entry_point_runs_the_cli():
    result = subprocess.run(
        [sys.executable, "tools/scene_uncertainty.py", "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert "evaluate-knn" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_peak_memory_instrumentation_survives_being_the_first_cuda_call():
    """`reset_peak_memory_stats` raises on a process that has not initialised CUDA yet.

    Run in a subprocess, because that is the only place the claim can be tested: once anything
    else in the session has touched CUDA the context already exists and the unfixed call would
    pass too, which would make this test quietly vacuous inside the full suite.
    """
    program = (
        "import torch;"
        "from src.scene_uncertainty import pipeline;"
        "pipeline._reset_peak_memory(torch.device('cuda:0'));"
        "assert pipeline._peak_memory_bytes(torch.device('cuda:0')) >= 0;"
        "assert pipeline._peak_memory_bytes(torch.device('cpu')) == 0;"
        "print('cuda_ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert "cuda_ok" in result.stdout


def test_an_unavailable_git_commit_is_announced_and_names_the_override(monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise OSError("no git here")

    monkeypatch.delenv("SCENE_UNCERTAINTY_GIT_COMMIT", raising=False)
    monkeypatch.setattr(pipeline.subprocess, "check_output", unavailable)
    assert pipeline._git_commit() == "unknown"
    warning = capsys.readouterr().err
    assert "WARNING" in warning
    assert "git_commit=unknown" in warning
    assert "SCENE_UNCERTAINTY_GIT_COMMIT" in warning
    assert "resum" in warning


def test_an_overridden_git_commit_is_recorded_without_a_warning(monkeypatch, capsys):
    monkeypatch.setenv("SCENE_UNCERTAINTY_GIT_COMMIT", "  abc123\n")
    assert pipeline._git_commit() == "abc123"
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------------------


def _write_coco(path: Path, image_ids, category_ids=(1, 2, 3)) -> Path:
    annotations = [
        {
            "id": 1000 + index,
            "image_id": int(image_id),
            "category_id": int(category_ids[index % len(category_ids)]),
            "bbox": [0, 0, 4, 4],
            "area": 16,
            "iscrowd": 0,
        }
        for index, image_id in enumerate(image_ids)
    ]
    path.write_text(json.dumps({
        "images": [{"id": int(image_id), "file_name": f"{image_id:012d}.jpg", "width": 8, "height": 8}
                   for image_id in image_ids],
        "annotations": annotations,
        "categories": [{"id": int(category_id), "name": str(category_id)} for category_id in category_ids],
    }), encoding="utf-8")
    return path


def _select_args(tmp_path: Path, **overrides):
    train = _write_coco(tmp_path / "train.json", range(1, 13))
    val = _write_coco(tmp_path / "val.json", range(100, 110))
    command = [
        "select", "--train-ann", str(train), "--val-ann", str(val),
        "--output", str(tmp_path / "selection"),
        "--natural-count", overrides.get("natural_count", "4"),
        "--augmentation-budget", overrides.get("augmentation_budget", "2"),
        "--quota", overrides.get("quota", "2"),
    ]
    return build_parser().parse_args(command)


def test_select_writes_reference_and_evaluation_ids(tmp_path: Path):
    pipeline.command_select(_select_args(tmp_path))
    reference = json.loads((tmp_path / "selection" / "reference.json").read_text())
    evaluation = json.loads((tmp_path / "selection" / "evaluation.json").read_text())
    assert len(reference["natural_ids"]) == 4
    assert len(reference["augmentation_ids"]) == 2
    assert not set(reference["natural_ids"]) & set(reference["augmentation_ids"])
    assert reference["seed"] == 42
    assert sorted(evaluation["tuning_ids"] + evaluation["test_ids"]) == list(range(100, 110))
    assert set(evaluation["tuning_pilot_ids"]) <= set(evaluation["tuning_ids"])


def test_select_records_the_cost_of_the_greedy_rare_class_pass(tmp_path: Path):
    pipeline.command_select(_select_args(tmp_path))
    reference = json.loads((tmp_path / "selection" / "reference.json").read_text())
    stats = reference["run_stats"]
    assert stats["wall_seconds"] >= 0.0
    assert stats["candidate_image_count"] == 12
    assert stats["augmentation_budget"] == 2


def test_select_refuses_a_request_larger_than_the_dataset(tmp_path: Path):
    args = _select_args(tmp_path, natural_count="20")
    with pytest.raises(PipelineError, match="20"):
        pipeline.command_select(args)
    assert not (tmp_path / "selection" / "reference.json").exists()


def test_select_refuses_a_missing_annotation_file(tmp_path: Path):
    args = _select_args(tmp_path)
    args.train_ann = str(tmp_path / "absent.json")
    with pytest.raises(PipelineError, match="absent.json"):
        pipeline.command_select(args)


# --------------------------------------------------------------------------------------
# selected image groups
# --------------------------------------------------------------------------------------


def test_reference_groups_label_natural_and_augmentation_images():
    selection = {"natural_ids": [1, 2], "augmentation_ids": [3]}
    assert pipeline._selected_image_groups(selection, True, None) == {
        1: "natural", 2: "natural", 3: "augmentation",
    }
    assert pipeline._selected_image_groups(selection, True, 2) == {1: "natural", 2: "natural"}


def test_evaluation_groups_split_a_limit_across_both_partitions():
    selection = {"tuning_pilot_ids": [1, 2, 3], "test_pilot_ids": [4, 5, 6]}
    assert pipeline._selected_image_groups(selection, False, None) == {
        1: "tuning", 2: "tuning", 3: "tuning", 4: "test", 5: "test", 6: "test",
    }
    limited = pipeline._selected_image_groups(selection, False, 3)
    assert limited == {1: "tuning", 2: "tuning", 4: "test"}


# --------------------------------------------------------------------------------------
# extraction, driven with a fake detector
# --------------------------------------------------------------------------------------


def _extract_args(tmp_path: Path, command: str, selection: dict, **overrides):
    for name in ("config.yml", "checkpoint.pth", "instances.json"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "images").mkdir(exist_ok=True)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"seed": 7, **selection}), encoding="utf-8")
    argv = [
        command,
        "--config", str(tmp_path / "config.yml"),
        "--checkpoint", str(tmp_path / "checkpoint.pth"),
        "--images", str(tmp_path / "images"),
        "--annotations", str(tmp_path / "instances.json"),
        "--selection", str(selection_path),
        "--output", str(overrides.get("output", tmp_path / "cache")),
        "--device", "cpu",
        "--num-workers", "0",
        "--shard-size", str(overrides.get("shard_size", 2)),
    ]
    return build_parser().parse_args(argv)


def _install_fake_detector(monkeypatch, fail_after=None, layer_shape=None):
    """Replace the detector, the loader and the extractor with recording stand-ins."""
    state = {"extractor_constructions": 0, "loader_calls": [], "extracted": 0}
    shape = layer_shape or (QUERY_COUNT, PERSISTENCE_DIM)

    class FakeExtractor:
        def __init__(self, model, matcher, decoder_layers):
            state["extractor_constructions"] += 1
            state["decoder_layers"] = list(decoder_layers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            state["closed"] = True
            return False

        def extract(self, samples, targets):
            records = []
            for target in targets:
                state["extracted"] += 1
                if fail_after is not None and state["extracted"] > fail_after:
                    raise RuntimeError("simulated interruption")
                image_id = int(target["image_id"].item())
                records.append({
                    "image_id": image_id,
                    "layers": {
                        layer_id: torch.full(shape, float(image_id + layer_id))
                        for layer_id in LAYERS
                    },
                    "logits": torch.zeros(QUERY_COUNT, 80),
                    "matched_gt_class": torch.tensor([0, 1, -1, -1]),
                })
            return records

    def fake_loader(image_root, annotation_file, image_ids, blur_radius, batch_size, num_workers):
        state["loader_calls"].append((float(blur_radius), list(image_ids)))
        batches = []
        for start in range(0, len(image_ids), batch_size):
            chunk = list(image_ids)[start:start + batch_size]
            batches.append((
                torch.zeros(len(chunk), 3, 4, 4),
                [{"image_id": torch.tensor(image_id), "labels": torch.zeros(0, dtype=torch.int64)}
                 for image_id in chunk],
            ))
        return batches

    monkeypatch.setattr(pipeline, "load_frozen_detector", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline, "_matcher", lambda: object())
    monkeypatch.setattr(pipeline, "ClassificationPersistenceExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline, "make_coco_loader", fake_loader)
    monkeypatch.setattr(pipeline, "QUERY_COUNT", QUERY_COUNT)
    monkeypatch.setattr(pipeline, "PERSISTENCE_DIM", PERSISTENCE_DIM)
    monkeypatch.setattr(pipeline, "DECODER_LAYERS", LAYERS)
    return state


def test_blur_extraction_sweeps_every_severity_with_one_extractor(tmp_path: Path, monkeypatch):
    state = _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": [2]})
    pipeline.command_extract_blur(args)

    # One hook registration for the whole run: the extractor registers its forward hooks in
    # __init__ with no double-registration guard, so one instance per batch would leave three
    # stale hooks per batch attached to the same decoder layers.
    assert state["extractor_constructions"] == 1
    assert state["closed"] is True
    assert [radius for radius, _ in state["loader_calls"]] == [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]
    records = list(iter_records(args.output))
    assert len(records) == 12
    assert sorted((record["image_id"], record["severity"]) for record in records) == sorted(
        (image_id, severity) for image_id in (1, 2) for severity in SEVERITIES
    )
    by_key = {(record["image_id"], record["severity"]): record for record in records}
    assert by_key[(1, 5)]["blur_radius"] == 12.0
    assert by_key[(1, 5)]["corruption_type"] == "gaussian_blur"
    assert by_key[(1, 0)]["source_partition"] == "tuning"
    assert by_key[(2, 0)]["source_partition"] == "test"
    assert by_key[(2, 0)]["reference_group"] is None


def test_reference_extraction_records_the_clean_condition_only(tmp_path: Path, monkeypatch):
    state = _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-reference", {"natural_ids": [1, 2], "augmentation_ids": [3]})
    pipeline.command_extract_reference(args)
    assert [radius for radius, _ in state["loader_calls"]] == [0.0]
    records = {record["image_id"]: record for record in iter_records(args.output)}
    assert set(records) == {1, 2, 3}
    assert all(record["severity"] == 0 for record in records.values())
    assert records[1]["reference_group"] == "natural"
    assert records[3]["reference_group"] == "augmentation"
    assert records[3]["source_partition"] == "reference"


def test_extraction_metadata_pins_everything_needed_to_reproduce_it(tmp_path: Path, monkeypatch):
    _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": [2]})
    pipeline.command_extract_blur(args)
    manifest = load_manifest(args.output)
    assert manifest["checkpoint_sha256"] == checkpoint_sha256(args.checkpoint)
    assert manifest["model_config_sha256"] == checkpoint_sha256(args.config)
    assert manifest["annotation_sha256"] == checkpoint_sha256(args.annotations)
    assert manifest["model_config"] == str(Path(args.config).resolve())
    assert manifest["image_root"] == str(Path(args.images).resolve())
    assert manifest["image_ids"] == [1, 2]
    assert manifest["split_seed"] == 7
    assert manifest["source_kind"] == "evaluation"
    assert manifest["corruption"] == {"type": "gaussian_blur", "radii": {"0": 0.0, "1": 1.0, "2": 2.0, "3": 4.0, "4": 8.0, "5": 12.0}}
    assert manifest["decoder_layers"] == list(LAYERS)
    assert manifest["classification_layers"] == ["decoder.dec_score_head.0", "decoder.dec_score_head.1"]
    assert manifest["persistence_dim"] == PERSISTENCE_DIM
    assert manifest["query_count"] == QUERY_COUNT
    assert isinstance(manifest["git_commit"], str)
    assert "run_stats" not in manifest
    assert len(manifest["artifact_id"]) == 64


def test_an_identical_re_extraction_has_an_identical_artifact_id(tmp_path: Path, monkeypatch):
    """The one property a content address exists for: same inputs, same id.

    Wall time and peak memory are per-run facts, so folding them into the hashed manifest
    would break this -- and `source_cache_id` / `feature_cache_id` carry that id downstream.
    """
    selection = {"tuning_pilot_ids": [1], "test_pilot_ids": [2]}
    ids = []
    for name in ("first", "second"):
        _install_fake_detector(monkeypatch)
        args = _extract_args(tmp_path, "extract-blur", selection, output=tmp_path / name)
        pipeline.command_extract_blur(args)
        ids.append(load_manifest(args.output)["artifact_id"])
    assert ids[0] == ids[1]


def test_extraction_timings_are_written_beside_the_manifest(tmp_path: Path, monkeypatch):
    _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": [2]})
    pipeline.command_extract_blur(args)
    stats = json.loads((Path(args.output) / "run_stats.json").read_text())
    assert stats["wall_seconds"] >= 0.0
    assert stats["peak_cuda_memory_bytes"] == 0
    assert stats["record_count"] == 12
    assert stats["records_written_this_segment"] == 12
    assert stats["records_recovered_on_resume"] == 0
    assert stats["covers_whole_extraction"] is True


def test_extraction_rejects_an_unexpected_persistence_shape(tmp_path: Path, monkeypatch):
    _install_fake_detector(monkeypatch, layer_shape=(QUERY_COUNT, PERSISTENCE_DIM + 1))
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": []})
    with pytest.raises(PipelineError, match=r"layer 0"):
        pipeline.command_extract_blur(args)


def test_interrupted_extraction_resumes_without_redoing_completed_shards(tmp_path: Path, monkeypatch, capsys):
    interrupted = _install_fake_detector(monkeypatch, fail_after=5)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": [2]})
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.command_extract_blur(args)
    assert interrupted["extractor_constructions"] == 1
    assert not (Path(args.output) / "manifest.json").exists()

    resumed = _install_fake_detector(monkeypatch)
    pipeline.command_extract_blur(args)
    assert resumed["extractor_constructions"] == 1
    # Four records were flushed into two shards before the interruption; the fifth was still
    # buffered and is redone. Nothing that reached a shard is extracted twice.
    assert resumed["extracted"] == 8
    requested = {radius: ids for radius, ids in resumed["loader_calls"]}
    assert set(requested) == {2.0, 4.0, 8.0, 12.0}
    assert requested[2.0] == [1, 2]
    records = list(iter_records(args.output))
    assert len(records) == 12
    assert len({(record["image_id"], record["severity"]) for record in records}) == 12
    assert load_manifest(args.output)["record_count"] == 12
    stats = json.loads((Path(args.output) / "run_stats.json").read_text())
    assert stats["records_recovered_on_resume"] == 4
    assert stats["records_written_this_segment"] == 8
    assert stats["covers_whole_extraction"] is False
    assert "resume" in capsys.readouterr().err.lower()


def test_completed_extraction_is_not_repeated(tmp_path: Path, monkeypatch):
    _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": []})
    pipeline.command_extract_blur(args)
    first = load_manifest(args.output)

    repeated = _install_fake_detector(monkeypatch)
    pipeline.command_extract_blur(args)
    assert repeated["extracted"] == 0
    assert load_manifest(args.output) == first


def test_completed_extraction_refuses_an_incompatible_rerun(tmp_path: Path, monkeypatch):
    _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": []})
    pipeline.command_extract_blur(args)
    Path(args.checkpoint).write_text("a different detector", encoding="utf-8")
    _install_fake_detector(monkeypatch)
    with pytest.raises(PipelineError, match="checkpoint_sha256"):
        pipeline.command_extract_blur(args)


def test_extraction_refuses_an_unusable_device(tmp_path: Path, monkeypatch):
    _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": []})
    args.device = "not-a-device"
    with pytest.raises(PipelineError, match="not-a-device"):
        pipeline.command_extract_blur(args)


def test_extraction_refuses_a_missing_checkpoint(tmp_path: Path, monkeypatch):
    _install_fake_detector(monkeypatch)
    args = _extract_args(tmp_path, "extract-blur", {"tuning_pilot_ids": [1], "test_pilot_ids": []})
    args.checkpoint = str(tmp_path / "absent.pth")
    with pytest.raises(PipelineError, match="absent.pth"):
        pipeline.command_extract_blur(args)


# --------------------------------------------------------------------------------------
# cache fixtures for bank building and scoring
# --------------------------------------------------------------------------------------


def _vectors(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(QUERY_COUNT, PERSISTENCE_DIM, generator=generator)


def _cache_metadata(image_ids, source_kind: str) -> dict:
    return {
        "git_commit": "commit",
        "checkpoint_sha256": "checkpoint",
        "model_config": "config.yml",
        "image_ids": [int(image_id) for image_id in image_ids],
        "source_kind": source_kind,
        "decoder_layers": list(LAYERS),
        "persistence_dim": PERSISTENCE_DIM,
        "query_count": QUERY_COUNT,
    }


def _write_reference_cache(directory: Path) -> Path:
    """Four natural and two augmentation images, clean only. Written once per directory."""
    if (directory / "manifest.json").exists():
        return directory
    groups = {1: "natural", 2: "natural", 3: "natural", 4: "natural", 5: "augmentation", 6: "augmentation"}
    with ShardWriter(directory, _cache_metadata(groups, "reference"), shard_size=4) as writer:
        for image_id, group in groups.items():
            offset = 0.0 if group == "natural" else 100.0
            writer.add({
                "image_id": image_id,
                "severity": 0,
                "source_partition": "reference",
                "reference_group": group,
                "layers": {layer_id: _vectors(image_id * 10 + layer_id) + offset for layer_id in LAYERS},
                "logits": torch.zeros(QUERY_COUNT, 80),
                "matched_gt_class": torch.tensor([0, 1, -1, -1]),
                "matched_annotation_id": torch.tensor([10, 11, -1, -1]),
                "predicted_class": torch.tensor([0, 1, 0, 0]),
                "is_matched": torch.tensor([True, True, False, False]),
                "is_correct": torch.tensor([True, False, False, False]),
            })
    return directory


def _evaluation_record(image_id: int, severity: int, partition: str) -> dict:
    return {
        "image_id": image_id,
        "severity": severity,
        "blur_radius": pipeline.BLUR_RADII[severity],
        "corruption_type": "gaussian_blur",
        "source_partition": partition,
        "reference_group": None,
        # Distance from the bank grows with severity, so the trend statistics are defined.
        "layers": {
            layer_id: _vectors(image_id * 10 + layer_id) + severity
            for layer_id in LAYERS
        },
        "logits": torch.full((QUERY_COUNT, 80), -1.0),
        "matched_gt_class": torch.tensor([0, 1, -1, -1]),
        "matched_annotation_id": torch.tensor([10, 11, -1, -1]),
        "predicted_class": torch.tensor([0, 1, 0, 0]),
        "is_matched": torch.tensor([True, True, False, False]),
        "is_correct": torch.tensor([True, False, False, False]),
    }


def _write_evaluation_cache(directory: Path, records=None, image_ids=(11, 12)) -> Path:
    partitions = {11: "tuning", 12: "test"}
    if records is None:
        records = [
            _evaluation_record(image_id, severity, partitions[image_id])
            for image_id in image_ids
            for severity in SEVERITIES
        ]
    with ShardWriter(directory, _cache_metadata(image_ids, "evaluation"), shard_size=5) as writer:
        for record in records:
            writer.add(record)
    return directory


def _bank_args(tmp_path: Path, cache: Path, population="natural", capacity="8", output=None):
    return build_parser().parse_args([
        "build-bank", "--cache", str(cache), "--output", str(output or tmp_path / "bank"),
        "--population", population, "--capacity", capacity, "--seed", "3",
    ])


def _built_bank(tmp_path: Path, population="natural") -> Path:
    cache = _write_reference_cache(tmp_path / "reference")
    args = _bank_args(tmp_path, cache, population=population)
    pipeline.command_build_bank(args)
    return Path(args.output)


# --------------------------------------------------------------------------------------
# build-bank
# --------------------------------------------------------------------------------------


def test_natural_bank_uses_only_the_uniformly_drawn_reference_images(tmp_path: Path):
    bank_root = _built_bank(tmp_path)
    manifest = json.loads((bank_root / "manifest.json").read_text())
    assert manifest["artifact_type"] == "class_independent_query_bank"
    assert manifest["class_independent"] is True
    assert manifest["population"] == "natural"
    assert manifest["capacity"] == 8
    assert manifest["seed"] == 3
    assert manifest["source_image_ids"] == [1, 2, 3, 4, 5, 6]
    assert manifest["checkpoint_sha256"] == "checkpoint"
    assert sorted(manifest["layers"]) == ["0", "1"]
    assert len(manifest["artifact_id"]) == 64
    for layer_id, layer in manifest["layers"].items():
        vectors = torch.load(bank_root / layer["path"], map_location="cpu", weights_only=False)["vectors"]
        assert vectors.shape == (8, PERSISTENCE_DIM)
        assert layer["vector_count"] == 8
        assert layer["sampling"]["natural_vectors"] == 8
        # The augmentation images were written 100 away from every natural vector.
        assert vectors.max() < 100.0


def test_coverage_bank_reports_its_object_and_background_split(tmp_path: Path):
    bank_root = _built_bank(tmp_path, population="coverage")
    manifest = json.loads((bank_root / "manifest.json").read_text())
    assert manifest["population"] == "coverage"
    for layer in manifest["layers"].values():
        assert layer["sampling"]["object_vectors"] >= 1
        assert layer["sampling"]["background_vectors"] >= 1
        assert layer["vector_count"] == 8


def test_bank_records_the_cache_it_was_built_from(tmp_path: Path):
    cache = _write_reference_cache(tmp_path / "reference")
    bank_root = _built_bank(tmp_path)
    manifest = json.loads((bank_root / "manifest.json").read_text())
    assert manifest["source_cache_id"] == load_manifest(cache)["artifact_id"]
    assert manifest["run_stats"]["bank_bytes_excluding_manifest"] > 0


def test_bank_refuses_an_evaluation_cache(tmp_path: Path):
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    with pytest.raises(PipelineError, match="reference"):
        pipeline.command_build_bank(_bank_args(tmp_path, cache))


def test_bank_refuses_to_overwrite_a_finished_bank(tmp_path: Path):
    cache = _write_reference_cache(tmp_path / "reference")
    args = _bank_args(tmp_path, cache)
    pipeline.command_build_bank(args)
    with pytest.raises(FileExistsError, match="already complete"):
        pipeline.command_build_bank(args)


def test_bank_refuses_a_missing_cache(tmp_path: Path):
    args = _bank_args(tmp_path, tmp_path / "absent")
    with pytest.raises(PipelineError, match="absent"):
        pipeline.command_build_bank(args)


# --------------------------------------------------------------------------------------
# evaluate-knn
# --------------------------------------------------------------------------------------


def _evaluate_args(tmp_path: Path, cache: Path, bank: Path, **overrides):
    argv = [
        "evaluate-knn", "--cache", str(cache), "--bank", str(bank),
        "--output", str(overrides.get("output", tmp_path / "results.csv")),
        "--normalization", overrides.get("normalization", "raw"),
        "--partition", overrides.get("partition", "tuning"),
        "--policies", overrides.get("policies", "all,top10"),
        "--aggregations", overrides.get("aggregations", "mean"),
        "--k", "2", "--device", "cpu",
    ]
    if "bank_chunk_size" in overrides:
        argv += ["--bank-chunk-size", overrides["bank_chunk_size"]]
    return build_parser().parse_args(argv)


def _scored(tmp_path: Path, **overrides):
    bank = _built_bank(tmp_path)
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    args = _evaluate_args(tmp_path, cache, bank, **overrides)
    pipeline.command_evaluate_knn(args)
    return args


def test_evaluate_writes_rows_distances_normalizers_and_a_manifest(tmp_path: Path):
    args = _scored(tmp_path)
    output = Path(args.output)
    rows = pipeline.read_result_csv(output)
    assert {row["image_id"] for row in rows} == {11}
    assert {row["policy"] for row in rows} == {"all", "top10"}
    assert len(rows) == 12
    assert all(row["valid"] for row in rows)

    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["artifact_type"] == "knn_scene_uncertainty_results"
    assert manifest["normalization"] == "raw"
    assert manifest["source_partition"] == "tuning"
    assert manifest["k"] == 2
    assert manifest["row_count"] == 12
    assert manifest["policies"] == ["all", "top10"]
    assert manifest["aggregations"] == ["mean"]
    assert manifest["query_distance_record_count"] == 6
    assert manifest["bank_id"] == json.loads((Path(args.bank) / "manifest.json").read_text())["artifact_id"]
    assert manifest["feature_cache_id"] == load_manifest(args.cache)["artifact_id"]

    distances = torch.load(output.with_suffix(".query_distances.pt"), map_location="cpu", weights_only=False)
    assert len(distances) == 6
    assert distances[0]["query_scores_by_layer"][0].dtype == torch.float16
    normalizers = torch.load(output.with_suffix(".normalizers.pt"), map_location="cpu", weights_only=False)
    assert set(normalizers["feature_normalizers"]) == set(LAYERS)
    assert set(normalizers["layer_score_scales"]) == set(LAYERS)


def test_clean_distance_diagnostics_are_readable_without_unpickling(tmp_path: Path):
    args = _scored(tmp_path)
    manifest = json.loads(Path(args.output).with_suffix(".manifest.json").read_text())
    fit = manifest["clean_distance_fit"]
    assert sorted(fit) == ["0", "1"]
    for layer in fit.values():
        assert layer["scale"] > 0.0
        assert layer["center"] >= 0.0
        assert layer["sample_count"] == 8
        assert layer["min_neighbor_distance"] >= 0.0
        assert 0.0 <= layer["near_duplicate_fraction"] <= 1.0


def test_bank_chunk_size_is_recorded_but_stays_out_of_the_artifact_id(tmp_path: Path):
    bank = _built_bank(tmp_path)
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    manifests = []
    scores = []
    # The same output path twice, so the chunk width is the only thing that differs.
    for chunk_size in ("2", "8192"):
        args = _evaluate_args(tmp_path, cache, bank, bank_chunk_size=chunk_size)
        pipeline.command_evaluate_knn(args)
        manifests.append(json.loads(Path(args.output).with_suffix(".manifest.json").read_text()))
        scores.append([row["raw_score"] for row in pipeline.read_result_csv(args.output)])
    assert [manifest["run_stats"]["bank_chunk_size"] for manifest in manifests] == [2, 8192]
    assert manifests[0]["artifact_id"] == manifests[1]["artifact_id"]
    # Chunked kNN is exact but not bit-identical, so the two runs are interchangeable rather
    # than equal: the scores agree far more closely than any signal the study reads.
    assert scores[0] == pytest.approx(scores[1], rel=1e-5)


def test_content_address_ignores_performance_facts_but_nothing_else(tmp_path: Path):
    base = {"artifact_type": "results", "k": 5, "run_stats": {"wall_seconds": 1.0, "bank_chunk_size": 2}}
    faster = {**base, "run_stats": {"wall_seconds": 90.0, "bank_chunk_size": 8192}}
    assert pipeline._content_address(base) == pipeline._content_address(faster)
    assert pipeline._content_address(base) == pipeline._content_address({**base, "artifact_id": "x"})
    assert pipeline._content_address(base) == pipeline._content_address(
        {**base, "clean_distance_fit": {"0": {"scale": 1.0}}}
    )
    assert pipeline._content_address(base) != pipeline._content_address({**base, "k": 6})


def test_a_contaminated_clean_distance_fit_is_announced_when_it_is_fitted(capsys):
    clean = {"0": {"near_duplicate_fraction": 0.0, "scale": 1.0}}
    contaminated = {"1": {"near_duplicate_fraction": 0.2, "scale": 2.8}}
    pipeline._warn_about_contaminated_fits(clean)
    assert capsys.readouterr().err == ""
    pipeline._warn_about_contaminated_fits(contaminated)
    warning = capsys.readouterr().err
    assert "WARNING" in warning
    assert "layer 1" in warning
    assert "20.0%" in warning


def test_evaluate_scores_every_partition_when_asked(tmp_path: Path):
    args = _scored(tmp_path, partition="all", output=Path(tmp_path / "all.csv"))
    rows = pipeline.read_result_csv(args.output)
    assert {row["source_partition"] for row in rows} == {"tuning", "test"}


def test_evaluate_refuses_a_partition_that_selects_no_record(tmp_path: Path):
    bank = _built_bank(tmp_path)
    cache = _write_evaluation_cache(tmp_path / "evaluation", image_ids=(11,))
    args = _evaluate_args(tmp_path, cache, bank, partition="test")
    with pytest.raises(PipelineError, match="test"):
        pipeline.command_evaluate_knn(args)


def test_evaluate_refuses_reference_evaluation_leakage(tmp_path: Path):
    bank = _built_bank(tmp_path)
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_image_ids"] = [11]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    with pytest.raises(PipelineError, match="leakage"):
        pipeline.command_evaluate_knn(_evaluate_args(tmp_path, cache, bank))


def test_evaluate_refuses_an_incomplete_blur_sweep(tmp_path: Path):
    bank = _built_bank(tmp_path)
    records = [_evaluation_record(11, severity, "tuning") for severity in (0, 1, 2)]
    cache = _write_evaluation_cache(tmp_path / "evaluation", records=records, image_ids=(11,))
    with pytest.raises(PipelineError, match="Incomplete blur sweeps"):
        pipeline.command_evaluate_knn(_evaluate_args(tmp_path, cache, bank))


def test_evaluate_refuses_a_duplicated_record(tmp_path: Path):
    bank = _built_bank(tmp_path)
    records = [_evaluation_record(11, severity, "tuning") for severity in SEVERITIES]
    records.append(_evaluation_record(11, 3, "tuning"))
    cache = _write_evaluation_cache(tmp_path / "evaluation", records=records, image_ids=(11,))
    with pytest.raises(PipelineError, match="Duplicate"):
        pipeline.command_evaluate_knn(_evaluate_args(tmp_path, cache, bank))


def test_evaluate_refuses_a_bank_from_another_checkpoint(tmp_path: Path):
    bank = _built_bank(tmp_path)
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_sha256"] = "another-detector"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    with pytest.raises(PipelineError, match="checkpoint_sha256"):
        pipeline.command_evaluate_knn(_evaluate_args(tmp_path, cache, bank))


def test_evaluate_refuses_a_bank_missing_a_decoder_layer(tmp_path: Path):
    bank = _built_bank(tmp_path)
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layers"].pop("1")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    with pytest.raises(PipelineError, match="layer"):
        pipeline.command_evaluate_knn(_evaluate_args(tmp_path, cache, bank))


def test_evaluate_refuses_a_k_larger_than_the_bank(tmp_path: Path):
    bank = _built_bank(tmp_path)
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    args = _evaluate_args(tmp_path, cache, bank)
    args.k = 8
    with pytest.raises(PipelineError, match="--k 8"):
        pipeline.command_evaluate_knn(args)


_EXECUTED: list[str] = []


def _run_on_unpickle():
    """Stand-in for whatever a hostile bank file would run. Records that it ran."""
    _EXECUTED.append("executed")
    return torch.zeros(8, PERSISTENCE_DIM)


class _ArbitraryCodePayload:
    """Pickles as a call to `_run_on_unpickle`, the way any `__reduce__` payload would."""

    def __reduce__(self):
        return (_run_on_unpickle, ())


def test_evaluate_reads_the_bank_without_executing_it(tmp_path: Path):
    """A bank directory is a float tensor plus a str-keyed sampling summary.

    Banks are copied between hosts like every other artifact here, so loading one must
    not be able to run what the copy contains. Nothing in the file needs the unpickler's
    ability to call arbitrary code, so it is read with `weights_only=True`.
    """
    bank = _built_bank(tmp_path)
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    layer = next(iter(json.loads((bank / "manifest.json").read_text())["layers"].values()))
    torch.save({"vectors": _ArbitraryCodePayload(), "sampling": {}}, bank / layer["path"])
    _EXECUTED.clear()
    with pytest.raises(pickle.UnpicklingError):
        pipeline.command_evaluate_knn(_evaluate_args(tmp_path, cache, bank))
    assert _EXECUTED == []


def test_evaluate_refuses_a_missing_bank(tmp_path: Path):
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    with pytest.raises(PipelineError, match="absent"):
        pipeline.command_evaluate_knn(_evaluate_args(tmp_path, cache, tmp_path / "absent"))


def test_evaluate_validates_names_before_touching_the_bank(tmp_path: Path, monkeypatch):
    bank = _built_bank(tmp_path)
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    args = _evaluate_args(tmp_path, cache, bank)
    args.policies = ("all", "made_up_policy")

    def explode(*call_args, **call_kwargs):
        raise AssertionError("the bank was loaded before the policy names were checked")

    monkeypatch.setattr(pipeline, "fit_normalizer", explode)
    with pytest.raises(PipelineError, match="made_up_policy"):
        pipeline.command_evaluate_knn(args)


def test_evaluate_accepts_a_comma_separated_string_of_names(tmp_path: Path):
    bank = _built_bank(tmp_path)
    cache = _write_evaluation_cache(tmp_path / "evaluation")
    args = _evaluate_args(tmp_path, cache, bank)
    args.policies = "all"
    args.aggregations = "mean,median"
    pipeline.command_evaluate_knn(args)
    manifest = json.loads(Path(args.output).with_suffix(".manifest.json").read_text())
    assert manifest["policies"] == ["all"]
    assert manifest["aggregations"] == ["mean", "median"]


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def test_a_crash_after_the_csv_leaves_no_manifest_to_mislabel_it(tmp_path: Path, monkeypatch):
    first = _scored(tmp_path, policies="all", aggregations="mean")
    manifest_path = Path(first.output).with_suffix(".manifest.json")
    assert json.loads(manifest_path.read_text())["policies"] == ["all"]

    def explode(value, path):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline, "_atomic_torch_save", explode)
    second = _evaluate_args(
        tmp_path, first.cache, first.bank, policies="all,top10", aggregations="median",
    )
    with pytest.raises(RuntimeError, match="disk full"):
        pipeline.command_evaluate_knn(second)
    assert not manifest_path.exists()

    report = build_parser().parse_args([
        "report", "--results", str(first.output), "--output", str(tmp_path / "report"),
    ])
    with pytest.raises(PipelineError, match="No result manifest"):
        pipeline.command_report(report)


def test_report_summarizes_the_scored_rows_with_the_run_metadata(tmp_path: Path):
    scored = _scored(tmp_path)
    args = build_parser().parse_args([
        "report", "--results", str(scored.output), "--output", str(tmp_path / "report"),
    ])
    pipeline.command_report(args)
    output = Path(args.output)
    for name in ("per_scene.csv", "summary.json", "run_metadata.json", "raw_trend.png",
                 "relative_trend.png", "class_switch.png"):
        assert (output / name).exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["run_metadata"]["artifact_type"] == "knn_scene_uncertainty_results"
    assert summary["run_metadata"]["clean_distance_fit"]["0"]["scale"] > 0.0
    combined = [group for group in summary["groups"] if group["score_scope"] == "combined"]
    assert {group["policy"] for group in combined} == {"all", "top10"}
    assert all(group["median_spearman"] == 1.0 for group in combined)


def test_report_names_an_empty_result_set_instead_of_failing_inside_pandas(tmp_path: Path):
    results = tmp_path / "empty.csv"
    pipeline.write_result_csv([], results)
    results.with_suffix(".manifest.json").write_text(json.dumps({"source_partition": "test"}), encoding="utf-8")
    args = build_parser().parse_args([
        "report", "--results", str(results), "--output", str(tmp_path / "report"),
    ])
    with pytest.raises(PipelineError, match="empty.csv"):
        pipeline.command_report(args)


def test_report_refuses_a_concatenated_results_csv(tmp_path: Path):
    """`--results` accepts any CSV, and merging two result files is an obvious move.

    One `evaluate-knn` run writes one row per (image_id, severity, policy, aggregation,
    source_partition), and `_validate_evaluation_cache` refuses a duplicated cache
    record, so this is unreachable inside the sanctioned chain -- and entirely reachable
    from the command line, where it inflates the adjacent-step counts with no error.
    """
    scored = _scored(tmp_path)
    rows = pipeline.read_result_csv(scored.output)
    concatenated = tmp_path / "concatenated.csv"
    pipeline.write_result_csv(rows + rows, concatenated)
    concatenated.with_suffix(".manifest.json").write_bytes(
        Path(scored.output).with_suffix(".manifest.json").read_bytes()
    )
    args = build_parser().parse_args([
        "report", "--results", str(concatenated), "--output", str(tmp_path / "report"),
    ])
    with pytest.raises(PipelineError, match="image_id=11"):
        pipeline.command_report(args)
    assert not (tmp_path / "report").exists()


def test_report_refuses_results_without_their_manifest(tmp_path: Path):
    scored = _scored(tmp_path)
    Path(scored.output).with_suffix(".manifest.json").unlink()
    args = build_parser().parse_args([
        "report", "--results", str(scored.output), "--output", str(tmp_path / "report"),
    ])
    with pytest.raises(PipelineError, match="manifest"):
        pipeline.command_report(args)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def test_main_dispatches_and_reports_success(tmp_path: Path, monkeypatch):
    seen = {}
    monkeypatch.setitem(pipeline.COMMANDS, "report", lambda args: seen.setdefault("output", args.output))
    assert main(["report", "--results", "r.csv", "--output", str(tmp_path)]) == 0
    assert seen["output"] == str(tmp_path)


def test_main_reports_a_user_error_as_one_line(tmp_path: Path, capsys):
    results = tmp_path / "empty.csv"
    pipeline.write_result_csv([], results)
    results.with_suffix(".manifest.json").write_text("{}", encoding="utf-8")
    status = main(["report", "--results", str(results), "--output", str(tmp_path / "report")])
    captured = capsys.readouterr()
    assert status == 2
    assert captured.err.startswith("scene_uncertainty report: error:")
    assert len(captured.err.strip().splitlines()) == 1
    assert "Traceback" not in captured.err


def test_main_lets_an_unexpected_failure_surface(tmp_path: Path, monkeypatch):
    def explode(args):
        raise ZeroDivisionError("a real bug")

    monkeypatch.setitem(pipeline.COMMANDS, "report", explode)
    with pytest.raises(ZeroDivisionError):
        main(["report", "--results", "r.csv", "--output", str(tmp_path)])
