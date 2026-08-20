"""End-to-end checks that the shipped modules compose into the study's artifacts.

Nothing here is faked. The caches are written by the real `ShardWriter`, the bank by the
real `build-bank`, the scores by the real `evaluate-knn` and the report by the real
`report`; the CLI chain is driven through `main`, so the artifact hand-offs are exercised
exactly as the pilot exercises them. What is small is the *data* -- eight queries of four
features over three decoder layers, instead of three hundred queries of three hundred and
thirty-five -- so the whole chain runs on CPU in seconds with no checkpoint, no COCO and
no download. The one link this file cannot cover is the detector forward pass that fills a
cache; the pilot in the README covers that, and `tests/scene_uncertainty/test_extractor.py`
covers the extractor against a stub decoder.

Two of the assertions below look like bugs at first reading and are not:

* `smooth_1` is reported under `weighted_mean` and never under the aggregations that were
  asked for. A smoothed policy carries its signal in per-query weights, so the aggregation
  is forced and the row is relabelled with the one that actually ran.
* A threshold policy that stops selecting under blur publishes `median_spearman == 1.0`
  from the severities it survived. That is why `scored_severity_count` is asserted next to
  it: the counts are the only thing separating a truncated curve from a whole one.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

from src.scene_uncertainty.artifacts import ShardWriter, iter_records, load_manifest
from src.scene_uncertainty.cli import main
from src.scene_uncertainty.evaluate import score_cached_record
from src.scene_uncertainty.reporting import write_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

QUERY_COUNT = 8
PERSISTENCE_DIM = 4
LAYERS = (0, 1, 2)
SEVERITIES = tuple(range(6))
CLASS_COUNT = 80

# Two matched objects and six background queries, which is what a coverage bank needs.
MATCHED_GT_CLASS = torch.tensor([0, 1, -1, -1, -1, -1, -1, -1])
MATCHED_ANNOTATION_ID = torch.tensor([10, 11, -1, -1, -1, -1, -1, -1])
IS_MATCHED = MATCHED_GT_CLASS >= 0


def test_cached_knn_report_pipeline(tmp_path: Path):
    """A cache written, read back, scored against a bank, and summarized -- in one pass."""
    cache = tmp_path / "cache"
    metadata = {"checkpoint_sha256": "test", "decoder_layers": [2], "query_count": 3}
    with ShardWriter(cache, metadata, shard_size=2) as writer:
        for severity in range(6):
            writer.add({
                "image_id": 1,
                "severity": severity,
                "layers": {2: torch.tensor([[severity, 0.0], [severity, 1.0], [severity, 2.0]])},
                "logits": torch.tensor([[4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]),
                "matched_annotation_id": torch.tensor([10, -1, 11]),
                "predicted_class": torch.tensor([0, 0, 0]),
            })
    bank = {2: torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])}
    rows = [score_cached_record(record, bank, "all", "mean", k=1) for record in iter_records(cache)]
    report = tmp_path / "report"
    write_report(rows, report)
    assert (report / "summary.json").exists()
    assert rows[-1]["raw_score"] > rows[0]["raw_score"]

    # Each query sits `severity` away from the bank row it came from, so the scene score is
    # the severity itself: a monotone curve measured over all six severities, not over a
    # surviving subset of them.
    assert [row["raw_score"] for row in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    combined = next(group for group in summary["groups"] if group["score_scope"] == "combined")
    assert combined["median_spearman"] == 1.0
    assert combined["endpoint_increase_rate"] == 1.0
    assert combined["image_count"] == combined["scored_image_count"] == 1
    assert combined["scored_severity_count"] == combined["total_severity_count"] == 6
    assert summary["diagnostics"]["groups_with_unscored_severities"] == 0


# --------------------------------------------------------------------------------------
# the CLI chain: two feature caches -> bank -> scores -> report
# --------------------------------------------------------------------------------------


def _vectors(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(QUERY_COUNT, PERSISTENCE_DIM, generator=generator)


def _logits(confident: bool) -> torch.Tensor:
    """Logits whose maximum sigmoid sits either side of the 0.5 threshold."""
    return torch.full((QUERY_COUNT, CLASS_COUNT), 2.0 if confident else -2.0)


def _metadata(image_ids, source_kind: str) -> dict:
    return {
        "git_commit": "integration",
        "checkpoint_sha256": "checkpoint",
        "model_config": "config.yml",
        "image_ids": [int(image_id) for image_id in image_ids],
        "source_kind": source_kind,
        "decoder_layers": list(LAYERS),
        "persistence_dim": PERSISTENCE_DIM,
        "query_count": QUERY_COUNT,
    }


def _write_reference_cache(directory: Path) -> Path:
    image_ids = [1, 2, 3, 4, 5, 6]
    with ShardWriter(directory, _metadata(image_ids, "reference"), shard_size=4) as writer:
        for image_id in image_ids:
            writer.add({
                "image_id": image_id,
                "severity": 0,
                "source_partition": "reference",
                "reference_group": "natural" if image_id <= 4 else "augmentation",
                "layers": {layer_id: _vectors(image_id * 10 + layer_id) for layer_id in LAYERS},
                "logits": _logits(confident=True),
                "matched_gt_class": MATCHED_GT_CLASS,
                "matched_annotation_id": MATCHED_ANNOTATION_ID,
                "predicted_class": torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]),
                "is_matched": IS_MATCHED,
                "is_correct": torch.tensor([True, True, False, False, False, False, False, False]),
            })
    return directory


def _write_blur_cache(directory: Path) -> Path:
    """Two tuning and two test images swept over all six severities.

    Distance from the bank grows with severity, so the scene score has a trend to find, and
    confidence drops below 0.5 from severity three, so `threshold_0.5` stops selecting -- the
    same collapse a real threshold policy shows under blur. One annotation changes its
    predicted class at severity two, so the class-switch split has a step on both sides.
    """
    partitions = {11: "tuning", 12: "tuning", 13: "test", 14: "test"}
    with ShardWriter(directory, _metadata(list(partitions), "evaluation"), shard_size=5) as writer:
        for image_id, partition in partitions.items():
            for severity in SEVERITIES:
                writer.add({
                    "image_id": image_id,
                    "severity": severity,
                    "blur_radius": float(severity),
                    "corruption_type": "gaussian_blur",
                    "source_partition": partition,
                    "reference_group": None,
                    "layers": {
                        layer_id: _vectors(image_id * 10 + layer_id) + severity
                        for layer_id in LAYERS
                    },
                    "logits": _logits(confident=severity < 3),
                    "matched_gt_class": MATCHED_GT_CLASS,
                    "matched_annotation_id": MATCHED_ANNOTATION_ID,
                    "predicted_class": torch.tensor(
                        [0, 1 if severity < 2 else 5, 0, 0, 0, 0, 0, 0]
                    ),
                    "is_matched": IS_MATCHED,
                    "is_correct": torch.tensor(
                        [True, severity < 2, False, False, False, False, False, False]
                    ),
                })
    return directory


def _run_chain(tmp_path: Path) -> dict:
    """Run build-bank, evaluate-knn and report through the CLI, and collect the artifacts."""
    reference = _write_reference_cache(tmp_path / "reference_cache")
    blur = _write_blur_cache(tmp_path / "blur_cache")
    bank = tmp_path / "bank"
    results = tmp_path / "results" / "raw_k3.csv"
    report = tmp_path / "report"

    assert main([
        "build-bank", "--cache", str(reference), "--output", str(bank),
        "--population", "coverage", "--capacity", "16", "--seed", "3",
    ]) == 0
    assert main([
        "evaluate-knn", "--cache", str(blur), "--bank", str(bank), "--output", str(results),
        "--normalization", "raw", "--partition", "tuning", "--k", "3", "--device", "cpu",
        "--policies", "all,threshold_0.5,smooth_1", "--aggregations", "mean,median",
    ]) == 0
    assert main(["report", "--results", str(results), "--output", str(report)]) == 0
    return {
        "reference_manifest": load_manifest(reference),
        "blur_manifest": load_manifest(blur),
        "bank_manifest": json.loads((bank / "manifest.json").read_text(encoding="utf-8")),
        "result_manifest": json.loads(results.with_suffix(".manifest.json").read_text(encoding="utf-8")),
        "summary": json.loads((report / "summary.json").read_text(encoding="utf-8")),
        "report": report,
    }


def test_cli_chain_writes_every_report_artifact(tmp_path: Path):
    artifacts = _run_chain(tmp_path)
    for name in ("per_scene.csv", "summary.json", "run_metadata.json",
                 "raw_trend.png", "relative_trend.png", "class_switch.png"):
        assert (artifacts["report"] / name).exists()


def test_cli_chain_carries_provenance_from_the_caches_into_the_report(tmp_path: Path):
    artifacts = _run_chain(tmp_path)
    bank_manifest = artifacts["bank_manifest"]
    result_manifest = artifacts["result_manifest"]
    metadata = artifacts["summary"]["run_metadata"]

    assert bank_manifest["source_cache_id"] == artifacts["reference_manifest"]["artifact_id"]
    assert result_manifest["bank_id"] == bank_manifest["artifact_id"]
    assert result_manifest["feature_cache_id"] == artifacts["blur_manifest"]["artifact_id"]
    # The report reads its metadata from the result manifest, so the whole chain is
    # recoverable from `summary.json` alone.
    assert metadata["bank_id"] == bank_manifest["artifact_id"]
    assert metadata["feature_cache_id"] == artifacts["blur_manifest"]["artifact_id"]
    assert metadata["normalization"] == "raw"
    assert metadata["k"] == 3
    assert metadata["source_partition"] == "tuning"
    assert metadata["clean_distance_fit"]["0"]["scale"] > 0.0
    assert 0.0 <= metadata["clean_distance_fit"]["0"]["near_duplicate_fraction"] <= 1.0
    assert artifacts["blur_manifest"]["record_count"] == 24
    # The bank keeps its timings in its own manifest, and `evaluate-knn` rolls them up. The
    # feature cache keeps them in a `run_stats.json` sidecar that only `extract` writes, so
    # a cache written directly by `ShardWriter` rolls up as empty rather than as a failure.
    assert metadata["run_stats"]["bank"]["bank_bytes_excluding_manifest"] > 0
    assert metadata["run_stats"]["feature_cache"] == {}


def test_cli_chain_scores_only_the_requested_partition(tmp_path: Path):
    artifacts = _run_chain(tmp_path)
    groups = artifacts["summary"]["groups"]
    assert {group["source_partition"] for group in groups} == {"tuning"}
    combined = [group for group in groups if group["score_scope"] == "combined"]
    # Two tuning images, six severities each, and the held-out test images stay out.
    assert all(group["image_count"] == 2 for group in combined)
    assert all(group["total_severity_count"] == 12 for group in combined)


def test_growing_distance_from_the_bank_reports_as_a_rising_scene_score(tmp_path: Path):
    artifacts = _run_chain(tmp_path)
    combined = {
        (group["policy"], group["aggregation"]): group
        for group in artifacts["summary"]["groups"]
        if group["score_scope"] == "combined"
    }
    for aggregation in ("mean", "median"):
        group = combined[("all", aggregation)]
        assert group["median_spearman"] == 1.0
        assert group["endpoint_increase_rate"] == 1.0
        assert group["mean_adjacent_monotonicity"] == 1.0
        assert group["empty_selection_frequency"] == 0.0
        assert group["scored_severity_count"] == group["total_severity_count"] == 12
        assert group["scored_image_count"] == group["image_count"] == 2
    # Every decoder layer is summarized beside the combined score.
    scopes = {group["score_scope"] for group in artifacts["summary"]["groups"]}
    assert scopes == {"combined", "layer_0", "layer_1", "layer_2"}


def test_a_smoothed_policy_is_reported_under_the_aggregation_that_ran(tmp_path: Path):
    """`--aggregations mean,median` gives `smooth_1/weighted_mean`, and nothing else."""
    artifacts = _run_chain(tmp_path)
    labelled = {
        (group["policy"], group["aggregation"])
        for group in artifacts["summary"]["groups"]
        if group["score_scope"] == "combined"
    }
    assert labelled == {
        ("all", "mean"),
        ("all", "median"),
        ("threshold_0.5", "mean"),
        ("threshold_0.5", "median"),
        ("smooth_1", "weighted_mean"),
    }
    assert artifacts["result_manifest"]["aggregations"] == ["mean", "median"]


def test_a_collapsing_policy_is_distinguishable_from_a_surviving_one(tmp_path: Path):
    """The counts, not the trend statistics, are what expose a truncated sweep."""
    artifacts = _run_chain(tmp_path)
    combined = {
        (group["policy"], group["aggregation"]): group
        for group in artifacts["summary"]["groups"]
        if group["score_scope"] == "combined"
    }
    collapsed = combined[("threshold_0.5", "mean")]
    survived = combined[("all", "mean")]
    # Both read as a perfect trend. Only one of them was measured over the whole sweep.
    assert collapsed["median_spearman"] == survived["median_spearman"] == 1.0
    assert collapsed["scored_severity_count"] == 6
    assert collapsed["total_severity_count"] == 12
    assert collapsed["images_with_unscored_severities"] == 2
    assert collapsed["empty_selection_frequency"] == 0.5
    assert collapsed["median_selected_count_by_severity"] == {
        "0": 8.0, "1": 8.0, "2": 8.0, "3": 0.0, "4": 0.0, "5": 0.0,
    }
    assert artifacts["summary"]["diagnostics"]["groups_with_unscored_severities"] > 0


def test_class_switch_steps_are_split_out_of_the_stable_ones(tmp_path: Path):
    artifacts = _run_chain(tmp_path)
    group = next(
        group for group in artifacts["summary"]["groups"]
        if (group["policy"], group["aggregation"], group["score_scope"]) == ("all", "mean", "combined")
    )
    # One annotation changes class at severity two, so exactly one of each image's five
    # adjacent steps is a switch step.
    assert group["class_switch_step_count"] == 2
    assert group["no_switch_step_count"] == 8
    assert group["mean_adjacent_query_overlap"] == 1.0


def test_the_scene_uncertainty_pipeline_imports_without_supervisely():
    """The study must run in an environment that has no `supervisely` installed.

    Checked in a subprocess: once any other test in the session has imported the training
    solvers, `sys.modules` in this process would answer for them and not for the pipeline.
    """
    program = (
        "import sys;"
        "import src.scene_uncertainty.pipeline;"
        "assert 'supervisely' not in sys.modules;"
        "print('no_supervisely')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert "no_supervisely" in result.stdout


# --------------------------------------------------------------------------------------
# requirements.txt
# --------------------------------------------------------------------------------------

# Import name -> the distribution that provides it, wherever the two differ.
_DISTRIBUTION_OF = {"PIL": "pillow", "yaml": "pyyaml", "cv2": "opencv-python"}


def _declared_requirements() -> dict[str, str]:
    """Every requirement line, keyed by its normalised distribution name."""
    declared = {}
    for line in (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]
        declared[name.lower().replace("_", "-")] = line
    return declared


def _declared_floor(distribution: str) -> tuple[int, ...]:
    match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", _declared_requirements()[distribution])
    assert match, f"{distribution} declares no >= floor"
    return tuple(int(part) for part in match.group(1).split("."))


def test_requirements_declare_every_third_party_module_the_pipeline_imports():
    """A clean environment built from `requirements.txt` must be able to run the CLI.

    `reporting.py` imports pandas and matplotlib at module scope and `pipeline.py` imports
    `reporting`, so an undeclared one of those fails *all six* subcommands at import --
    and it cannot be excused by "it arrives with supervisely", because
    `test_the_scene_uncertainty_pipeline_imports_without_supervisely` forbids exactly that
    dependency being present.
    """
    declared = _declared_requirements()
    missing: dict[str, set[str]] = {}
    for path in sorted((REPOSITORY_ROOT / "src" / "scene_uncertainty").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                if not root or root == "src" or root in sys.stdlib_module_names:
                    continue
                distribution = _DISTRIBUTION_OF.get(root, root).lower().replace("_", "-")
                if distribution not in declared:
                    missing.setdefault(distribution, set()).add(path.name)
    assert not missing, f"requirements.txt declares none of {missing}"


def test_requirements_demand_a_torchvision_that_accepts_a_tuple_labels_getter():
    """`make_coco_loader` passes a tuple-returning `labels_getter`.

    `SanitizeBoundingBoxes` rejected anything but a single tensor or None up to and
    including 0.17.2; the tuple/list branch first appears in 0.18.0. A floor below that
    -- 0.15.2, say -- declares an environment in which the blur loader raises on every
    batch, so the floor is asserted here rather than left to a comment.
    """
    floor = _declared_floor("torchvision")
    assert floor >= (0, 18), f"torchvision floor {floor} predates tuple labels_getter support"
    installed = importlib.metadata.version("torchvision").split("+")[0].split(".")
    assert tuple(int(part) for part in installed[:len(floor)]) >= floor
