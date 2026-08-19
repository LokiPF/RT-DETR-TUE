"""What each subcommand actually does, and the provenance it leaves behind.

Four things in here are easy to get subtly wrong, so they are stated once, up front:

* **One extractor instance per run, never one per batch.** `ClassificationPersistenceExtractor`
  registers its forward hooks in `__init__`, not in `__enter__`, and has no double-registration
  guard. Constructing one per batch leaves one stale hook per decoder layer per batch attached
  to the live model: every later forward pays an extra `F.linear` for each of them and keeps its
  `(B, query, hidden)` and `(B, query, class)` captures alive, so GPU memory grows linearly over
  a long sweep. The single `with` around the whole severity sweep below is load-bearing.
* **One writer per artifact directory.** `ShardWriter`'s immutability guard is a file check and
  is therefore racy: two extractions pointed at the same `--output` both pass it, the second
  resumes the first's partial manifest, and both then write the same shard index. Records are
  lost with no error. This is documented in `--help` and cannot be enforced from here.
* **Timing and chunk widths are recorded but are not part of a content address.** Wall time,
  peak memory and `--bank-chunk-size` describe how an artifact was produced, not what it
  contains; chunked kNN is algebraically exact, so two runs that differ only in the chunk width
  hold the same answers to ~1 ULP. `_content_address` therefore hashes everything except
  `run_stats`, and nothing anywhere compares `run_stats` when deciding whether two artifacts are
  interchangeable.
* **The clean-distance fit is published as JSON, not only inside the pickled normalizers.**
  `fit_clean_distance_scale` returns `min_neighbor_distance` and `near_duplicate_fraction`
  precisely so a human can tell whether near-duplicate bank rows contaminated the fit -- 20%
  twins inflate `scale` by ~2.8x, which rescales every score in the results table by ~65%.
  Those numbers are useless if reading them needs a `torch.load`, so they also go into the
  result manifest, and a contaminated fit prints a warning.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO

from src.zoo.rtdetr.matcher import HungarianMatcher

from .artifacts import ShardWriter, assert_compatible, iter_records, load_manifest, manifest_id
from .bank import deterministic_reservoir, streaming_coverage_bank
from .dataset import make_coco_loader
from .evaluate import compute_query_distances, score_cached_record
from .extractor import ClassificationPersistenceExtractor
from .knn import fit_clean_distance_scale
from .normalization import fit_normalizer, transform_vectors
from .query_policy import ORACLE_POLICIES
from .reference import select_evaluation_ids, select_reference_ids
from .reporting import read_result_csv, write_report, write_result_csv
from .runtime import checkpoint_sha256, load_frozen_detector


BLUR_RADII = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0, 5: 12.0}
DEPLOYABLE_POLICIES = ("all", "top10", "top20", "top50", "threshold_0.2", "threshold_0.3", "threshold_0.5", "smooth_1", "smooth_2")
POLICIES = DEPLOYABLE_POLICIES + ORACLE_POLICIES
AGGREGATIONS = ("mean", "median", "q90", "top20_mean")

DECODER_LAYERS = (0, 1, 2)
QUERY_COUNT = 300
PERSISTENCE_DIM = 335

# What a manifest is addressed by is the inputs and settings that determine its contents, and
# these three keys are not that. `run_stats` is wall time, peak memory and the kNN chunk width.
# `clean_distance_fit` is a float-valued *output* republished as a diagnostic, and it moves in
# the last ULP with the chunk width (chunked kNN is algebraically exact but not bit-identical),
# so hashing it would put that pure performance knob back into the address through the back
# door and give two interchangeable runs two different identities. See the module docstring.
NON_IDENTIFYING_KEYS = ("artifact_id", "run_stats", "clean_distance_fit")

# Above this share of sampled bank rows sitting on top of a near-duplicate, the clean-distance
# `scale` starts to move measurably (see `fit_clean_distance_scale`), so the run says so out loud.
NEAR_DUPLICATE_WARNING_FRACTION = 0.01


class PipelineError(ValueError):
    """A failure the operator can act on: bad arguments, a missing or incompatible artifact.

    Subclasses `ValueError` so the modules underneath, which raise plain `ValueError` for the
    same class of problem, can be re-raised as one without changing what callers catch.
    """


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _content_address(manifest: dict) -> str:
    return manifest_id({
        key: value for key, value in manifest.items() if key not in NON_IDENTIFYING_KEYS
    })


def _report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _existing_path(value, flag: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise PipelineError(f"{flag} does not exist: {path}")
    return path


def _existing_artifact(value, flag: str) -> Path:
    directory = _existing_path(value, flag)
    if not (directory / "manifest.json").exists():
        raise PipelineError(f"{flag} is not a finished artifact, no manifest.json in: {directory}")
    return directory


def _device(name: str) -> torch.device:
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as error:
        raise PipelineError(f"--device {name} is not a device: {error}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PipelineError(f"--device {name} was requested but CUDA is not available")
    return device


def _reset_peak_memory(device: torch.device) -> None:
    """Start this run's peak-CUDA-memory measurement.

    `torch.cuda.reset_peak_memory_stats` raises `RuntimeError: Invalid device argument` on a
    process that has not touched CUDA yet (measured on torch 2.11.0+cu128, for a `torch.device`
    and for a bare index alike), which is exactly the state every one of these commands is in
    when it starts. Initialising the context first is what makes the instrumentation survive
    being the first CUDA call in the process.
    """
    if device.type != "cuda":
        return
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_bytes(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _names(value, allowed: tuple[str, ...], flag: str) -> tuple[str, ...]:
    """Accept either the parser's validated tuple or a raw comma-separated string.

    The parser already rejects unknown names, which is what makes a typo cost nothing instead of
    an hour of GPU time. This repeats the check because `command_evaluate_knn` is also called
    directly, from tests and from a driver script, where nothing has validated anything.
    """
    if isinstance(value, str):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    else:
        names = tuple(value)
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise PipelineError(f"{flag} names {unknown} are unknown; choose from {list(allowed)}")
    if not names:
        raise PipelineError(f"{flag} needs at least one name")
    return names


def _git_commit() -> str:
    """The commit the run came from, or `unknown` when that cannot be established.

    The extraction hosts run from an rsync'd tree with no usable git metadata, so this must not
    be able to fail a six-hour extraction before it starts. `SCENE_UNCERTAINTY_GIT_COMMIT` is
    read first so such a host can be told the answer it cannot work out for itself.
    """
    override = os.environ.get("SCENE_UNCERTAINTY_GIT_COMMIT")
    if override:
        return override.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --------------------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------------------


def command_select(args) -> None:
    train_ann = _existing_path(args.train_ann, "--train-ann")
    val_ann = _existing_path(args.val_ann, "--val-ann")
    output = Path(args.output)
    train = COCO(str(train_ann))
    available = len(train.imgs)
    requested = args.natural_count + args.augmentation_budget
    if requested > available:
        raise PipelineError(
            f"--natural-count {args.natural_count} + --augmentation-budget "
            f"{args.augmentation_budget} = {requested} exceeds the {available} images in {train_ann}"
        )
    started = time.perf_counter()
    reference = select_reference_ids(
        train,
        natural_count=args.natural_count,
        augmentation_budget=args.augmentation_budget,
        quota=args.quota,
        seed=args.seed,
    )
    # The rare-class pass is a greedy loop that rescans every candidate image on every pick, so
    # it is the one part of this command whose cost is not obvious from the arguments. It is
    # measured rather than assumed, and the measurement is kept beside the ids it produced.
    elapsed = time.perf_counter() - started
    reference["run_stats"] = {
        "wall_seconds": elapsed,
        "candidate_image_count": available,
        "natural_count": args.natural_count,
        "augmentation_budget": args.augmentation_budget,
        "quota": args.quota,
        "unmet_category_count": len(reference["unmet_category_quotas"]),
    }
    _report(f"select: reference ids for {len(train.imgs)} candidates in {elapsed:.1f}s")
    _write_json(output / "reference.json", reference)
    validation = COCO(str(val_ann))
    _write_json(output / "evaluation.json", select_evaluation_ids(validation.getImgIds(), seed=args.seed))


# --------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------


def _matcher() -> HungarianMatcher:
    return HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )


def _selected_image_groups(selection: dict, reference: bool, limit: int | None) -> dict[int, str]:
    if reference:
        groups = {
            **{int(image_id): "natural" for image_id in selection["natural_ids"]},
            **{int(image_id): "augmentation" for image_id in selection["augmentation_ids"]},
        }
        if limit is not None:
            groups = dict(list(groups.items())[:limit])
        return groups
    tuning = [int(value) for value in selection["tuning_pilot_ids"]]
    test = [int(value) for value in selection["test_pilot_ids"]]
    if limit is not None:
        tuning_count = min(len(tuning), (limit + 1) // 2)
        test_count = min(len(test), limit - tuning_count)
        tuning, test = tuning[:tuning_count], test[:test_count]
    return {
        **{image_id: "tuning" for image_id in tuning},
        **{image_id: "test" for image_id in test},
    }


def _extraction_metadata(args, reference: bool, selection: dict, conditions, image_ids) -> dict:
    if "seed" not in selection:
        raise PipelineError(f"--selection {args.selection} has no seed; it was not written by `select`")
    return {
        "git_commit": _git_commit(),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "model_config": str(Path(args.config).resolve()),
        "model_config_sha256": checkpoint_sha256(args.config),
        "annotation_file": str(Path(args.annotations).resolve()),
        "annotation_sha256": checkpoint_sha256(args.annotations),
        "image_root": str(Path(args.images).resolve()),
        "image_ids": image_ids,
        "split_seed": int(selection["seed"]),
        "source_kind": "reference" if reference else "evaluation",
        "preprocessing": "resize_640x640_then_pil_gaussian_blur_then_float_tensor",
        "corruption": {"type": "gaussian_blur", "radii": dict(conditions)},
        "persistence_extraction_version": 1,
        "classification_layers": [f"decoder.dec_score_head.{layer_id}" for layer_id in DECODER_LAYERS],
        "decoder_layers": list(DECODER_LAYERS),
        "persistence_dim": PERSISTENCE_DIM,
        "query_count": QUERY_COUNT,
    }


def _extract(args, reference: bool) -> None:
    for flag in ("config", "checkpoint", "images", "annotations", "selection"):
        _existing_path(getattr(args, flag), f"--{flag}")
    device = _device(args.device)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    groups = _selected_image_groups(selection, reference, args.limit)
    if not groups:
        raise PipelineError(f"--selection {args.selection} names no images to extract")
    conditions = [(0, 0.0)] if reference else sorted(BLUR_RADII.items())
    image_ids = list(groups)
    metadata = _extraction_metadata(args, reference, selection, conditions, image_ids)

    output = Path(args.output)
    if (output / "manifest.json").exists():
        # Already finished, so this is a no-op -- but only once the finished artifact has been
        # checked to be the one this command would have produced.
        try:
            assert_compatible(load_manifest(output), metadata, keys=metadata)
        except ValueError as error:
            raise PipelineError(f"{output} was extracted with different settings: {error}") from error
        _report(f"extract: {output} is already complete, nothing to do")
        return

    started = time.perf_counter()
    _reset_peak_memory(device)
    try:
        writer = ShardWriter(output, metadata, args.shard_size)
    except ValueError as error:
        raise PipelineError(f"{output} cannot be resumed by this command: {error}") from error
    with writer:
        scan_started = time.perf_counter()
        completed = writer.existing_record_keys()
        if completed:
            # Recovering these keys deserializes every finished shard, tensor payloads included,
            # so on a large cache this is a full re-read of what is already on disk. It is
            # reported rather than hidden, because it can dominate the restart of a big run.
            _report(
                f"extract: resume scan re-read {len(writer.shards)} shards "
                f"({len(completed)} records) in {time.perf_counter() - scan_started:.1f}s"
            )
        model = load_frozen_detector(args.config, args.checkpoint, device)
        expected_shape = (metadata["query_count"], metadata["persistence_dim"])
        # One extractor, one hook registration, for every severity and every batch below.
        with ClassificationPersistenceExtractor(model, _matcher(), DECODER_LAYERS) as extractor:
            for severity, radius in conditions:
                pending_ids = [
                    image_id for image_id in image_ids if (image_id, severity) not in completed
                ]
                if not pending_ids:
                    continue
                loader = make_coco_loader(
                    args.images, args.annotations, pending_ids, radius,
                    args.batch_size, args.num_workers,
                )
                for samples, targets in loader:
                    samples = samples.to(device)
                    device_targets = [
                        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
                        for target in targets
                    ]
                    for record in extractor.extract(samples, device_targets):
                        for layer_id, vectors in record["layers"].items():
                            if tuple(vectors.shape) != expected_shape:
                                raise PipelineError(
                                    f"Unexpected persistence shape at layer {layer_id}: "
                                    f"{tuple(vectors.shape)}, expected {expected_shape}"
                                )
                        record["severity"] = int(severity)
                        record["blur_radius"] = float(radius)
                        record["corruption_type"] = "gaussian_blur"
                        record["source_partition"] = "reference" if reference else groups[record["image_id"]]
                        record["reference_group"] = groups[record["image_id"]] if reference else None
                        writer.add(record)
        writer.metadata["run_stats"] = {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": _peak_memory_bytes(device),
        }
    _report(f"extract: wrote {load_manifest(output)['record_count']} records to {output}")


def command_extract_reference(args) -> None:
    _extract(args, reference=True)


def command_extract_blur(args) -> None:
    _extract(args, reference=False)


# --------------------------------------------------------------------------------------
# build-bank
# --------------------------------------------------------------------------------------


def _atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _iter_layer_vectors(cache: str | Path, layer_id: int, natural_only: bool):
    for record in iter_records(cache):
        if natural_only and record["reference_group"] != "natural":
            continue
        yield from record["layers"][layer_id]


def command_build_bank(args) -> None:
    started = time.perf_counter()
    cache = _existing_artifact(args.cache, "--cache")
    output = Path(args.output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Bank artifact is already complete: {output}")
    cache_manifest = load_manifest(cache)
    if cache_manifest["source_kind"] != "reference":
        raise PipelineError(
            f"Banks must be built from a reference cache; {cache} holds "
            f"{cache_manifest['source_kind']} features"
        )
    output.mkdir(parents=True, exist_ok=True)
    bank_manifest = {
        "schema_version": 1,
        "artifact_type": "class_independent_query_bank",
        "source_cache_id": cache_manifest["artifact_id"],
        "source_image_ids": cache_manifest["image_ids"],
        "checkpoint_sha256": cache_manifest["checkpoint_sha256"],
        "decoder_layers": cache_manifest["decoder_layers"],
        "persistence_dim": cache_manifest["persistence_dim"],
        "query_count": cache_manifest["query_count"],
        "population": args.population,
        "capacity": args.capacity,
        "seed": args.seed,
        "class_independent": True,
        "layers": {},
    }
    for layer_id in cache_manifest["decoder_layers"]:
        try:
            if args.population == "coverage":
                bank, sampling = streaming_coverage_bank(
                    iter_records(cache), layer_id, args.capacity,
                    object_fraction=0.5, class_count=80, seed=args.seed + layer_id,
                )
            else:
                bank = deterministic_reservoir(
                    _iter_layer_vectors(cache, layer_id, natural_only=True),
                    args.capacity, args.seed + layer_id,
                )
                sampling = {"natural_vectors": int(bank.shape[0])}
        except ValueError as error:
            raise PipelineError(
                f"Cannot build a {args.population} bank for layer {layer_id} from {cache}: {error}"
            ) from error
        layer_path = output / f"layer_{layer_id}.pt"
        _atomic_torch_save({"vectors": bank, "sampling": sampling}, layer_path)
        bank_manifest["layers"][str(layer_id)] = {
            "path": layer_path.name,
            "vector_count": int(bank.shape[0]),
            "sampling": sampling,
        }
    bank_manifest["run_stats"] = {
        "wall_seconds": time.perf_counter() - started,
        "bank_bytes_excluding_manifest": sum(
            (output / layer["path"]).stat().st_size for layer in bank_manifest["layers"].values()
        ),
    }
    bank_manifest["artifact_id"] = _content_address(bank_manifest)
    _write_json(output / "manifest.json", bank_manifest)
    _report(f"build-bank: wrote {len(bank_manifest['layers'])} layer banks to {output}")


# --------------------------------------------------------------------------------------
# evaluate-knn
# --------------------------------------------------------------------------------------


def _validate_evaluation_cache(cache_manifest: dict, bank_manifest: dict, records) -> set[str]:
    """Check a feature cache against the bank it will be scored with. Returns its partitions."""
    try:
        assert_compatible(
            cache_manifest,
            bank_manifest,
            keys=("checkpoint_sha256", "decoder_layers", "persistence_dim", "query_count"),
        )
    except ValueError as error:
        raise PipelineError(f"The bank does not match the feature cache: {error}") from error
    bank_layers = {int(layer_id) for layer_id in bank_manifest["layers"]}
    if bank_layers != set(cache_manifest["decoder_layers"]):
        raise PipelineError(
            f"The bank holds layer banks for {sorted(bank_layers)}, the cache holds decoder "
            f"layers {sorted(cache_manifest['decoder_layers'])}"
        )
    overlap = set(cache_manifest["image_ids"]) & set(bank_manifest["source_image_ids"])
    if overlap:
        raise PipelineError(f"Reference/evaluation image leakage: {sorted(overlap)[:10]}")
    partitions: set[str] = set()
    severities: dict[int, set[int]] = {}
    for record in records:
        image_id = int(record["image_id"])
        severity = int(record["severity"])
        partitions.add(record["source_partition"])
        if tuple(record["logits"].shape) != (cache_manifest["query_count"], 80):
            raise PipelineError(f"Query-count mismatch for image {image_id}, severity {severity}")
        if severity in severities.setdefault(image_id, set()):
            raise PipelineError(f"Duplicate image/severity record: {(image_id, severity)}")
        severities[image_id].add(severity)
    incomplete = {image_id: sorted(values) for image_id, values in severities.items() if values != set(BLUR_RADII)}
    if incomplete:
        raise PipelineError(f"Incomplete blur sweeps: {dict(list(incomplete.items())[:5])}")
    return partitions


def _clean_distance_summary(layer_score_scales: dict[int, dict]) -> dict:
    """The clean-distance fit as plain JSON, so it can be read without unpickling anything."""
    return {
        str(layer_id): {
            "center": float(state["center"]),
            "scale": float(state["scale"]),
            "sample_count": int(state["sample_count"]),
            "min_neighbor_distance": float(state["min_neighbor_distance"]),
            "near_duplicate_fraction": float(state["near_duplicate_fraction"]),
        }
        for layer_id, state in sorted(layer_score_scales.items())
    }


def _warn_about_contaminated_fits(summary: dict) -> None:
    """Say out loud that a layer's clean-distance scale was fitted on near-duplicate bank rows.

    Printed as soon as the fit exists rather than with the manifest at the end, so an operator
    can stop a long scoring pass whose divisor is already known to be wrong.
    """
    for layer_id, fit in summary.items():
        if fit["near_duplicate_fraction"] > NEAR_DUPLICATE_WARNING_FRACTION:
            _report(
                f"evaluate-knn: WARNING layer {layer_id} fitted its clean-distance scale on a bank "
                f"where {fit['near_duplicate_fraction']:.1%} of sampled rows have a near-duplicate; "
                f"scale={fit['scale']:.4g} is inflated and every score is divided by it"
            )


def command_evaluate_knn(args) -> None:
    started = time.perf_counter()
    # Names and devices first: a typo must not cost a bank load, let alone a scoring pass.
    policies = _names(args.policies, POLICIES, "--policies")
    aggregations_requested = _names(args.aggregations, AGGREGATIONS, "--aggregations")
    device = _device(args.device)
    cache = _existing_artifact(args.cache, "--cache")
    bank_root = _existing_artifact(args.bank, "--bank")

    bank_manifest = json.loads((bank_root / "manifest.json").read_text(encoding="utf-8"))
    cache_manifest = load_manifest(cache)
    partitions = _validate_evaluation_cache(cache_manifest, bank_manifest, iter_records(cache))
    if args.partition != "all" and args.partition not in partitions:
        raise PipelineError(
            f"--partition {args.partition} matches no record in {cache}, which holds "
            f"{sorted(partitions)}"
        )
    smallest_bank = min(layer["vector_count"] for layer in bank_manifest["layers"].values())
    if args.k >= smallest_bank:
        raise PipelineError(
            f"--k {args.k} needs a bank of more than {args.k} vectors; the smallest layer bank "
            f"in {bank_root} holds {smallest_bank}"
        )

    _reset_peak_memory(device)
    raw_banks = {
        int(layer_id): torch.load(
            bank_root / layer["path"], map_location="cpu", weights_only=False
        )["vectors"]
        for layer_id, layer in bank_manifest["layers"].items()
    }
    normalizers = {
        layer_id: fit_normalizer(bank, args.normalization)
        for layer_id, bank in raw_banks.items()
    }
    banks = {
        layer_id: transform_vectors(bank, normalizers[layer_id]).to(device)
        for layer_id, bank in raw_banks.items()
    }
    layer_score_scales = {
        layer_id: fit_clean_distance_scale(
            bank, args.k, seed=42 + layer_id, bank_chunk_size=args.bank_chunk_size
        )
        for layer_id, bank in banks.items()
    }
    clean_distance_fit = _clean_distance_summary(layer_score_scales)
    _warn_about_contaminated_fits(clean_distance_fit)
    rows: list[dict] = []
    query_distance_rows: list[dict] = []
    for record in iter_records(cache):
        if args.partition != "all" and record["source_partition"] != args.partition:
            continue
        normalized = {**record, "layers": {
            layer_id: transform_vectors(record["layers"][layer_id], normalizers[layer_id])
            for layer_id in banks
        }}
        query_distances = compute_query_distances(
            normalized, banks, args.k, bank_chunk_size=args.bank_chunk_size
        )
        query_distance_rows.append({
            "image_id": int(record["image_id"]),
            "severity": int(record["severity"]),
            "source_partition": record["source_partition"],
            "query_scores_by_layer": {
                layer_id: values.to(torch.float16)
                for layer_id, values in query_distances.items()
            },
        })
        for policy in policies:
            aggregations = ("mean",) if policy.startswith("smooth_") else aggregations_requested
            for aggregation in aggregations:
                rows.append(score_cached_record(
                    normalized, banks, policy, aggregation, args.k,
                    bank_chunk_size=args.bank_chunk_size,
                    query_distances=query_distances,
                    layer_score_scales=layer_score_scales,
                ))
    if not rows:
        raise PipelineError(f"Nothing to score in {cache} for --partition {args.partition}")

    output = Path(args.output)
    write_result_csv(rows, output)
    query_distance_path = output.with_suffix(".query_distances.pt")
    _atomic_torch_save(query_distance_rows, query_distance_path)
    normalizer_path = output.with_suffix(".normalizers.pt")
    _atomic_torch_save({
        "feature_normalizers": normalizers,
        "layer_score_scales": layer_score_scales,
    }, normalizer_path)
    result_manifest = {
        "schema_version": 1,
        "artifact_type": "knn_scene_uncertainty_results",
        "feature_cache_id": cache_manifest["artifact_id"],
        "bank_id": bank_manifest["artifact_id"],
        "normalization": args.normalization,
        "source_partition": args.partition,
        "normalizer_path": normalizer_path.name,
        "combined_layer_score": "equal_mean_after_clean_reference_median_iqr_scaling",
        "clean_distance_fit": clean_distance_fit,
        "query_distance_path": query_distance_path.name,
        "query_distance_record_count": len(query_distance_rows),
        "k": args.k,
        "policies": list(policies),
        "aggregations": list(aggregations_requested),
        "row_count": len(rows),
        "run_stats": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": _peak_memory_bytes(device),
            # Performance only: chunked kNN is exact, so this changes speed and memory, never
            # the answer. Recorded for provenance, excluded from the artifact id, and never
            # compared when deciding whether two result sets are interchangeable.
            "bank_chunk_size": args.bank_chunk_size,
            "feature_cache": cache_manifest.get("run_stats", {}),
            "bank": bank_manifest.get("run_stats", {}),
        },
    }
    result_manifest["artifact_id"] = _content_address(result_manifest)
    _write_json(output.with_suffix(".manifest.json"), result_manifest)
    _report(f"evaluate-knn: scored {len(rows)} rows from {len(query_distance_rows)} scenes into {output}")


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def command_report(args) -> None:
    results = _existing_path(args.results, "--results")
    metadata_path = results.with_suffix(".manifest.json")
    if not metadata_path.exists():
        raise PipelineError(f"No result manifest beside {results}; expected {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = read_result_csv(results)
    if not rows:
        raise PipelineError(
            f"{results} holds no scored rows, so there is nothing to report; it was written for "
            f"--partition {metadata.get('source_partition', 'unknown')}"
        )
    write_report(rows, args.output, run_metadata=metadata)
    _report(f"report: summarized {len(rows)} rows into {args.output}")


COMMANDS = {
    "select": command_select,
    "extract-reference": command_extract_reference,
    "extract-blur": command_extract_blur,
    "build-bank": command_build_bank,
    "evaluate-knn": command_evaluate_knn,
    "report": command_report,
}
