from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as vision

from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr import RTDETR
from src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2

from .artifacts import (
    ShardWriter,
    _safe_load_shard,
    iter_records,
    load_manifest as load_artifact_manifest,
)
from .config import FIXED_CONFIG, ExperimentConfig
from .corruptions.base import Corruption, Severity
from .manifests import ManifestEntry
from .persistence import Layer2Capture, batched_persistence


CLEAN_ONLY = (Severity(0, 0.0),)
_ARTIFACT_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "shard_size",
        "record_count",
        "shards",
        "shard_sha256",
    }
)
_CACHE_TENSOR_DTYPES = {
    "boxes": torch.float32,
    "logits": torch.float16,
    "persistence": torch.float16,
}
_CACHE_RECORD_KEYS = frozenset(
    {"image_id", "severity", *_CACHE_TENSOR_DTYPES}
)


@dataclass(frozen=True)
class _RecordContract:
    query_count: int
    class_count: int
    persistence_dim: int


def build_fixed_detector() -> RTDETR:
    spatial_size = [640, 640]
    backbone = PResNet(
        depth=18,
        variant="d",
        num_stages=4,
        return_idx=[1, 2, 3],
        act="relu",
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
    )
    encoder = HybridEncoder(
        in_channels=[128, 256, 512],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10_000,
        expansion=0.5,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=spatial_size,
        version="v2",
    )
    decoder = RTDETRTransformerv2(
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[4, 4, 4],
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=spatial_size,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
    )
    return RTDETR(backbone=backbone, encoder=encoder, decoder=decoder)


def _tensor_state(value: Any, location: str) -> dict[str, Tensor]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} state must be a mapping of tensors")
    if not value:
        raise KeyError(f"{location} state is empty")
    if not all(
        isinstance(key, str) and isinstance(item, Tensor)
        for key, item in value.items()
    ):
        raise TypeError(f"{location} state must contain only named tensors")
    return dict(value)


def checkpoint_state(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        if not isinstance(ema, Mapping):
            raise TypeError("checkpoint ema must be a mapping")
        if "module" not in ema:
            raise KeyError("checkpoint ema has no module state")
        return _tensor_state(ema["module"], "checkpoint ema.module")
    if "model" in checkpoint:
        return _tensor_state(checkpoint["model"], "checkpoint model")
    if checkpoint and all(isinstance(item, Tensor) for item in checkpoint.values()):
        return _tensor_state(checkpoint, "checkpoint direct")
    if not checkpoint:
        raise KeyError("checkpoint state is empty")
    if any(isinstance(item, Tensor) for item in checkpoint.values()):
        raise TypeError("direct checkpoint state must contain only named tensors")
    raise KeyError("checkpoint has neither ema.module nor model state")


def load_frozen_detector(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model = build_fixed_detector()
    incompatible = model.load_state_dict(
        checkpoint_state(checkpoint),
        strict=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).eval().requires_grad_(False)
    return model


def resize_image(
    image: Image.Image,
    image_size: tuple[int, int],
) -> Image.Image:
    rgb = image.convert("RGB")
    resized: Image.Image | None = None
    try:
        resized = vision.resize(rgb, list(image_size), antialias=True)
        return resized
    finally:
        if rgb is not image and rgb is not resized:
            rgb.close()


def image_tensor(resized: Image.Image) -> Tensor:
    return vision.pil_to_tensor(resized).to(torch.float32).div_(255.0)


def prepare_image(
    image: Image.Image,
    image_size: tuple[int, int],
) -> Tensor:
    resized = resize_image(image, image_size)
    try:
        return image_tensor(resized)
    finally:
        resized.close()


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_image_size(value: object) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError("image_size must be two positive integers")
    return value


def _validated_entries(value: object) -> tuple[ManifestEntry, ...]:
    if type(value) is not tuple or not value:
        raise ValueError("entries must be a nonempty tuple")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for entry in value:
        if not isinstance(entry, ManifestEntry):
            raise ValueError("entries must contain only ManifestEntry values")
        if type(entry.image_id) is not str or not entry.image_id.strip():
            raise ValueError("each extraction entry needs a nonempty image_id")
        if entry.image_id in seen_ids:
            raise ValueError(f"duplicate extraction image_id {entry.image_id!r}")
        if not isinstance(entry.path, Path) or not entry.path.is_absolute():
            raise ValueError("each extraction entry path must be an absolute Path")
        if not entry.path.is_file():
            raise ValueError(f"extraction image does not exist: {entry.path}")
        if entry.path in seen_paths:
            raise ValueError(f"duplicate extraction image path: {entry.path}")
        seen_ids.add(entry.image_id)
        seen_paths.add(entry.path)
    return value


def _validated_metadata(value: object) -> dict:
    if type(value) is not dict:
        raise TypeError("extraction metadata must be a dictionary")
    reserved = sorted(_ARTIFACT_MANIFEST_KEYS.intersection(value))
    if reserved:
        raise ValueError(f"extraction metadata uses reserved key: {reserved[0]}")
    try:
        return json.loads(
            json.dumps(value, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("extraction metadata must be JSON serializable") from error


def _validated_severities(
    corruption: Corruption | None,
) -> tuple[Severity, ...]:
    if corruption is None:
        return CLEAN_ONLY
    severities = getattr(corruption, "severities", None)
    if type(severities) is not tuple or not all(
        isinstance(item, Severity) for item in severities
    ):
        raise ValueError("corruption severities must be a tuple of Severity values")
    levels = [item.level for item in severities]
    if any(type(level) is not int for level in levels) or levels != list(range(6)):
        raise ValueError(
            "corruption severity levels must be exactly 0 through 5"
        )
    if any(
        type(item.parameter) not in (int, float)
        or isinstance(item.parameter, bool)
        or not isfinite(item.parameter)
        for item in severities
    ):
        raise ValueError("corruption severity parameters must be finite numbers")
    if not callable(getattr(corruption, "apply", None)):
        raise ValueError("corruption must provide an apply method")
    return severities


def _expected_keys(
    entries: tuple[ManifestEntry, ...],
    severities: tuple[Severity, ...],
) -> list[tuple[str, int]]:
    return [
        (entry.image_id, severity.level)
        for entry in entries
        for severity in severities
    ]


def _record_identity(record: object, *, index: int) -> tuple[str, int]:
    if type(record) is not dict:
        raise RuntimeError(f"extractor record {index} must be a dictionary")
    image_id = record.get("image_id")
    if type(image_id) is not str:
        raise RuntimeError(
            f"extractor record {index} must contain a string image_id"
        )
    severity = record.get("severity")
    if type(severity) is not int:
        raise RuntimeError(
            f"extractor record {index} must contain an integer severity"
        )
    return image_id, severity


def _validated_record(
    record: object,
    *,
    index: int,
    contract: _RecordContract | None,
    require_cache_dtypes: bool,
) -> tuple[tuple[str, int], dict[str, Tensor], _RecordContract]:
    identity = _record_identity(record, index=index)
    assert type(record) is dict
    tensors: dict[str, Tensor] = {}
    for name, cache_dtype in _CACHE_TENSOR_DTYPES.items():
        tensor = record.get(name)
        if not isinstance(tensor, Tensor):
            raise RuntimeError(
                f"extractor record {index} must contain tensor {name}"
            )
        if tensor.layout != torch.strided:
            raise RuntimeError(
                f"extractor record {index} {name} must have strided layout"
            )
        if require_cache_dtypes and tensor.dtype != cache_dtype:
            raise RuntimeError(
                f"extractor record {index} {name} must have dtype "
                f"{cache_dtype}"
            )
        if not tensor.is_floating_point():
            raise RuntimeError(
                f"extractor record {index} {name} must be floating-point"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(
                f"extractor record {index} {name} must contain only finite values"
            )
        tensors[name] = tensor

    if frozenset(record) != _CACHE_RECORD_KEYS:
        raise RuntimeError(
            f"extractor record {index} must contain exactly the cache record keys"
        )

    for name in ("logits", "persistence"):
        if tensors[name].ndim != 2:
            raise RuntimeError(
                f"extractor record {index} {name} must have rank 2"
            )
    boxes = tensors["boxes"]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise RuntimeError(
            f"extractor record {index} boxes must have shape (queries, 4)"
        )
    query_counts = {
        tensors["logits"].shape[0],
        boxes.shape[0],
        tensors["persistence"].shape[0],
    }
    if len(query_counts) != 1:
        raise RuntimeError("extractor record query counts must match")
    query_count = query_counts.pop()
    if query_count <= 0:
        raise RuntimeError("extractor record query count must be positive")
    class_count = tensors["logits"].shape[1]
    persistence_dim = tensors["persistence"].shape[1]
    if class_count <= 0:
        raise RuntimeError("extractor record class count must be positive")
    if persistence_dim <= 0:
        raise RuntimeError("extractor record persistence width must be positive")
    current = _RecordContract(query_count, class_count, persistence_dim)
    if contract is not None and current != contract:
        raise RuntimeError(
            f"extractor cache record contract changed from {contract!r} "
            f"to {current!r}"
        )
    return identity, tensors, current


def _validated_records(
    records: object,
    identities: list[tuple[str, int]],
    contract: _RecordContract | None,
) -> tuple[list[dict[str, object]], _RecordContract]:
    if type(records) is not list:
        raise RuntimeError("extractor must return a list of records")
    if len(records) != len(identities):
        raise RuntimeError(
            f"extractor returned {len(records)} records; "
            f"expected {len(identities)}"
        )

    normalized: list[dict[str, object]] = []
    actual_keys: list[tuple[str, int]] = []
    for index, record in enumerate(records):
        identity, tensors, contract = _validated_record(
            record,
            index=index,
            contract=contract,
            require_cache_dtypes=False,
        )
        actual_keys.append(identity)
        normalized_tensors: dict[str, Tensor] = {}
        for name, tensor in tensors.items():
            normalized_tensor = tensor.detach().to(
                device="cpu",
                dtype=_CACHE_TENSOR_DTYPES[name],
                copy=True,
            )
            if not bool(torch.isfinite(normalized_tensor).all().item()):
                raise RuntimeError(
                    f"extractor record {index} normalized {name} "
                    "must contain only finite values"
                )
            normalized_tensors[name] = normalized_tensor
        normalized.append(
            {
                "image_id": identity[0],
                "severity": identity[1],
                **normalized_tensors,
            }
        )
    if actual_keys != identities:
        raise RuntimeError(
            f"extractor returned keys {actual_keys!r}; expected {identities!r}"
        )
    if contract is None:
        raise RuntimeError("extractor returned no cache record contract")
    return normalized, contract


def _published_record_state(
    writer: ShardWriter,
) -> tuple[list[tuple[str, int]], _RecordContract | None]:
    directory_fd = writer._lock_fd
    if directory_fd is None:
        raise RuntimeError("artifact writer has no active directory descriptor")
    identities: list[tuple[str, int]] = []
    contract: _RecordContract | None = None
    for name in writer.shards:
        records = _safe_load_shard(
            writer.directory,
            name,
            writer.shard_sha256[name],
            directory_fd=directory_fd,
        )
        for record in records:
            identity, _, contract = _validated_record(
                record,
                index=len(identities),
                contract=contract,
                require_cache_dtypes=True,
            )
            identities.append(identity)
    return identities, contract


def _entry_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _validate_completed_extraction(
    entries: tuple[ManifestEntry, ...],
    root: Path,
    metadata: dict,
    severities: tuple[Severity, ...],
) -> None:
    expected_keys = _expected_keys(entries, severities)
    actual = load_artifact_manifest(root)
    actual_metadata = {
        key: value
        for key, value in actual.items()
        if key not in _ARTIFACT_MANIFEST_KEYS
    }
    if actual_metadata != metadata:
        raise ValueError(
            "completed extraction metadata mismatch: "
            f"actual={actual_metadata!r}, expected={metadata!r}"
        )
    actual_count = actual.get("record_count")
    if type(actual_count) is not int or actual_count != len(expected_keys):
        raise RuntimeError(
            "completed extraction has "
            f"{actual_count!r} records; expected {len(expected_keys)}"
        )
    actual_keys: list[tuple[str, int]] = []
    contract: _RecordContract | None = None
    for index, record in enumerate(iter_records(root)):
        identity, _, contract = _validated_record(
            record,
            index=index,
            contract=contract,
            require_cache_dtypes=True,
        )
        actual_keys.append(identity)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "completed extraction record roster does not match the input"
        )


def validate_extraction_cache(
    entries: tuple[ManifestEntry, ...],
    directory: str | Path,
    metadata: dict,
    corruption: Corruption | None,
) -> bool:
    """Validate a complete cache, returning false only when no manifest exists."""
    entries = _validated_entries(entries)
    metadata = _validated_metadata(metadata)
    severities = _validated_severities(corruption)
    root = Path(directory)
    if not _entry_present(root / "manifest.json"):
        return False
    _validate_completed_extraction(entries, root, metadata, severities)
    return True


def _pending_samples(
    entries: tuple[ManifestEntry, ...],
    corruption: Corruption | None,
    severities: tuple[Severity, ...],
    existing: set[tuple[str, int]],
    image_size: tuple[int, int],
):
    expected_size = (image_size[1], image_size[0])
    for entry in entries:
        with Image.open(entry.path) as opened:
            resized = resize_image(opened, image_size)
            try:
                for severity in severities:
                    key = (entry.image_id, int(severity.level))
                    if key in existing:
                        continue
                    severity_input = resized.copy()
                    changed: Image.Image | object | None = None
                    try:
                        changed = (
                            severity_input
                            if corruption is None
                            else corruption.apply(
                                severity_input,
                                severity.level,
                            )
                        )
                        if not isinstance(changed, Image.Image):
                            raise RuntimeError(
                                "corruption must return a PIL image"
                            )
                        if changed.size != expected_size:
                            raise RuntimeError(
                                "corruption changed image size: "
                                f"got {changed.size}, expected {expected_size}"
                            )
                        if changed.mode != "RGB":
                            raise RuntimeError(
                                "corruption must return an RGB image"
                            )
                        tensor = image_tensor(changed)
                    finally:
                        if (
                            isinstance(changed, Image.Image)
                            and changed is not severity_input
                        ):
                            changed.close()
                        severity_input.close()
                    yield key, tensor
            finally:
                if resized is not opened:
                    resized.close()


def extract_manifest(
    entries: tuple[ManifestEntry, ...],
    directory: str | Path,
    metadata: dict,
    extractor,
    corruption: Corruption | None,
    *,
    image_size: tuple[int, int],
    batch_size: int,
    shard_size: int,
) -> None:
    entries = _validated_entries(entries)
    metadata = _validated_metadata(metadata)
    image_size = _validated_image_size(image_size)
    batch_size = _positive_integer(batch_size, name="batch_size")
    shard_size = _positive_integer(shard_size, name="shard_size")
    severities = _validated_severities(corruption)
    if not callable(getattr(extractor, "extract_batch", None)):
        raise ValueError("extractor must provide an extract_batch method")
    expected_keys = _expected_keys(entries, severities)
    expected_set = set(expected_keys)
    expected_count = len(expected_keys)
    root = Path(directory)
    final = root / "manifest.json"
    if _entry_present(final):
        _validate_completed_extraction(entries, root, metadata, severities)
        return

    with ShardWriter(root, metadata, shard_size=shard_size) as writer:
        existing = writer.existing_keys()
        published_keys, record_contract = _published_record_state(writer)
        if set(published_keys) != existing:
            raise RuntimeError(
                "partial extraction record roster does not match its shards"
            )
        if published_keys != expected_keys[: len(published_keys)]:
            raise RuntimeError(
                "partial extraction record roster is not a canonical prefix"
            )
        unexpected = existing - expected_set
        if unexpected:
            raise RuntimeError(
                "partial extraction record roster contains unexpected keys: "
                f"{sorted(unexpected)!r}"
            )
        if writer.record_count != len(existing):
            raise RuntimeError(
                "partial extraction record count does not match its roster"
            )
        identities: list[tuple[str, int]] = []
        tensors: list[Tensor] = []

        def flush_batch() -> None:
            nonlocal record_contract
            batch_identities = list(identities)
            records = extractor.extract_batch(
                list(batch_identities),
                torch.stack(tensors),
            )
            records, record_contract = _validated_records(
                records,
                batch_identities,
                record_contract,
            )
            for record in records:
                writer.add(record)

        pending = _pending_samples(
            entries,
            corruption,
            severities,
            existing,
            image_size,
        )
        with closing(pending):
            for identity, tensor in pending:
                identities.append(identity)
                tensors.append(tensor)
                if len(tensors) == batch_size:
                    flush_batch()
                    identities, tensors = [], []
        if tensors:
            flush_batch()
        if writer.record_count != expected_count:
            raise RuntimeError(
                f"extraction wrote {writer.record_count} records; "
                f"expected {expected_count}"
            )


def record_from_outputs(
    image_id: str,
    severity: int,
    logits: Tensor,
    boxes: Tensor,
    persistence: Tensor,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "severity": int(severity),
        "logits": logits.detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        ),
        "boxes": boxes.detach().to(
            device="cpu",
            dtype=torch.float32,
            copy=True,
        ),
        "persistence": persistence.detach().to(
            device="cpu",
            dtype=torch.float16,
            copy=True,
        ),
    }


class RTDETRExtractor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        config: ExperimentConfig = FIXED_CONFIG,
    ) -> None:
        self.device = device
        self.config = config
        self.model = load_frozen_detector(checkpoint_path, device)
        self.capture = Layer2Capture(
            self.model.decoder,
            layer=config.persistence_layer,
        )

    @torch.inference_mode()
    def extract_batch(
        self,
        identities: list[tuple[str, int]],
        samples: Tensor,
    ) -> list[dict[str, object]]:
        if not isinstance(samples, Tensor):
            raise TypeError("samples must be a Tensor")
        if samples.ndim != 4:
            raise ValueError("samples must be a 4D NCHW tensor")
        if samples.shape[0] == 0:
            raise ValueError("sample batch must be nonempty")
        if samples.shape[1] != 3:
            raise ValueError("samples must have 3 channels")
        spatial_size = tuple(samples.shape[2:])
        if spatial_size != self.config.image_size:
            raise ValueError(
                f"sample spatial size {spatial_size} does not match "
                f"configured {self.config.image_size}"
            )
        if samples.dtype != torch.float32:
            raise ValueError(
                "samples must have dtype torch.float32, "
                f"got {samples.dtype}"
            )
        identity_count = len(identities)
        sample_count = samples.shape[0]
        if identity_count != sample_count:
            raise ValueError(
                f"received {identity_count} identities for sample batch "
                f"{sample_count}"
            )

        outputs = self.model(samples.to(self.device))
        if not isinstance(outputs, Mapping):
            raise TypeError("detector output must be a mapping")
        try:
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
        except KeyError as error:
            raise KeyError(
                "detector output must contain pred_logits and pred_boxes"
            ) from error
        if not isinstance(logits, Tensor) or not isinstance(boxes, Tensor):
            raise TypeError("detector logits and boxes must be tensors")
        if logits.ndim == 0 or boxes.ndim == 0:
            raise RuntimeError("detector logits and boxes need batch dimensions")
        output_count = logits.shape[0]
        if boxes.shape[0] != output_count:
            raise RuntimeError(
                "detector logits and boxes have different batch sizes: "
                f"{output_count} and {boxes.shape[0]}"
            )
        if output_count != identity_count:
            raise RuntimeError(
                f"detector output batch {output_count} does not match "
                f"identities {identity_count}"
            )
        expected_logits = (
            identity_count,
            self.config.query_count,
            self.config.class_count,
        )
        expected_boxes = (identity_count, self.config.query_count, 4)
        if (
            tuple(logits.shape) != expected_logits
            or tuple(boxes.shape) != expected_boxes
        ):
            raise RuntimeError(
                "unexpected detector output shapes "
                f"logits={tuple(logits.shape)}, boxes={tuple(boxes.shape)}; "
                f"expected {expected_logits} and {expected_boxes}"
            )

        features, weight = self.capture.take()
        if features.ndim != 3:
            raise RuntimeError(
                "captured decoder features must have shape "
                "(batch, queries, hidden)"
            )
        flat_features = features.reshape(-1, features.shape[-1])
        diagrams = batched_persistence(weight, flat_features).reshape(
            features.shape[0],
            features.shape[1],
            -1,
        )
        expected = (
            identity_count,
            self.config.query_count,
            self.config.persistence_dim,
        )
        if tuple(diagrams.shape) != expected:
            raise RuntimeError(
                f"unexpected persistence shape {tuple(diagrams.shape)}; "
                f"expected {expected}"
            )

        return [
            record_from_outputs(
                image_id,
                severity,
                item_logits,
                item_boxes,
                item_persistence,
            )
            for (image_id, severity), item_logits, item_boxes, item_persistence
            in zip(identities, logits, boxes, diagrams, strict=True)
        ]

    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> RTDETRExtractor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
