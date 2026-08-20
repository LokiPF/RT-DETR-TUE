"""One synthetic six-severity artifact pair, shared by every confidence-decile test.

Two test modules building their own tiny cache would be two chances for one of them to
drift into a shape the loader does not actually meet in production -- a record without
`layers`, a distance row keyed by str, a manifest whose `feature_cache_id` was never a
real content address. Everything here goes through `ShardWriter`, so the cache side is
written by the same code that wrote the pilot's, and its `artifact_id` is a real content
address rather than a literal.

`write_decile_artifacts` is the plan's builder with one change: the result manifest is sealed
with its real content address instead of the literal `"synthetic-results"`. A fixture carrying
a made-up id would be a fixture that cannot exercise the check which makes `source_result_id`
mean anything, and a validation rule the fixtures bypass is a validation rule that is not
tested. `mutate_decile_artifacts` keeps the plan's four mutations byte-identical and adds
more, one per validation rule the design names, so a rule that stops being enforced fails a
test instead of quietly widening what the loader accepts.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.scene_uncertainty.artifacts import ShardWriter, load_manifest, manifest_id
from src.scene_uncertainty.decile_analysis import result_content_address


LAYERS = (0, 1, 2)
QUERY_COUNT = 20
PERSISTENCE_DIM = 4


def write_decile_artifacts(root: Path, severities=range(6)) -> dict[str, Path]:
    cache = root / "cache"
    metadata = {
        "source_kind": "evaluation", "image_ids": [11],
        "checkpoint_sha256": "checkpoint", "decoder_layers": list(LAYERS),
        "query_count": QUERY_COUNT, "persistence_dim": PERSISTENCE_DIM,
    }
    with ShardWriter(cache, metadata, shard_size=20) as writer:
        for severity in severities:
            confidence = torch.linspace(0.05, 0.95, QUERY_COUNT).sub(severity * 0.005).clamp(0.002, 0.998)
            logits = torch.full((QUERY_COUNT, 80), -20.0)
            logits[:, 0] = torch.logit(confidence)
            logits[-2:, 0] = torch.logit(torch.tensor(0.002))
            boxes = torch.arange(QUERY_COUNT * 4, dtype=torch.float32).reshape(QUERY_COUNT, 4) / 100
            boxes[-1] = boxes[-2]
            logits[-1] = logits[-2]
            layers = {
                layer_id: (
                    torch.arange(QUERY_COUNT * PERSISTENCE_DIM, dtype=torch.float32)
                    .reshape(QUERY_COUNT, PERSISTENCE_DIM)
                    .add(layer_id + severity)
                )
                for layer_id in LAYERS
            }
            for values in layers.values():
                values[-1] = values[-2]
            writer.add({
                "image_id": 11, "severity": int(severity), "source_partition": "tuning",
                "boxes": boxes, "logits": logits.to(torch.float16),
                "layers": {layer: values.to(torch.float16) for layer, values in layers.items()},
            })
    results = root / "results" / "raw_k5.csv"
    results.parent.mkdir(parents=True)
    results.write_text("source result rows are not read by this command\n", encoding="utf-8")
    distance_path = results.with_suffix(".query_distances.pt")
    distance_rows = [
        {
            "image_id": 11, "severity": int(severity), "source_partition": "tuning",
            "query_scores_by_layer": {
                layer: torch.arange(QUERY_COUNT, dtype=torch.float16).mul(0.01).add(
                    severity * 0.02 + layer * 0.001
                )
                for layer in LAYERS
            },
        }
        for severity in severities
    ]
    torch.save(distance_rows, distance_path)
    normalizer_path = results.with_suffix(".normalizers.pt")
    torch.save({
        "layer_score_scales": {
            layer: {"center": torch.tensor(0.0), "scale": torch.tensor(1.0)}
            for layer in LAYERS
        }
    }, normalizer_path)
    manifest = {
        "artifact_type": "knn_scene_uncertainty_results",
        "feature_cache_id": load_manifest(cache)["artifact_id"],
        "source_partition": "tuning", "normalization": "raw", "k": 5,
        "query_distance_path": distance_path.name,
        "normalizer_path": normalizer_path.name,
    }
    manifest["artifact_id"] = result_content_address(manifest)
    results.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return {"cache": cache, "results": results}


def _edit_cache_records(cache: Path, transform) -> None:
    """Rewrite the single shard of a synthetic cache, leaving `record_count` alone.

    The manifest's `record_count` deliberately keeps its original value after a mutation
    that removes a record. A loader that cross-checked the streamed count against the
    manifest would fire on that mismatch first and mask the rule the mutation is aimed at,
    so the fixture is written to make such a shortcut visible rather than to accommodate it.
    """
    manifest = load_manifest(cache)
    shard = Path(cache) / manifest["shards"][0]
    records = torch.load(shard, map_location="cpu", weights_only=True)
    torch.save(transform(records), shard)


def _edit_cache_manifest(cache: Path, changes: dict, reseal: bool = True) -> None:
    """Change cache-manifest fields, optionally re-sealing the content address.

    `reseal=True` recomputes `artifact_id` so the manifest stays self-consistent and the
    edited *field* is what a loader must object to. `reseal=False` leaves the stale address
    in place, which is the only way to exercise the integrity check itself.
    """
    path = Path(cache) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(changes)
    if reseal:
        manifest["artifact_id"] = manifest_id(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _edit_result_manifest(results: Path, mutate) -> None:
    path = results.with_suffix(".manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _pad_from(record: dict, start: int) -> dict:
    """Copy the final query over every slot from `start`, making a long identical tail."""
    record["boxes"][start:] = record["boxes"][-1]
    record["logits"][start:] = record["logits"][-1]
    for values in record["layers"].values():
        values[start:] = values[-1]
    return record


def _truncate(record: dict, count: int) -> dict:
    record["boxes"] = record["boxes"][:count]
    record["logits"] = record["logits"][:count]
    record["layers"] = {layer: values[:count] for layer, values in record["layers"].items()}
    return record


def mutate_decile_artifacts(artifacts: dict[str, Path], mutation: str) -> None:
    results = artifacts["results"]
    cache = artifacts["cache"]
    distance_path = results.with_suffix(".query_distances.pt")
    rows = torch.load(distance_path, map_location="cpu", weights_only=True)
    if mutation == "drop_distance":
        rows.pop()
    elif mutation == "duplicate_distance":
        rows.append(dict(rows[0]))
    elif mutation == "wrong_query_count":
        rows[0]["query_scores_by_layer"][2] = rows[0]["query_scores_by_layer"][2][:-1]
    elif mutation == "drop_severity":
        rows = [row for row in rows if int(row["severity"]) != 5]
        manifest = load_manifest(artifacts["cache"])
        shard = artifacts["cache"] / manifest["shards"][0]
        records = torch.load(shard, map_location="cpu", weights_only=True)
        torch.save([record for record in records if int(record["severity"]) != 5], shard)
    # -- additions below; each one exercises a rule the design's validation list names -----
    elif mutation == "drop_severity_zero":
        rows = [row for row in rows if int(row["severity"]) != 0]
        _edit_cache_records(cache, lambda records: [r for r in records if int(r["severity"]) != 0])
    elif mutation == "duplicate_cache_record":
        _edit_cache_records(cache, lambda records: records + [dict(records[0])])
    elif mutation == "extra_distance_row":
        rows.append({**rows[0], "image_id": 12})
    elif mutation == "test_partition_row":
        rows[0] = {**rows[0], "source_partition": "test"}
    elif mutation == "no_tuning_records":
        _edit_cache_records(
            cache,
            lambda records: [{**record, "source_partition": "test"} for record in records],
        )
    elif mutation == "cache_layer_missing":
        def drop_layer(records):
            records[0]["layers"] = {
                layer: values for layer, values in records[0]["layers"].items() if layer != 2
            }
            return records
        _edit_cache_records(cache, drop_layer)
    elif mutation == "distance_layer_missing":
        rows[0]["query_scores_by_layer"] = {
            layer: values for layer, values in rows[0]["query_scores_by_layer"].items()
            if layer != 2
        }
    elif mutation == "scale_layer_missing":
        normalizer_path = results.with_suffix(".normalizers.pt")
        state = torch.load(normalizer_path, map_location="cpu", weights_only=True)
        state["layer_score_scales"] = {
            layer: value for layer, value in state["layer_score_scales"].items() if layer != 2
        }
        torch.save(state, normalizer_path)
    elif mutation == "cache_query_count":
        _edit_cache_records(cache, lambda records: [_truncate(records[0], QUERY_COUNT - 1)] + records[1:])
    elif mutation == "everything_padded":
        _edit_cache_records(cache, lambda records: [_pad_from(records[0], 3)] + records[1:])
    elif mutation == "stale_confidence_field":
        _edit_cache_records(
            cache,
            lambda records: [
                {**record, "confidence": torch.zeros(QUERY_COUNT)} for record in records
            ],
        )
    elif mutation == "missing_distance_file":
        distance_path.unlink()
        return
    elif mutation == "missing_normalizer_file":
        results.with_suffix(".normalizers.pt").unlink()
    elif mutation == "missing_result_manifest":
        results.with_suffix(".manifest.json").unlink()
    elif mutation == "missing_cache_manifest":
        (Path(cache) / "manifest.json").unlink()
    elif mutation == "drop_layer_score_scales":
        normalizer_path = results.with_suffix(".normalizers.pt")
        state = torch.load(normalizer_path, map_location="cpu", weights_only=True)
        state.pop("layer_score_scales")
        torch.save(state, normalizer_path)
    elif mutation == "drop_result_cache_id":
        _edit_result_manifest(results, lambda manifest: manifest.pop("feature_cache_id"))
    elif mutation == "drop_result_k":
        _edit_result_manifest(results, lambda manifest: manifest.pop("k"))
    elif mutation == "wrong_result_artifact_type":
        _edit_result_manifest(
            results, lambda manifest: manifest.update({"artifact_type": "scene_uncertainty_results"})
        )
    elif mutation == "forged_result_manifest":
        # The reviewer's pilot forgery in miniature: repoint `query_distance_path` at another
        # run's distances -- same `feature_cache_id`, different numbers -- and relabel `k` and
        # `normalization`. Every field the loader *compares* still agrees, so without the
        # result manifest's own content address this loads cleanly and mislabels every row.
        other_path = results.with_suffix(".other_distances.pt")
        torch.save(
            [
                {
                    **row,
                    "query_scores_by_layer": {
                        layer: values * 2.0
                        for layer, values in row["query_scores_by_layer"].items()
                    },
                }
                for row in rows
            ],
            other_path,
        )
        _edit_result_manifest(results, lambda manifest: manifest.update({
            "query_distance_path": other_path.name, "k": 99, "normalization": "hand_edited",
        }))
    elif mutation == "cache_field_lengths_disagree":
        def truncate_boxes(records):
            records[0]["boxes"] = records[0]["boxes"][:-1]
            return records
        _edit_cache_records(cache, truncate_boxes)
    elif mutation == "cache_source_kind":
        _edit_cache_manifest(cache, {"source_kind": "reference"}, reseal=True)
    elif mutation == "cache_manifest_forged":
        _edit_cache_manifest(cache, {"checkpoint_sha256": "forged"}, reseal=False)
    else:
        raise ValueError(f"unknown fixture mutation: {mutation}")
    torch.save(rows, distance_path)
