"""The paper's CPU forward-pass experiments (docs/paper-storyline.md, claims C1, C5 and C8).

    PAPER_CACHE=<dir> CUDA_VISIBLE_DEVICES= python scripts/paper/forward.py operations|untrained|filters [--images 200]

- operations: pure image operations at graded strengths, on 200 COCO val images, through the trained backbone.
  - Gain (x * g, also the training's brightness jitter), contrast and saturation (torchvision's definitions, as in the
    training's jitter) and hue, inside and outside the training range.
  - A white veil ((1 - t) x + t), an additive offset (x + b), the benchmark's HSV-value offset, Gaussian blur and
    additive Gaussian noise.
  - The benchmark's fog split into its parts. Fog is clip(g x + g c P) with g = max / (max + c) and a plasma map P,
    so its parts are the gain alone, the gain with a uniform veil of the same mean, the full fog, and the plasma's
    zero-mean structure without the gain.
  Each condition is scored with the stored reference, like the benchmark's corruptions.
- untrained: the same backbone with random weights, its batch-norm statistics re-estimated on 200 clean train images
  (the "reserved" split). Its own 2,000-image bank and 500 z-statistics images give its reference. The 200 images run
  clean and under the 19 families at severities 1, 3 and 5. The trained backbone's numbers on the same images and
  conditions come from the stored statistics.
- filters: the stem's first convolution, trained and untrained: its response to a uniform offset against the image's
  own structure, and how much of each filter is tuned to luminance rather than colour.
Per-image statistics are cached under PAPER_CACHE/forward/, so a stopped run resumes.
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (CACHE, CONDITIONS, FAMILIES, SCORED, STAGES, cached, reference, settings,  # noqa: E402
                    write_csv, write_json)
from degradation_monitor import corruptions  # noqa: E402
from degradation_monitor.detector.model import IMAGE_SIZE, load_frozen_detector, prepare_image  # noqa: E402
from degradation_monitor.detector.rtdetrv2.nn.backbone.presnet import PResNet  # noqa: E402
from degradation_monitor.detector.taps import EarlyChannelTaps  # noqa: E402
from degradation_monitor.evaluation.metrics import binary_auroc  # noqa: E402
from degradation_monitor.method.reference import fit_own_average  # noqa: E402
from degradation_monitor.method.scores import EPS, _global_columns, peak_share  # noqa: E402
from degradation_monitor.method.statistics import KEYS  # noqa: E402
from degradation_monitor.stages.common import bounded  # noqa: E402
from degradation_monitor.stages.method import batch_statistics, image_statistics  # noqa: E402
from stored import columns, find_neighbours, standardised  # noqa: E402

FORWARD = CACHE / "forward"
SEED = 44
BATCH = 16
TRAINING_RANGE = {"gain": (0.875, 1.125), "contrast": (0.5, 1.5), "saturation": (0.5, 1.5), "hue": (0.0, 0.05)}
STRENGTHS = {
    "gain": (0.3, 0.4, 0.5, 0.6, 0.75, 0.875, 1.125, 1.25, 1.5, 2.0),
    "contrast": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.5, 2.0),
    "saturation": (0.0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0, 5.0),
    "hue": (0.02, 0.05, 0.1, 0.2, 0.3, 0.5),
    "veil": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    "offset": (0.05, 0.1, 0.2, 0.3, 0.5),
    "hsv_value": (0.05, 0.1, 0.2, 0.3, 0.5),
    "blur": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    "noise": (0.01, 0.02, 0.04, 0.08, 0.12, 0.18),
}
FOG_C = ((1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4))  # imagecorruptions' fog, severities 1-5
FOG_PARTS = ("fog_gain", "fog_gain_uniform_veil", "fog_full", "fog_plasma_structure")
UNTRAINED_SEVERITIES = (1, 3, 5)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluation_positions(count: int) -> np.ndarray:
    return np.sort(np.random.default_rng(SEED).choice(5000, count, replace=False))


# --- the operations, on float RGB in [0, 1] ------------------------------------------------------------------------

def _gray(x):
    return 0.2989 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]  # torchvision's rgb_to_grayscale


def _hsv(x):
    from skimage.color import rgb2hsv
    return rgb2hsv(x)


def _rgb(x):
    from skimage.color import hsv2rgb
    return hsv2rgb(x)


def operate(x: np.ndarray, operation: str, strength: float, rng) -> np.ndarray:
    if operation == "gain":
        return x * strength
    if operation == "contrast":
        return strength * x + (1 - strength) * _gray(x).mean()
    if operation == "saturation":
        return strength * x + (1 - strength) * _gray(x)[..., None]
    if operation == "hue":
        hsv = _hsv(x)
        hsv[..., 0] = (hsv[..., 0] + strength) % 1.0
        return _rgb(hsv)
    if operation == "veil":
        return (1 - strength) * x + strength
    if operation == "offset":
        return x + strength
    if operation == "hsv_value":
        hsv = _hsv(x)
        hsv[..., 2] = np.clip(hsv[..., 2] + strength, 0, 1)
        return _rgb(hsv)
    if operation == "blur":
        from skimage.filters import gaussian
        return gaussian(x, sigma=strength, channel_axis=-1)
    if operation == "noise":
        return x + rng.normal(scale=strength, size=x.shape)
    raise ValueError(operation)


def fog_parts(x: np.ndarray, image_id: str, severity: int) -> dict:
    """The benchmark's fog of this image and severity, with its plasma drawn from the same seed, and its parts."""
    from imagecorruptions.corruptions import next_power_of_2, plasma_fractal

    corruptions._prepare_compatibility()
    c = FOG_C[severity - 1]
    saved = np.random.get_state()
    try:
        np.random.seed(corruptions.variant_seed(image_id, "fog", severity))
        plasma = plasma_fractal(mapsize=next_power_of_2(int(max(x.shape))), wibbledecay=c[1])[:x.shape[0], :x.shape[1]]
    finally:
        np.random.set_state(saved)
    gain = x.max() / (x.max() + c[0])
    veil = gain * c[0] * plasma[..., None]
    return {"fog_gain": gain * x, "fog_gain_uniform_veil": gain * x + veil.mean(),
            "fog_full": (x + c[0] * plasma[..., None]) * x.max() / (x.max() + c[0]),  # the benchmark's arithmetic
            "fog_plasma_structure": x + veil - veil.mean()}


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(x, 0, 1) * 255)  # truncation, as imagecorruptions returns its images


def operation_conditions() -> list[tuple[str, float]]:
    out = [("clean", 0.0)]
    for operation, strengths in STRENGTHS.items():
        out += [(operation, float(s)) for s in strengths]
    for part in FOG_PARTS:
        out += [(part, float(s)) for s in range(1, 6)]
    return out


def operation_variants(path) -> tuple[str, list[np.ndarray]]:
    """Worker: the image under every operation condition, in operation_conditions() order."""
    path = Path(path)
    with Image.open(path) as source:
        pixels = np.asarray(source.convert("RGB"), dtype=np.uint8)
    x = pixels.astype(np.float64) / 255.0
    out = []
    parts = {}
    for operation, strength in operation_conditions():
        if operation == "clean":
            out.append(pixels.copy())
        elif operation.startswith("fog_"):
            severity = int(strength)
            if severity not in parts:
                parts[severity] = fog_parts(x, path.name, severity)
            out.append(to_uint8(parts[severity][operation]))
        else:
            rng = np.random.default_rng(corruptions.variant_seed(path.name, operation, int(round(strength * 1000))))
            out.append(to_uint8(operate(x, operation, strength, rng)))
    return path.name, out


def untrained_conditions() -> list[tuple[str, int]]:
    return [("clean", 0)] + [(family, severity) for family in FAMILIES for severity in UNTRAINED_SEVERITIES]


def benchmark_variants(path) -> tuple[str, list[np.ndarray]]:
    """Worker: the image clean and under the benchmark's corruptions at severities 1, 3 and 5 (the stored seeds)."""
    path = Path(path)
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    out = [np.asarray(rgb, dtype=np.uint8).copy()]
    for family, severity in untrained_conditions()[1:]:
        out.append(np.asarray(corruptions.corrupt(rgb, path.name, family, severity), dtype=np.uint8).copy())
    return path.name, out


# --- running a backbone over the images ------------------------------------------------------------------------------

def run(taps, paths, make_variants, folder: Path, workers: int) -> dict:
    """KEYS -> (images, conditions, C) for every image, cached image by image under `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    pending = [p for p in paths if not (folder / f"{Path(p).stem}.npz").exists()]
    started = time.time()
    if pending:
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers) as pool:
            for done, (name, arrays) in enumerate(bounded(pool, make_variants, pending, 2 * workers), start=1):
                statistics = image_statistics(taps, arrays, BATCH)
                np.savez(folder / f"{Path(name).stem}.tmp.npz", **statistics)
                (folder / f"{Path(name).stem}.tmp.npz").replace(folder / f"{Path(name).stem}.npz")
                if done % 10 == 0 or done == len(pending):
                    rate = (time.time() - started) / done
                    log(f"{folder.name}: {done}/{len(pending)} images, {rate:.1f} s per image, "
                        f"{rate * (len(pending) - done) / 60:.0f} min left")
    out = {key: [] for key in KEYS}
    for p in paths:
        with np.load(folder / f"{Path(p).stem}.npz") as item:
            for key in KEYS:
                out[key].append(item[key])
    return {key: np.stack(values) for key, values in out.items()}


def clean_statistics(taps, paths, path: Path) -> dict:
    """KEYS -> (images, C) of clean images, cached in one file."""
    if path.exists():
        with np.load(path) as data:
            return {k: data[k] for k in KEYS}
    parts = []
    started = time.time()
    for start in range(0, len(paths), BATCH):
        batch = []
        for p in paths[start:start + BATCH]:
            with Image.open(p) as source:
                batch.append(prepare_image(source.convert("RGB"), IMAGE_SIZE))
        parts.append(batch_statistics(taps, torch.stack(batch)))
        if (start // BATCH) % 25 == 0:
            log(f"{path.stem}: {start + len(batch)}/{len(paths)} images in {time.time() - started:.0f} s")
    out = {k: np.concatenate([part[k] for part in parts]) for k in KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


# --- scoring ---------------------------------------------------------------------------------------------------------

def score_rows(test: dict, bank: dict, zstats: dict) -> dict:
    """Our rows (images, conditions) against a reference: two-axis, both arms, peak share, global level by stage."""
    from degradation_monitor.evaluation.metrics import stage_zstats, zscored_sum

    images, conditions = test["means_s1"].shape[:2]
    test_columns = columns(test, bank, find_neighbours(test, bank))
    clean_columns = columns(zstats, bank, find_neighbours(zstats, bank))
    shape = (images, conditions)
    flatter = standardised(test_columns["flatter"], clean_columns["flatter"]).reshape(shape)
    level = standardised(test_columns["level"], clean_columns["level"]).reshape(shape)
    mean, std = stage_zstats(clean_columns["shape"])
    out = {"two_axis": np.maximum(flatter, level), "flatter_arm": flatter, "level_arm": level,
           "peak_share": zscored_sum(test_columns["shape"], mean, std).reshape(shape)}
    clean_global = _global_columns(zstats, bank, SCORED)
    mean, std = stage_zstats(clean_global)
    out["global_level"] = zscored_sum(_global_columns(test, bank, SCORED), mean, std).reshape(shape)
    for layer_index, layer in enumerate(STAGES):
        out[f"global_level_{layer}"] = _global_columns(test, bank, (layer,)).reshape(shape)
        centre, spread = fit_own_average(peak_share(bank, layer))
        deviation = (peak_share(test, layer) - centre) / spread
        out[f"global_share_{layer}"] = np.abs(deviation).mean(axis=1).reshape(shape)
        out[f"global_flatter_{layer}"] = (-deviation).mean(axis=1).reshape(shape)
    return out


def changes(test: dict, condition: int, layer: str) -> dict:
    """How the condition moves the stage's channels against the same images clean."""
    means = np.asarray(test[f"means_{layer}"], dtype=np.float64)
    top = np.asarray(test[f"top_{layer}"], dtype=np.float64)
    d_level = np.log(means[:, condition] + EPS) - np.log(means[:, 0] + EPS)
    d_top = np.log(top[:, condition] + EPS) - np.log(top[:, 0] + EPS)
    relative = (means[:, condition] - means[:, 0]) / (means[:, 0] + EPS)
    return {f"d_log_level_{layer}": float(d_level.mean()), f"d_log_top_{layer}": float(d_top.mean()),
            f"d_peak_share_{layer}": float((d_top - d_level).mean()),
            f"channels_flatter_{layer}": float((d_top - d_level < 0).mean()),
            f"total_level_ratio_{layer}": float(np.median(means[:, condition].sum(1) / means[:, 0].sum(1))),
            f"total_top_ratio_{layer}": float(np.median(top[:, condition].sum(1) / top[:, 0].sum(1))),
            f"channels_up_10pct_{layer}": float((relative > 0.1).mean()),
            f"channels_down_10pct_{layer}": float((relative < -0.1).mean())}


# --- the experiments -------------------------------------------------------------------------------------------------

def operations(count: int, workers: int) -> None:
    paths = [settings().dataset.evaluation_images()[i] for i in evaluation_positions(count)]
    conditions = operation_conditions()
    detector = load_frozen_detector(settings().checkpoint, torch.device("cpu"))
    with EarlyChannelTaps(detector.backbone) as taps:
        test = run(taps, paths, operation_variants, FORWARD / "operations", workers)
    bank, zstats = reference()
    # The clean and full-fog conditions also exist in the stored (GPU) statistics: the CPU pass must match them.
    stored = cached("method")
    positions = evaluation_positions(count)
    fog = {s: CONDITIONS.index(("fog", s)) for s in range(1, 6)}
    check = {}
    for key in KEYS:
        gpu_clean = np.asarray(stored[key][positions, 0], dtype=np.float64)
        check[key] = float(np.max(np.abs(test[key][:, 0] - gpu_clean) / (np.abs(gpu_clean) + 1e-3)))
        for s, condition in fog.items():
            gpu = np.asarray(stored[key][positions, condition], dtype=np.float64)
            ours = test[key][:, conditions.index(("fog_full", float(s)))]
            check[f"{key}_fog{s}"] = float(np.max(np.abs(ours - gpu) / (np.abs(gpu) + 1e-3)))
    log("operations: largest relative difference to the stored GPU statistics: "
        f"clean {max(v for k, v in check.items() if 'fog' not in k):.2e}, fog {max(v for k, v in check.items() if 'fog' in k):.2e}")
    rows = score_rows(test, bank, zstats)
    table = []
    for index, (operation, strength) in enumerate(conditions):
        if operation == "clean":
            continue
        row = {"operation": operation, "strength": strength}
        if operation in TRAINING_RANGE:
            low, high = TRAINING_RANGE[operation]
            row["inside_training_range"] = bool(low <= strength <= high) if operation != "hue" else strength <= high
        else:
            row["inside_training_range"] = ""
        for name in ("two_axis", "level_arm", "flatter_arm", "peak_share", "global_level"):
            row[f"auroc_{name}"] = binary_auroc(rows[name][:, 0], rows[name][:, index])
        row["flatter_arm_larger"] = float((rows["flatter_arm"][:, index] > rows["level_arm"][:, index]).mean())
        for layer in STAGES:
            row.update(changes(test, index, layer))
        table.append(row)
        log(f"operations: {operation:<22} {strength:>6.3f}  two-axis {row['auroc_two_axis']:.3f}  level "
            f"{row['auroc_level_arm']:.3f}  flatter {row['auroc_flatter_arm']:.3f}  s1 dlog level "
            f"{row['d_log_level_s1']:+.3f} dlog top {row['d_log_top_s1']:+.3f} flatter {row['channels_flatter_s1']:.2f}"
            f"  total x{row['total_level_ratio_s1']:.3f} top x{row['total_top_ratio_s1']:.3f}  up/down "
            f"{row['channels_up_10pct_s1']:.2f}/{row['channels_down_10pct_s1']:.2f}")
    write_csv("dose_response", table)
    write_json("dose_response_check", {"images": count, "max_relative_difference_to_stored": check})


def untrained_backbone(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return PResNet(depth=18, variant="d", num_stages=4, return_idx=[1, 2, 3], act="relu", freeze_at=-1,
                   freeze_norm=False, pretrained=False)


@torch.no_grad()
def recalibrate(backbone: torch.nn.Module, paths) -> None:
    """Re-estimate every batch-norm's running statistics on clean images (cumulative average), then freeze."""
    for module in backbone.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None
    backbone.train()
    for start in range(0, len(paths), BATCH):
        batch = []
        for p in paths[start:start + BATCH]:
            with Image.open(p) as source:
                batch.append(prepare_image(source.convert("RGB"), IMAGE_SIZE))
        backbone(torch.stack(batch))
    backbone.eval().requires_grad_(False)


def _summaries(rows: dict, conditions: list) -> dict:
    """AUROC per condition against clean, and group means (common, extra, severity 1) over the 57 conditions."""
    from degradation_monitor.corruptions import COMMON_FAMILIES

    out = {}
    for name, values in rows.items():
        aurocs = np.array([binary_auroc(values[:, 0], values[:, i]) for i in range(1, len(conditions))])
        families = np.array([f for f, _ in conditions[1:]])
        severities = np.array([s for _, s in conditions[1:]])
        common = np.isin(families, COMMON_FAMILIES)
        out[name] = {"common": float(aurocs[common].mean()), "extra": float(aurocs[~common].mean()),
                     "common_s1": float(aurocs[common & (severities == 1)].mean()),
                     "fog_s1": float(aurocs[(families == "fog") & (severities == 1)][0]),
                     "gaussian_noise_s1": float(aurocs[(families == "gaussian_noise") & (severities == 1)][0]),
                     "per_condition": aurocs.tolist()}
    return out


def untrained(count: int, workers: int) -> None:
    dataset = settings().dataset
    positions = evaluation_positions(count)
    paths = [dataset.evaluation_images()[i] for i in positions]
    conditions = untrained_conditions()
    backbone = untrained_backbone()
    state = FORWARD / "untrained" / "backbone.pt"
    if state.exists():
        backbone.load_state_dict(torch.load(state, weights_only=True))
        backbone.eval().requires_grad_(False)
    else:
        log("untrained: re-estimating batch-norm statistics on 200 clean train images")
        recalibrate(backbone, dataset.reference_split("reserved"))
        state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state_dict(), state)
    with EarlyChannelTaps(backbone) as taps:
        bank = clean_statistics(taps, dataset.reference_split("bank"), FORWARD / "untrained" / "bank.npz")
        zstats = clean_statistics(taps, dataset.reference_split("zstats"), FORWARD / "untrained" / "zstats.npz")
        test = run(taps, paths, benchmark_variants, FORWARD / "untrained" / "images", workers)
    stored = cached("method")
    stored_conditions = [CONDITIONS.index(c) for c in conditions]
    trained_test = {k: np.asarray(stored[k][positions][:, stored_conditions]) for k in KEYS}
    trained_bank, trained_zstats = reference()
    results = {}
    for label, (t, b, z) in {"trained": (trained_test, trained_bank, trained_zstats),
                             "untrained": (test, bank, zstats)}.items():
        results[label] = _summaries(score_rows(t, b, z), conditions)
        # what the key carries: the neighbours' share of the clean variance, on the 500 z-statistics images
        neighbours = find_neighbours(z, b)
        explained = {}
        for layer in SCORED:
            values = np.asarray(z[f"means_{layer}"], dtype=np.float64)
            reference_values = np.asarray(b[f"means_{layer}"], dtype=np.float64)
            centre, spread = fit_own_average(reference_values)
            around_neighbours = ((values - reference_values[neighbours].mean(axis=1)) / spread) ** 2
            around_mean = ((values - centre) / spread) ** 2
            explained[layer] = float(1 - around_neighbours.mean() / around_mean.mean())
        results[label]["neighbours_explain"] = explained
        kept = []
        test_neighbours = find_neighbours(t, b).reshape(count, len(conditions), -1)
        for i in range(1, len(conditions)):
            kept.append(np.mean([len(set(test_neighbours[j, i]) & set(test_neighbours[j, 0])) / test_neighbours.shape[2]
                                 for j in range(count)]))
        results[label]["neighbours_kept_by_severity"] = {
            s: float(np.mean([k for k, (_, sev) in zip(kept, conditions[1:]) if sev == s])) for s in UNTRAINED_SEVERITIES}
    table = []
    for name in results["trained"]:
        if name in ("neighbours_explain", "neighbours_kept_by_severity"):
            continue
        for label in ("trained", "untrained"):
            r = results[label][name]
            table.append({"score": name, "backbone": label, **{k: r[k] for k in ("common", "extra", "common_s1",
                                                                                    "fog_s1", "gaussian_noise_s1")}})
            log(f"untrained: {name:<20} {label:<9} {r['common']:.3f} / {r['extra']:.3f}  sev1 {r['common_s1']:.3f}"
                f"  fog1 {r['fog_s1']:.3f}  noise1 {r['gaussian_noise_s1']:.3f}")
    for label in ("trained", "untrained"):
        log(f"untrained: {label} neighbours explain {results[label]['neighbours_explain']}; kept "
            f"{results[label]['neighbours_kept_by_severity']}")
    write_csv("untrained_backbone", table)
    write_json("untrained_backbone", {label: {k: v for k, v in r.items()} for label, r in results.items()})


@torch.no_grad()
def filters(count: int, workers: int) -> None:
    """What the first convolution can see (C8): its response to a uniform offset and its colour tuning.

    For each of the 32 filters of the stem's first convolution, on `count` clean val images:
    - offset: the shift of its output under a uniform input offset of 0.1 (brightness at severity 1 adds 0.1 to the
      HSV value), in units of the output's spatial spread on the clean image. A gain g changes the same spread by
      1 - g (0.6 for fog at severity 1), so this ratio says how an offset compares with a loss of contrast;
    - luminance share: the share of the filter's energy along R = G = B at each tap (1/3 for a random filter).
    The trained filters are compared with the untrained backbone's.
    """
    paths = [settings().dataset.evaluation_images()[i] for i in evaluation_positions(count)]
    batch = torch.stack([prepare_image(Image.open(p).convert("RGB"), IMAGE_SIZE) for p in paths])
    rows = []
    for label, backbone in (("trained", load_frozen_detector(settings().checkpoint, torch.device("cpu")).backbone),
                            ("untrained", untrained_backbone())):
        conv = backbone.conv1.conv1_1.conv
        weight = conv.weight.double()  # (32, 3, 3, 3)
        output = torch.nn.functional.conv2d(batch.double(), weight, stride=conv.stride, padding=conv.padding)
        spread = output.std(dim=(2, 3)).mean(dim=0)  # per filter, the spatial spread averaged over images
        offset = 0.1 * weight.sum(dim=(1, 2, 3)).abs() / spread
        luminance = (weight.sum(dim=1) ** 2 / 3).sum(dim=(1, 2)) / (weight ** 2).sum(dim=(1, 2, 3))
        for index in range(weight.shape[0]):
            rows.append({"backbone": label, "filter": index, "offset_over_spread": float(offset[index]),
                         "luminance_share": float(luminance[index])})
        log(f"filters: {label}: offset 0.1 / spread median {offset.median():.3f} [{offset.quantile(0.25):.3f}, "
            f"{offset.quantile(0.75):.3f}], max {offset.max():.3f}; luminance share median {luminance.median():.3f} "
            f"[{luminance.quantile(0.25):.3f}, {luminance.quantile(0.75):.3f}], filters above 0.8: "
            f"{int((luminance > 0.8).sum())} of {len(luminance)}")
    write_csv("first_conv_filters", rows)


def untrained_key(count: int, workers: int) -> None:
    """Does the untrained backbone's stage-4 key also find similar scenes? COCO panoptic label overlap with the 50
    neighbours, against 50 random bank images, for both backbones on the same clean images (run after untrained)."""
    from stored import _panoptic

    dataset = settings().dataset
    positions = evaluation_positions(count)
    paths = [dataset.evaluation_images()[i] for i in positions]
    annotations = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations")
    val, _ = _panoptic(annotations / "panoptic_val2017.json")
    train, _ = _panoptic(annotations / "panoptic_train2017.json")
    empty = {"things": set(), "stuff": set(), "count": 0}
    labels = [val.get(int(p.stem), empty) for p in paths]
    bank_labels = [train.get(int(p.stem), empty) for p in dataset.reference_split("bank")]
    stored = cached("method")
    trained_bank, _ = reference()
    with np.load(FORWARD / "untrained" / "bank.npz") as data:
        untrained_bank = {k: data[k] for k in KEYS}
    untrained_clean = {"means_s4": np.stack([np.load(FORWARD / "untrained" / "images" / f"{p.stem}.npz")["means_s4"][0]
                                             for p in paths])}
    trained_clean = {"means_s4": np.asarray(stored["means_s4"][positions, 0])}
    random_rows = np.random.default_rng(SEED).integers(0, len(bank_labels), size=(count, 50))

    def jaccard(a, b):
        union = a["things"] | a["stuff"] | b["things"] | b["stuff"]
        return len((a["things"] | a["stuff"]) & (b["things"] | b["stuff"])) / len(union) if union else np.nan

    out = {}
    for label, (values, bank) in {"trained": (trained_clean, trained_bank),
                                  "untrained": (untrained_clean, untrained_bank)}.items():
        neighbours = find_neighbours(values, bank)
        out[label] = float(np.nanmean([np.nanmean([jaccard(labels[i], bank_labels[j]) for j in neighbours[i]])
                                       for i in range(count)]))
    out["random"] = float(np.nanmean([np.nanmean([jaccard(labels[i], bank_labels[j]) for j in random_rows[i]])
                                      for i in range(count)]))
    log(f"untrained key: label Jaccard with the 50 neighbours: trained {out['trained']:.3f}, untrained "
        f"{out['untrained']:.3f}, random bank images {out['random']:.3f}")
    write_json("untrained_key_categories", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=("operations", "untrained", "filters", "untrained_key"))
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    experiments = {"operations": operations, "untrained": untrained, "filters": filters,
                   "untrained_key": untrained_key}
    experiments[args.experiment](args.images, args.workers)


if __name__ == "__main__":
    main()
