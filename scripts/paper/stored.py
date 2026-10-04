"""The paper's evidence from the stored statistics: CPU only, no forward pass (docs/paper-storyline.md, C1-C8).

    PAPER_CACHE=<dir> python scripts/paper/stored.py [analysis ...]

Analyses, all on the 5,000 COCO val images under the 96 conditions (default: all, in this order):
- check: our reimplementation of the scores, given the neighbours, against the package's two-axis score;
- depth: AUROC of each stage's score, ours and the baselines' (claim C1, Figure 3);
- stability: how many of the 50 neighbours survive a corruption, how far the key moves against the neighbourhood's
  radius, and how far the neighbours' mean moves against the image itself (C1);
- categories: COCO panoptic thing and stuff overlap between an image and its neighbours, against random bank
  images (C1);
- spread: the clean spread around the neighbours' mean against the spread around the bank's mean, and the shifts in
  both units (C3, C4);
- flatten: the change in log level, log top-1% and peak share, the share of flattening channels, and the share of
  the shift along the all-channels direction, per family, severity and stage (C5, Figure 1);
- arms: which arm is larger, each arm's AUROC, and the larger arm against the sum, with intervals (C6);
- twin: the scores with the clean twin's neighbours, which removes the key's own drift (C4, C8);
- severity: the ladder's gains by severity, with intervals (C4);
- ablation: k from 1 to 500, and bank sizes from 100 to 2,000 (C4);
- linear: a ridge map from the key instead of the neighbours, and how far its prediction follows the corruption (C4);
- statistic: the second axis with the raw ratio, the log top-1% mean or the top-1% mean instead of the peak share (C5);
- threshold: a threshold set on the 500 clean z-statistics train images, applied to the val images (deployment view);
- opposite: whether dense channels flatten while sparse ones sharpen, per channel at severity 1 (a claim to check).
"""
from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (CACHE, COMMON, CONDITIONS, EXTRA, FAMILIES, FAMILY, ROOT, SCORED, SEVERITY, STAGES,  # noqa: E402
                    cached, image_sets, per_condition, reference, settings, statistics, summarise, write_csv,
                    write_json)
from degradation_monitor.evaluation.metrics import (bootstrap, condition_aurocs, group_separation,  # noqa: E402
                                                    stage_zstats, zscored_sum)
from degradation_monitor.method import scores as method_scores  # noqa: E402
from degradation_monitor.method.reference import CHUNK, NEIGHBOURS, STD_FLOOR, fit_own_average, nearest_rows  # noqa: E402
from degradation_monitor.method.scores import EPS, peak_share  # noqa: E402

DERIVED = CACHE / "derived"
IMAGES, VARIANTS = 5000, len(CONDITIONS)
SEED = 44
WORKERS = 12
NOTE_FAMILIES = ("fog", "contrast", "zoom_blur", "gaussian_noise", "brightness", "saturate", "spatter", "frost")


def flat(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values.reshape(-1, values.shape[-1])


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --- the method's pieces, given the neighbours -------------------------------------------------------------------

def key_space(bank: dict, key: str = "s4"):
    centre, spread = fit_own_average(bank[f"means_{key}"])
    return centre, spread, (np.asarray(bank[f"means_{key}"], dtype=np.float64) - centre) / spread


def find_neighbours(values: dict, bank: dict, k: int = NEIGHBOURS) -> np.ndarray:
    """Each row's k nearest bank rows in the standardised key, exactly as method.scores._columns finds them."""
    centre, spread, bank_keys = key_space(bank)
    queries = flat(values["means_s4"])
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), CHUNK):
        rows = slice(start, start + CHUNK)
        out[rows] = nearest_rows((queries[rows] - centre) / spread, bank_keys, k)
    return out


def sorted_neighbours(values: dict, bank: dict, k: int) -> np.ndarray:
    """Each row's k nearest bank rows, nearest first."""
    centre, spread, bank_keys = key_space(bank)
    queries = flat(values["means_s4"])
    bank_sq = (bank_keys ** 2).sum(axis=1)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), CHUNK):
        rows = (queries[start:start + CHUNK] - centre) / spread
        distances = (rows ** 2).sum(axis=1)[:, None] + bank_sq[None] - 2 * rows @ bank_keys.T
        part = np.argpartition(distances, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(distances, part, axis=1), axis=1)
        out[start:start + CHUNK] = np.take_along_axis(part, order, axis=1)
    return out


def columns(values: dict, bank: dict, neighbours: np.ndarray, scored=SCORED, extras: bool = False,
            share=peak_share) -> dict:
    """Per-stage level, shape and flatter columns (rows, stages) given the neighbours; as method.scores._columns.

    With extras, also the channel average of the neighbours' mean and of the row itself, in bank-spread units, for
    the level and the peak share (nb_level, img_level, nb_share, img_share).
    """
    n = len(neighbours)
    kinds = ("level", "shape", "flatter") + (("nb_level", "img_level", "nb_share", "img_share") if extras else ())
    out = {kind: np.empty((n, len(scored))) for kind in kinds}
    for column, layer in enumerate(scored):
        level_reference = np.asarray(bank[f"means_{layer}"], dtype=np.float64)
        level_spread = fit_own_average(level_reference)[1]
        level_values = flat(values[f"means_{layer}"])
        share_reference = share(bank, layer)
        share_spread = fit_own_average(share_reference)[1]
        share_values = share(values, layer)
        for start in range(0, n, CHUNK):
            rows = slice(start, start + CHUNK)
            level_mean = level_reference[neighbours[rows]].mean(axis=1)
            out["level"][rows, column] = (np.abs(level_values[rows] - level_mean) / level_spread).mean(axis=1)
            share_mean = share_reference[neighbours[rows]].mean(axis=1)
            deviation = (share_values[rows] - share_mean) / share_spread
            out["shape"][rows, column] = np.abs(deviation).mean(axis=1)
            out["flatter"][rows, column] = (-deviation).mean(axis=1)
            if extras:
                out["nb_level"][rows, column] = (level_mean / level_spread).mean(axis=1)
                out["img_level"][rows, column] = (level_values[rows] / level_spread).mean(axis=1)
                out["nb_share"][rows, column] = (share_mean / share_spread).mean(axis=1)
                out["img_share"][rows, column] = (share_values[rows] / share_spread).mean(axis=1)
    return out


def standardised(test: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """The z-scored sum over stages, re-standardised on the clean z-statistics images: one arm of the two-axis score."""
    mean, std = stage_zstats(clean)
    reference_sum = zscored_sum(clean, mean, std)
    return (zscored_sum(test, mean, std) - reference_sum.mean()) / reference_sum.std()


def rows_of(test_columns: dict, clean_columns: dict) -> dict:
    """Our rows (images, 96) from the columns: two-axis, both arms, their sum, level and peak share."""
    shape = (IMAGES, VARIANTS)
    flatter = standardised(test_columns["flatter"], clean_columns["flatter"]).reshape(shape)
    level = standardised(test_columns["level"], clean_columns["level"]).reshape(shape)
    mean, std = stage_zstats(clean_columns["shape"])
    return {"two_axis": np.maximum(flatter, level), "flatter_arm": flatter, "level_arm": level,
            "sum_of_arms": flatter + level, "peak_share": zscored_sum(test_columns["shape"], mean, std).reshape(shape)}


# --- shared, cached inputs ---------------------------------------------------------------------------------------

class Context:
    def __init__(self):
        self._stats = None
        self.bank, self.zstats = reference()

    @property
    def stats(self) -> dict:
        if self._stats is None:
            log("loading the stored statistics")
            self._stats = statistics()
        return self._stats

    def derived(self, name: str, compute):
        path = DERIVED / f"{name}.npy"
        if path.exists():
            return np.load(path)
        DERIVED.mkdir(parents=True, exist_ok=True)
        started = time.time()
        value = compute()
        np.save(path.with_suffix(".tmp.npy"), value)
        path.with_suffix(".tmp.npy").replace(path)
        log(f"computed {name} in {time.time() - started:.0f} s")
        return value

    def neighbours(self) -> np.ndarray:
        return self.derived("neighbours_k50", lambda: find_neighbours(self.stats, self.bank).astype(np.int32))

    def zstats_neighbours(self) -> np.ndarray:
        return find_neighbours(self.zstats, self.bank)

    def own_columns(self) -> dict:
        names = ("level", "shape", "flatter", "nb_level", "img_level", "nb_share", "img_share")
        if all((DERIVED / f"own_{n}.npy").exists() for n in names):
            return {n: np.load(DERIVED / f"own_{n}.npy") for n in names}
        out = columns(self.stats, self.bank, self.neighbours().astype(np.int64), extras=True)
        for name, value in out.items():
            np.save(DERIVED / f"own_{name}.npy", value)
        return out

    def clean_columns(self) -> dict:
        return columns(self.zstats, self.bank, self.zstats_neighbours())

    def rows(self) -> dict:
        return rows_of(self.own_columns(), self.clean_columns())


def table_rows(aurocs: dict, label_fields: dict) -> list[dict]:
    """One row per score: its group summaries and its severity-1 AUROC on the noted families."""
    out = []
    for name, values in aurocs.items():
        row = {**label_fields(name), **summarise(values)}
        for family in NOTE_FAMILIES:
            row[f"{family}_s1"] = float(values[(FAMILY == family) & (SEVERITY == 1)][0])
        out.append(row)
    return out


def family_rows(aurocs: dict) -> list[dict]:
    """Per family and severity, one column per score."""
    out = []
    for condition in range(1, VARIANTS):
        family, severity = CONDITIONS[condition]
        out.append({"family": family, "severity": severity, "group": "common" if COMMON[condition] else "extra",
                    **{name: float(values[condition]) for name, values in aurocs.items()}})
    return out


# --- analyses ----------------------------------------------------------------------------------------------------

def check(ctx: Context) -> dict:
    """Our columns reproduce the package's two-axis score on the stored statistics (all 5,000 images)."""
    ours = ctx.rows()["two_axis"]
    package, _ = method_scores.two_axis_scores(ctx.stats, ctx.bank, ctx.zstats)
    difference = float(np.abs(ours - package).max())
    summary = summarise(per_condition(ours))
    log(f"check: max |ours - package| = {difference:.2e}; two-axis {summary['common']:.4f} / {summary['extra']:.4f}")
    if difference > 1e-9:
        raise SystemExit("the reimplementation does not match the package's two-axis score")
    return {"max_abs_difference": difference, **summary}


def depth(ctx: Context) -> dict:
    """AUROC of every stage's own score: ours against the global and the similar-scene reference, and the baselines'."""
    stats, bank = ctx.stats, ctx.bank
    shape = (IMAGES, VARIANTS)
    columns_of = {}
    global_level = method_scores._global_columns(stats, bank, STAGES).reshape(*shape, len(STAGES))
    for index, layer in enumerate(STAGES):
        columns_of[("level vs bank mean", layer)] = global_level[..., index]
        mean, spread = fit_own_average(peak_share(bank, layer))
        deviation = (peak_share(stats, layer) - mean) / spread
        columns_of[("peak share vs bank mean", layer)] = np.abs(deviation).mean(axis=1).reshape(shape)
        columns_of[("flatter vs bank mean", layer)] = (-deviation).mean(axis=1).reshape(shape)
    own = ctx.own_columns()
    for index, layer in enumerate(SCORED):
        columns_of[("level vs similar scenes", layer)] = own["level"][:, index].reshape(shape)
        columns_of[("peak share vs similar scenes", layer)] = own["shape"][:, index].reshape(shape)
        columns_of[("flatter vs similar scenes", layer)] = own["flatter"][:, index].reshape(shape)
    activations = cached("activations")
    for index, stage in enumerate(("C1 stem", "C2 res1", "C3 res2", "C4 res3", "C5 res4")):
        columns_of[("activation CDF", stage)] = np.asarray(activations["cdf_stages"][..., index])
    for index, stage in enumerate(("encoder map 1 (stride 8)", "encoder map 2 (stride 16)",
                                   "encoder map 3 (stride 32)")):
        columns_of[("Hashemi", stage)] = np.asarray(activations["hashemi_encoder_maps"][..., index])
    columns_of[("Hashemi", "decoder queries")] = np.asarray(activations["hashemi_decoder"])
    aurocs = {name: per_condition(values) for name, values in columns_of.items()}
    rows = table_rows(aurocs, lambda name: {"score": name[0], "stage": name[1]})
    write_csv("depth_curve", rows)
    for row in rows:
        log(f"depth: {row['score']:<30} {row['stage']:<26} {row['common']:.3f} / {row['extra']:.3f}  "
            f"sev1 {row['common_s1']:.3f}  fog1 {row['fog_s1']:.3f}  noise1 {row['gaussian_noise_s1']:.3f}")
    return {f"{r['score']} | {r['stage']}": {k: r[k] for k in ("common", "extra", "common_s1")} for r in rows}


def stability(ctx: Context) -> dict:
    """How the content key and the reference move under corruption."""
    neighbours = ctx.neighbours().reshape(IMAGES, VARIANTS, NEIGHBOURS)
    member = np.zeros((IMAGES, len(ctx.bank["means_s4"])), dtype=bool)
    np.put_along_axis(member, neighbours[:, 0].astype(np.int64), True, axis=1)
    overlap = np.take_along_axis(member[:, None, :], neighbours.astype(np.int64), axis=2).mean(axis=2)

    centre, spread, bank_keys = key_space(ctx.bank)
    keys = (np.asarray(ctx.stats["means_s4"], dtype=np.float64) - centre) / spread  # (images, 96, 512)
    moved = np.linalg.norm(keys - keys[:, :1], axis=2)
    clean_keys = keys[:, 0]
    distances = np.sqrt(np.maximum((clean_keys ** 2).sum(1)[:, None] + (bank_keys ** 2).sum(1)[None]
                                   - 2 * clean_keys @ bank_keys.T, 0))
    radius = np.sort(distances, axis=1)[:, NEIGHBOURS - 1]
    del keys

    own = ctx.own_columns()
    shifts = {name: own[name].reshape(IMAGES, VARIANTS, len(SCORED)) for name in
              ("nb_level", "img_level", "nb_share", "img_share")}
    shifts = {name: value - value[:, :1] for name, value in shifts.items()}
    rows = []
    for condition in range(1, VARIANTS):
        family, severity = CONDITIONS[condition]
        row = {"family": family, "severity": severity,
               "neighbours_kept": float(overlap[:, condition].mean()),
               "neighbours_kept_median": float(np.median(overlap[:, condition])),
               "key_move_over_radius_median": float(np.median(moved[:, condition] / radius))}
        for index, layer in enumerate(SCORED):
            for name in ("img_level", "nb_level", "img_share", "nb_share"):
                row[f"{name}_shift_{layer}"] = float(shifts[name][:, condition, index].mean())
        rows.append(row)
    write_csv("key_stability", rows)
    result = {}
    for row in rows:
        if row["severity"] in (1, 3, 5):
            result[f"{row['family']}_{row['severity']}"] = {k: row[k] for k in row if k not in ("family", "severity")}
    by_severity = {s: {"neighbours_kept_mean": float(np.mean([r["neighbours_kept"] for r in rows
                                                              if r["severity"] == s])),
                       "key_move_over_radius_median_mean": float(np.mean([r["key_move_over_radius_median"]
                                                                          for r in rows if r["severity"] == s]))}
                   for s in range(1, 6)}
    for s, values in by_severity.items():
        log(f"stability: severity {s}: neighbours kept {values['neighbours_kept_mean']:.3f}, key move / radius "
            f"{values['key_move_over_radius_median_mean']:.3f}")
    fog = next(r for r in rows if r["family"] == "fog" and r["severity"] == 1)
    log(f"stability: fog 1 at s1: image level {fog['img_level_shift_s1']:+.3f}, neighbours {fog['nb_level_shift_s1']:+.3f};"
        f" image share {fog['img_share_shift_s1']:+.3f}, neighbours {fog['nb_share_shift_s1']:+.3f}")
    return {"by_severity": by_severity, "conditions": result}


def _panoptic(path: Path) -> tuple[dict, dict]:
    with path.open() as handle:
        data = json.load(handle)
    things = {c["id"] for c in data["categories"] if c["isthing"]}
    names = {c["id"]: c["name"] for c in data["categories"]}
    present = {}
    for annotation in data["annotations"]:
        ids = [s["category_id"] for s in annotation["segments_info"]]
        present[annotation["image_id"]] = {"things": {i for i in ids if i in things},
                                           "stuff": {i for i in ids if i not in things},
                                           "count": sum(1 for i in ids if i in things)}
    return present, names


def categories(ctx: Context) -> dict:
    """Do the 50 neighbours show the same kind of scene? COCO panoptic labels, against 50 random bank images."""
    from scipy.stats import spearmanr

    annotations = Path("/home/yuchen/YuchenZ/Datasets/coco/annotations")
    log("categories: reading the panoptic annotations")
    val, names = _panoptic(annotations / "panoptic_val2017.json")
    train, _ = _panoptic(annotations / "panoptic_train2017.json")
    dataset = settings().dataset
    val_ids = [int(Path(n).stem) for n in (p.name for p in dataset.evaluation_images())]
    bank_ids = [int(p.stem) for p in dataset.reference_split("bank")]
    empty = {"things": set(), "stuff": set(), "count": 0}
    image_labels = [val.get(i, empty) for i in val_ids]
    bank_labels = [train.get(i, empty) for i in bank_ids]
    neighbours = ctx.neighbours().reshape(IMAGES, VARIANTS, NEIGHBOURS)[:, 0].astype(np.int64)
    random_rows = np.random.default_rng(SEED).integers(0, len(bank_ids), size=neighbours.shape)

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a | b else np.nan

    out = {}
    for label, rows in (("neighbours", neighbours), ("random", random_rows)):
        values = {kind: [] for kind in ("things", "stuff", "all")}
        counts = []
        for image, members in zip(image_labels, rows):
            for kind in ("things", "stuff"):
                values[kind].append(np.nanmean([jaccard(image[kind], bank_labels[j][kind]) for j in members]
                                               or [np.nan]))
            both = image["things"] | image["stuff"]
            values["all"].append(np.nanmean([jaccard(both, bank_labels[j]["things"] | bank_labels[j]["stuff"])
                                             for j in members]))
            counts.append(np.mean([bank_labels[j]["count"] for j in members]))
        out[label] = {f"jaccard_{kind}": float(np.nanmean(v)) for kind, v in values.items()}
        out[label]["object_count_spearman"] = float(spearmanr([x["count"] for x in image_labels], counts)[0])
    bank_rate = {}
    for labels in bank_labels:
        for c in labels["things"] | labels["stuff"]:
            bank_rate[c] = bank_rate.get(c, 0) + 1 / len(bank_labels)
    frequent = sorted(bank_rate, key=bank_rate.get, reverse=True)[:12]
    per_category = []
    for c in frequent:
        holders = [k for k, image in enumerate(image_labels) if c in image["things"] | image["stuff"]]
        others = [k for k, image in enumerate(image_labels) if c not in image["things"] | image["stuff"]]

        def rate(rows_of_images, table):
            return float(np.mean([[c in bank_labels[j]["things"] | bank_labels[j]["stuff"] for j in table[k]]
                                  for k in rows_of_images]))
        per_category.append({"category": names[c], "bank_rate": bank_rate[c],
                             "neighbours_if_present": rate(holders, neighbours),
                             "neighbours_if_absent": rate(others, neighbours), "images_with": len(holders)})
    write_csv("key_categories", per_category)
    out["per_category"] = per_category
    log(f"categories: Jaccard all {out['neighbours']['jaccard_all']:.3f} vs random {out['random']['jaccard_all']:.3f};"
        f" things {out['neighbours']['jaccard_things']:.3f} vs {out['random']['jaccard_things']:.3f}; stuff "
        f"{out['neighbours']['jaccard_stuff']:.3f} vs {out['random']['jaccard_stuff']:.3f}; object-count rho "
        f"{out['neighbours']['object_count_spearman']:.3f} vs {out['random']['object_count_spearman']:.3f}")
    for row in per_category:
        log(f"categories: {row['category']:<22} bank {row['bank_rate']:.2f}  neighbours if present "
            f"{row['neighbours_if_present']:.2f}, if absent {row['neighbours_if_absent']:.2f}")
    return out


def _residuals(values: dict, bank: dict, neighbours: np.ndarray, layer: str, statistic: str):
    reference_values = np.asarray(bank[f"means_{layer}"], dtype=np.float64) if statistic == "level" \
        else peak_share(bank, layer)
    own = flat(values[f"means_{layer}"]) if statistic == "level" else peak_share(values, layer)
    centre, spread = fit_own_average(reference_values)
    return own - reference_values[neighbours].mean(axis=1), own - centre, spread


def spread(ctx: Context) -> dict:
    """The clean spread around the neighbours' mean against around the bank's mean; shifts in both units (C3, C4)."""
    clean_val = {k: v[:, 0] for k, v in ctx.stats.items()}
    val_neighbours = ctx.neighbours().reshape(IMAGES, VARIANTS, NEIGHBOURS)[:, 0].astype(np.int64)
    zstats_neighbours = ctx.zstats_neighbours()
    rows, residual_spread = [], {}
    for statistic in ("level", "share"):
        for layer in SCORED:
            for label, values, neighbours in (("zstats", ctx.zstats, zstats_neighbours),
                                              ("val", clean_val, val_neighbours)):
                around_neighbours, around_mean, bank_spread = _residuals(values, ctx.bank, neighbours, layer,
                                                                         statistic)
                alive = around_mean.std(axis=0) > 0  # a channel silent on every image has no spread to compare
                ratio = around_neighbours.std(axis=0)[alive] / around_mean.std(axis=0)[alive]
                pooled = np.sqrt(((around_neighbours / bank_spread) ** 2).mean()
                                 / ((around_mean / bank_spread) ** 2).mean())
                rows.append({"statistic": statistic, "stage": layer, "images": label,
                             "sd_ratio_median": float(np.median(ratio)),
                             "sd_ratio_q25": float(np.quantile(ratio, 0.25)),
                             "sd_ratio_q75": float(np.quantile(ratio, 0.75)),
                             "variance_explained_median": float(np.median(1 - ratio ** 2)),
                             "rms_ratio_pooled": float(pooled)})
                if label == "zstats":
                    residual = around_neighbours.std(axis=0)  # floored like the bank's spread
                    residual_spread[(statistic, layer)] = np.maximum(
                        residual, STD_FLOOR * float(np.median(residual[residual > 0])))
    write_csv("spread_ratio", rows)
    for row in rows:
        log(f"spread: {row['statistic']:<5} {row['stage']} {row['images']:<6} sd ratio median "
            f"{row['sd_ratio_median']:.3f} [{row['sd_ratio_q25']:.3f}, {row['sd_ratio_q75']:.3f}], variance "
            f"explained {row['variance_explained_median']:.3f}, pooled rms ratio {row['rms_ratio_pooled']:.3f}")

    effects = []
    for statistic in ("level", "share"):
        for layer in SCORED:
            if statistic == "level":
                values = np.asarray(ctx.stats[f"means_{layer}"], dtype=np.float64)
                bank_spread = fit_own_average(ctx.bank[f"means_{layer}"])[1]
            else:
                values = peak_share(ctx.stats, layer).reshape(IMAGES, VARIANTS, -1)
                bank_spread = fit_own_average(peak_share(ctx.bank, layer))[1]
            change = values - values[:, :1]
            for condition in range(1, VARIANTS):
                family, severity = CONDITIONS[condition]
                in_bank = change[:, condition] / bank_spread
                in_residual = change[:, condition] / residual_spread[(statistic, layer)]
                effects.append({"statistic": statistic, "stage": layer, "family": family, "severity": severity,
                                "signed_in_bank_sd": float(in_bank.mean()),
                                "signed_in_residual_sd": float(in_residual.mean()),
                                "abs_in_bank_sd": float(np.abs(in_bank).mean()),
                                "abs_in_residual_sd": float(np.abs(in_residual).mean())})
    write_csv("effect_sizes", effects)
    for row in effects:
        if row["severity"] == 1 and row["stage"] == "s1" and row["family"] in NOTE_FAMILIES:
            log(f"effect: {row['statistic']:<5} s1 {row['family']:<15} |shift| {row['abs_in_bank_sd']:.3f} bank SD, "
                f"{row['abs_in_residual_sd']:.3f} residual SD")
    return {"spread": rows, "effects_s1": [r for r in effects if r["severity"] == 1]}


def flatten(ctx: Context) -> dict:
    """Flatten or shift: how each family moves each stage's level and peak share (C5, Figure 1)."""
    rows, uniform = [], {}
    for layer in STAGES:
        means = np.asarray(ctx.stats[f"means_{layer}"], dtype=np.float64)
        top = np.asarray(ctx.stats[f"top_{layer}"], dtype=np.float64)
        log_level, log_top = np.log(means + EPS), np.log(top + EPS)
        d_level, d_top = log_level - log_level[:, :1], log_top - log_top[:, :1]
        d_share = d_top - d_level
        share_spread = fit_own_average(peak_share(ctx.bank, layer))[1]
        bank_share = (peak_share(ctx.bank, layer) - peak_share(ctx.bank, layer).mean(axis=0)) / share_spread
        channels = bank_share.shape[1]
        direction = np.full(channels, 1 / np.sqrt(channels))
        uniform[layer] = float((bank_share @ direction).var() / bank_share.var(axis=0).sum())
        level_spread = fit_own_average(np.log(np.asarray(ctx.bank[f"means_{layer}"], dtype=np.float64) + EPS))[1]
        for condition in range(1, VARIANTS):
            family, severity = CONDITIONS[condition]
            z = d_share[:, condition] / share_spread
            along = (z @ direction) ** 2 / np.maximum((z ** 2).sum(axis=1), 1e-300)
            mean_shift = z.mean(axis=0)
            level_z = d_level[:, condition] / level_spread
            relative = (means[:, condition] - means[:, 0]) / (means[:, 0] + EPS)
            rows.append({"family": family, "severity": severity, "stage": layer,
                         "total_level_ratio": float(np.median(means[:, condition].sum(1) / means[:, 0].sum(1))),
                         "total_top_ratio": float(np.median(top[:, condition].sum(1) / top[:, 0].sum(1))),
                         "channels_up_10pct": float((relative > 0.1).mean()),
                         "channels_down_10pct": float((relative < -0.1).mean()),
                         "d_log_level": float(d_level[:, condition].mean()),
                         "d_log_top": float(d_top[:, condition].mean()),
                         "d_peak_share": float(d_share[:, condition].mean()),
                         "d_log_level_in_bank_sd": float(level_z.mean()),
                         "d_peak_share_in_bank_sd": float(z.mean()),
                         "channels_flatter": float((d_share[:, condition] < 0).mean()),
                         "channels_level_up": float((d_level[:, condition] > 0).mean()),
                         "shift_along_uniform_per_image": float(along.mean()),
                         "shift_along_uniform_of_mean": float((mean_shift @ direction) ** 2 / (mean_shift ** 2).sum()),
                         "top_over_level_ratio": float(d_top[:, condition].mean() / d_level[:, condition].mean())})
    write_csv("flatten_or_shift", rows)
    write_json("flatten_uniform_clean_share", uniform)
    log(f"flatten: clean variation along the all-channels direction: "
        + ", ".join(f"{k} {v:.3f}" for k, v in uniform.items()))
    for row in rows:
        if row["severity"] == 1 and row["stage"] == "s1":
            log(f"flatten: s1 {row['family']:<18} dlog level {row['d_log_level']:+.3f}  dlog top {row['d_log_top']:+.3f}"
                f"  dshare {row['d_peak_share']:+.3f}  flatter {row['channels_flatter']:.2f}  level up "
                f"{row['channels_level_up']:.2f}  along uniform {row['shift_along_uniform_per_image']:.2f}")
    return {"uniform_clean_share": uniform, "rows": rows}


def arms(ctx: Context) -> dict:
    """Which arm is larger per family, each arm's AUROC, and the larger arm against the sum (C6)."""
    rows = ctx.rows()
    flatter, level = rows["flatter_arm"], rows["level_arm"]
    aurocs = {name: per_condition(rows[name]) for name in ("two_axis", "flatter_arm", "level_arm", "sum_of_arms",
                                                            "peak_share")}
    table = family_rows(aurocs)
    for row in table:
        condition = next(c for c in range(1, VARIANTS) if CONDITIONS[c] == (row["family"], row["severity"]))
        row["flatter_arm_larger"] = float((flatter[:, condition] > level[:, condition]).mean())
    write_csv("arms_by_family", table)
    clean_share = float((flatter[:, 0] > level[:, 0]).mean())
    sets = image_sets(IMAGES)
    groups = {"common": np.flatnonzero(COMMON), "extra": np.flatnonzero(EXTRA)}
    separation = []
    for set_name in ("all", "untouched", "held_out"):
        images = sets[set_name]
        for name in ("two_axis", "sum_of_arms", "flatter_arm", "level_arm"):
            values = rows[name][images]
            for group, conditions in groups.items():
                auroc, aupr, fpr = group_separation(values[:, 0], values[:, conditions].T)
                separation.append({"images": set_name, "row": name, "group": group, "auroc": auroc, "aupr": aupr,
                                   "fpr95": fpr})
    write_csv("max_vs_sum", separation)
    intervals = {}
    for set_name in ("all", "untouched"):
        images = sets[set_name]
        maximum, total = rows["two_axis"][images], rows["sum_of_arms"][images]

        def statistic(draw, maximum=maximum, total=total):
            out = {}
            for group, conditions in groups.items():
                a = condition_aurocs(maximum[draw, 0], maximum[draw][:, conditions].T).mean()
                b = condition_aurocs(total[draw, 0], total[draw][:, conditions].T).mean()
                out[f"max_minus_sum_{group}"] = a - b
            return out
        point = statistic(np.arange(len(images)))
        interval = bootstrap(statistic, len(images), samples=1000, seed=SEED, workers=WORKERS)
        intervals[set_name] = {k: {"point": float(point[k]), "interval": interval[k]} for k in point}
        log(f"arms: {set_name}: max - sum " + "; ".join(
            f"{k.split('_')[-1]} {v['point']:+.4f} [{v['interval'][0]:+.4f}, {v['interval'][1]:+.4f}]"
            for k, v in intervals[set_name].items()))
    for row in separation:
        if row["images"] == "all":
            log(f"arms: all {row['row']:<12} {row['group']:<6} AUROC {row['auroc']:.4f} FPR95 {row['fpr95']:.3f}")
    larger = {}
    for family in FAMILIES:
        larger[family] = [r["flatter_arm_larger"] for r in table if r["family"] == family]
        log(f"arms: {family:<18} flatter arm larger in " + " ".join(f"{v:.2f}" for v in larger[family]) +
            f"   AUROC s1 flatter {next(r['flatter_arm'] for r in table if r['family'] == family and r['severity'] == 1):.3f}"
            f" level {next(r['level_arm'] for r in table if r['family'] == family and r['severity'] == 1):.3f}")
    log(f"arms: clean images with the flatter arm larger: {clean_share:.3f}")
    return {"clean_flatter_larger": clean_share, "intervals": intervals, "separation": separation,
            "flatter_larger_by_family": larger}


def twin(ctx: Context) -> dict:
    """The scores with the clean twin's neighbours: no key drift (C4), and what the features alone miss (C8)."""
    neighbours = ctx.neighbours().reshape(IMAGES, VARIANTS, NEIGHBOURS).astype(np.int64)
    twin_neighbours = np.broadcast_to(neighbours[:, :1], neighbours.shape).reshape(-1, NEIGHBOURS)
    twin_columns = columns(ctx.stats, ctx.bank, np.ascontiguousarray(twin_neighbours))
    clean_columns = ctx.clean_columns()
    own_rows, twin_rows = ctx.rows(), rows_of(twin_columns, clean_columns)
    aurocs = {}
    for name in ("two_axis", "level_arm", "flatter_arm", "peak_share"):
        aurocs[f"{name}_own"] = per_condition(own_rows[name])
        aurocs[f"{name}_twin"] = per_condition(twin_rows[name])
    write_csv("twin_key", family_rows(aurocs))
    summary = {name: summarise(values) for name, values in aurocs.items()}
    for name, values in summary.items():
        log(f"twin: {name:<18} {values['common']:.4f} / {values['extra']:.4f}  sev1 {values['common_s1']:.4f}")
    for family in NOTE_FAMILIES:
        log(f"twin: {family:<15} sev1 two-axis own {aurocs['two_axis_own'][(FAMILY == family) & (SEVERITY == 1)][0]:.3f}"
            f" twin {aurocs['two_axis_twin'][(FAMILY == family) & (SEVERITY == 1)][0]:.3f}; level own "
            f"{aurocs['level_arm_own'][(FAMILY == family) & (SEVERITY == 1)][0]:.3f} twin "
            f"{aurocs['level_arm_twin'][(FAMILY == family) & (SEVERITY == 1)][0]:.3f}")
    return summary


def severity(ctx: Context) -> dict:
    """The ladder by severity: global level -> level -> peak share -> two-axis, with paired intervals (C4)."""
    rows = ctx.rows()
    stats, bank, zstats = ctx.stats, ctx.bank, ctx.zstats
    scores = {"global_level": method_scores.global_level_scores(stats, bank, zstats)[0],
              "level": rows["level_arm"], "peak_share": rows["peak_share"], "two_axis": rows["two_axis"]}
    masks = {(group, s): np.flatnonzero(mask & (SEVERITY == s))
             for group, mask in (("common", COMMON), ("extra", EXTRA)) for s in range(1, 6)}
    pairs = (("level", "global_level"), ("peak_share", "level"), ("two_axis", "level"), ("two_axis", "global_level"))

    def statistic(draw):
        aurocs = {name: np.concatenate([[np.nan], condition_aurocs(v[draw, 0], v[draw, 1:].T)])
                  for name, v in scores.items()}
        out = {}
        for (group, s), conditions in masks.items():
            for a, b in pairs:
                out[f"{a}-{b}|{group}|{s}"] = float(aurocs[a][conditions].mean() - aurocs[b][conditions].mean())
        return out

    point_aurocs = {name: per_condition(v) for name, v in scores.items()}
    point = statistic(np.arange(IMAGES))
    interval = bootstrap(statistic, IMAGES, samples=1000, seed=SEED, workers=WORKERS)
    table = []
    for (group, s), conditions in masks.items():
        row = {"group": group, "severity": s}
        row.update({name: float(values[conditions].mean()) for name, values in point_aurocs.items()})
        for a, b in pairs:
            key = f"{a}-{b}|{group}|{s}"
            row[f"{a}-{b}"] = point[key]
            row[f"{a}-{b}_low"], row[f"{a}-{b}_high"] = interval[key]
        table.append(row)
    write_csv("ladder_by_severity", table)
    for row in table:
        log(f"severity: {row['group']:<6} {row['severity']}  global {row['global_level']:.3f} level {row['level']:.3f}"
            f" share {row['peak_share']:.3f} two-axis {row['two_axis']:.3f}  level-global {row['level-global_level']:+.3f}"
            f" [{row['level-global_level_low']:+.3f}, {row['level-global_level_high']:+.3f}]")
    return {"rows": table}


_ABLATION = {}


def _ablation_run(task):
    """One bank subset at k = 50 (forked worker): the AUROC summaries of the two-axis score, the level and the share."""
    from threadpoolctl import threadpool_limits

    size, seed = task
    threadpool_limits(4)  # four workers share the 16 cores
    stats, bank, zstats = _ABLATION["stats"], _ABLATION["bank"], _ABLATION["zstats"]
    if size < len(bank["means_s4"]):
        keep = np.sort(np.random.default_rng(seed).choice(len(bank["means_s4"]), size, replace=False))
        bank = {k: v[keep] for k, v in bank.items()}
    k = min(NEIGHBOURS, size)
    test_columns = columns(stats, bank, find_neighbours(stats, bank, k))
    clean_columns = columns(zstats, bank, find_neighbours(zstats, bank, k))
    rows = rows_of(test_columns, clean_columns)
    return {"bank": size, "seed": seed, "k": k,
            **{f"{name}_{group}": summarise(per_condition(rows[name]))[group]
               for name in ("two_axis", "level_arm", "peak_share") for group in ("common", "extra", "common_s1")}}


def ablation(ctx: Context) -> dict:
    """k from 1 to 500 on the 2,000-image bank (one sorted search), and bank sizes 100-2,000 at k = 50 (C4)."""
    ks = (1, 2, 5, 10, 20, 50, 100, 200, 500)
    largest = max(ks)
    stats, bank, zstats = ctx.stats, ctx.bank, ctx.zstats
    log("ablation: sorted neighbours for k up to 500")
    test_sorted = ctx.derived("neighbours_sorted_k500", lambda: sorted_neighbours(stats, bank, largest).astype(np.int32))
    clean_sorted = sorted_neighbours(zstats, bank, largest)
    k_rows = []
    for k in ks:
        test_columns = columns(stats, bank, np.ascontiguousarray(test_sorted[:, :k]).astype(np.int64))
        clean_columns = columns(zstats, bank, np.ascontiguousarray(clean_sorted[:, :k]))
        rows = rows_of(test_columns, clean_columns)
        row = {"k": k, **{f"{name}_{group}": summarise(per_condition(rows[name]))[group]
                          for name in ("two_axis", "level_arm", "peak_share") for group in ("common", "extra",
                                                                                            "common_s1")}}
        k_rows.append(row)
        log(f"ablation: k {k:>3}: two-axis {row['two_axis_common']:.4f} / {row['two_axis_extra']:.4f}, level "
            f"{row['level_arm_common']:.4f} / {row['level_arm_extra']:.4f}, share {row['peak_share_common']:.4f} / "
            f"{row['peak_share_extra']:.4f}")
    write_csv("ablation_k", k_rows)
    tasks = [(size, seed) for size in (100, 250, 500, 1000) for seed in range(3)] + [(2000, 0)]
    _ABLATION.update(stats={k: np.asarray(v) for k, v in stats.items()}, bank=bank, zstats=zstats)
    with multiprocessing.get_context("fork").Pool(4) as pool:
        bank_rows = pool.map(_ablation_run, tasks)
    _ABLATION.clear()
    write_csv("ablation_bank", bank_rows)
    for row in bank_rows:
        log(f"ablation: bank {row['bank']:>4} seed {row['seed']}: two-axis {row['two_axis_common']:.4f} / "
            f"{row['two_axis_extra']:.4f}, level {row['level_arm_common']:.4f} / {row['level_arm_extra']:.4f}")
    return {"k": k_rows, "bank": bank_rows}


def _top(values: dict, layer: str) -> np.ndarray:
    return flat(values[f"top_{layer}"])


def _raw_ratio(values: dict, layer: str) -> np.ndarray:
    return (flat(values[f"top_{layer}"]) + EPS) / (flat(values[f"means_{layer}"]) + EPS)


def _log_top(values: dict, layer: str) -> np.ndarray:
    return np.log(flat(values[f"top_{layer}"]) + EPS)


SHARE_VARIANTS = {"log ratio (ours)": peak_share, "raw ratio": _raw_ratio, "log top-1% mean": _log_top,
                  "top-1% mean": _top}


def statistic(ctx: Context) -> dict:
    """Why the peak share is the log of a ratio (C5): the second axis with other per-channel statistics.

    Each variant replaces the peak share in both the |deviation| row and the signed flattening arm; the level arm
    stays, so the two-axis column shows what each variant adds to it.
    """
    neighbours = ctx.neighbours().astype(np.int64)
    zstats_neighbours = ctx.zstats_neighbours()
    out = []
    for name, share in SHARE_VARIANTS.items():
        test_columns = columns(ctx.stats, ctx.bank, neighbours, share=share)
        clean_columns = columns(ctx.zstats, ctx.bank, zstats_neighbours, share=share)
        rows = rows_of(test_columns, clean_columns)
        row = {"statistic": name}
        for score in ("peak_share", "flatter_arm", "two_axis"):
            summary = summarise(per_condition(rows[score]))
            row.update({f"{score}_{k}": v for k, v in summary.items()})
        out.append(row)
        log(f"statistic: {name:<18} |dev| row {row['peak_share_common']:.4f} / {row['peak_share_extra']:.4f}; "
            f"flatter arm {row['flatter_arm_common']:.4f} / {row['flatter_arm_extra']:.4f}; two-axis "
            f"{row['two_axis_common']:.4f} / {row['two_axis_extra']:.4f} (sev1 {row['two_axis_common_s1']:.4f})")
    write_csv("second_axis_statistic", out)
    return {"rows": out}


def threshold(ctx: Context) -> dict:
    """A fixed threshold set on clean train images only: the false-alarm rate on clean val images, and the share of
    corrupted images flagged, per group and severity (the deployment view of the two-axis score)."""
    clean_columns = ctx.clean_columns()
    flatter = standardised(clean_columns["flatter"], clean_columns["flatter"])
    level = standardised(clean_columns["level"], clean_columns["level"])
    reference_scores = np.maximum(flatter, level)  # the 500 clean z-statistics train images
    scores = ctx.rows()["two_axis"]
    out = []
    for alpha in (0.10, 0.05, 0.01):
        cut = float(np.quantile(reference_scores, 1 - alpha))
        row = {"alpha": alpha, "threshold": cut, "false_alarms_clean_val": float((scores[:, 0] > cut).mean())}
        for group, mask in (("common", COMMON), ("extra", EXTRA)):
            for severity in range(1, 6):
                conditions = np.flatnonzero(mask & (SEVERITY == severity))
                row[f"flagged_{group}_s{severity}"] = float((scores[:, conditions] > cut).mean())
        out.append(row)
        log(f"threshold: alpha {alpha:.2f}: cut {cut:.3f}, false alarms on clean val {row['false_alarms_clean_val']:.3f};"
            f" flagged common s1 {row['flagged_common_s1']:.3f} s3 {row['flagged_common_s3']:.3f} s5 "
            f"{row['flagged_common_s5']:.3f}; extra s1 {row['flagged_extra_s1']:.3f} s5 {row['flagged_extra_s5']:.3f}")
    write_csv("fixed_threshold", out)
    return {"rows": out}


def opposite(ctx: Context) -> dict:
    """Do dense channels flatten while sparse ones sharpen? Per channel at severity 1: how peaky it is on clean
    images against how often it gets flatter, for the flattening families ("opposite moves", a claim to check)."""
    from scipy.stats import spearmanr

    out = []
    for family in ("fog", "contrast", "zoom_blur", "defocus_blur", "gaussian_blur"):
        condition = CONDITIONS.index((family, 1))
        for layer in SCORED:
            means = np.asarray(ctx.stats[f"means_{layer}"][:, [0, condition]], dtype=np.float64)
            top = np.asarray(ctx.stats[f"top_{layer}"][:, [0, condition]], dtype=np.float64)
            share = np.log(top + EPS) - np.log(means + EPS)
            clean_share = share[:, 0].mean(axis=0)
            flattens = (share[:, 1] < share[:, 0]).mean(axis=0)
            low, high = np.quantile(clean_share, [1 / 3, 2 / 3])
            out.append({"family": family, "stage": layer,
                        "spearman_peakiness_vs_flattening": float(spearmanr(clean_share, flattens)[0]),
                        "densest_third_flatten": float(flattens[clean_share <= low].mean()),
                        "peakiest_third_flatten": float(flattens[clean_share >= high].mean()),
                        "channels_sharpening_on_most_images": int((flattens < 0.5).sum()),
                        "channels": int(len(flattens))})
            row = out[-1]
            log(f"opposite: {family:<13} {layer}: rho {row['spearman_peakiness_vs_flattening']:+.2f}; densest third "
                f"flatten on {row['densest_third_flatten']:.2f}, peakiest third {row['peakiest_third_flatten']:.2f}; "
                f"{row['channels_sharpening_on_most_images']} of {row['channels']} sharpen on most images")
    write_csv("opposite_moves", out)
    return {"rows": out}


RIDGE = 10.0  # the penalty of the linear map the dev log compared with the neighbours (1 October)


def linear(ctx: Context) -> dict:
    """Why neighbours and not a linear map from the key (C4): the map's prediction follows the corrupted key.

    A ridge map from the standardised key predicts each stage's channel means in bank-spread units. The level score
    is then the mean |residual| per stage, z-scored on the z-statistics images and summed, from the image's own key or
    from its clean twin's. Each row also records how far the prediction moves under the corruption.
    """
    stats, bank, zstats = ctx.stats, ctx.bank, ctx.zstats
    centre, spread, bank_keys = key_space(bank)

    def design(keys):
        return np.c_[keys, np.ones(len(keys))]
    x = design(bank_keys)
    penalty = RIDGE * np.diag(np.r_[np.ones(bank_keys.shape[1]), 0.0])
    keys = (flat(stats["means_s4"]) - centre) / spread
    clean_keys = keys.reshape(IMAGES, VARIANTS, -1)[:, 0]
    zstats_keys = (np.asarray(zstats["means_s4"], dtype=np.float64) - centre) / spread
    image_of_row = np.arange(len(keys)) // VARIANTS
    test = {"own": np.empty((len(keys), len(SCORED))), "twin": np.empty((len(keys), len(SCORED)))}
    clean = np.empty((len(zstats_keys), len(SCORED)))
    moved = {name: np.empty((len(keys), len(SCORED))) for name in ("ridge", "image")}
    r2 = {}
    for column, layer in enumerate(SCORED):
        mean, layer_spread = fit_own_average(bank[f"means_{layer}"])
        target = (np.asarray(bank[f"means_{layer}"], dtype=np.float64) - mean) / layer_spread
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ target)
        values = (flat(stats[f"means_{layer}"]) - mean) / layer_spread
        clean_prediction = design(clean_keys) @ weights
        for start in range(0, len(keys), CHUNK):
            rows = slice(start, start + CHUNK)
            own = design(keys[rows]) @ weights
            twin_prediction = clean_prediction[image_of_row[rows]]
            test["own"][rows, column] = np.abs(values[rows] - own).mean(axis=1)
            test["twin"][rows, column] = np.abs(values[rows] - twin_prediction).mean(axis=1)
            moved["ridge"][rows, column] = (own - twin_prediction).mean(axis=1)
            moved["image"][rows, column] = (values[rows] - values.reshape(IMAGES, VARIANTS, -1)[:, 0][
                image_of_row[rows]]).mean(axis=1)
        zstats_values = (np.asarray(zstats[f"means_{layer}"], dtype=np.float64) - mean) / layer_spread
        residual = zstats_values - design(zstats_keys) @ weights
        clean[:, column] = np.abs(residual).mean(axis=1)
        r2[layer] = float(1 - (residual ** 2).sum() / ((zstats_values - zstats_values.mean(0)) ** 2).sum())
    aurocs = {f"ridge_{name}": per_condition(standardised(test[name], clean).reshape(IMAGES, VARIANTS))
              for name in test}
    summary = {name: summarise(values) for name, values in aurocs.items()}
    own = ctx.own_columns()
    knn_moved = own["nb_level"].reshape(IMAGES, VARIANTS, -1)
    knn_moved = (knn_moved - knn_moved[:, :1]).reshape(len(keys), -1)
    rows_out = []
    for condition in range(1, VARIANTS):
        family, severity = CONDITIONS[condition]
        take = np.arange(condition, len(keys), VARIANTS)
        row = {"family": family, "severity": severity,
               **{f"auroc_{name}": float(values[condition]) for name, values in aurocs.items()}}
        for column, layer in enumerate(SCORED):
            row[f"image_shift_{layer}"] = float(moved["image"][take, column].mean())
            row[f"ridge_shift_{layer}"] = float(moved["ridge"][take, column].mean())
            row[f"knn_shift_{layer}"] = float(knn_moved[take, column].mean())
        rows_out.append(row)
    write_csv("linear_vs_neighbours", rows_out)
    log("linear: held-out ridge R^2 " + ", ".join(f"{k} {v:.3f}" for k, v in r2.items()))
    for name, values in summary.items():
        log(f"linear: {name:<11} {values['common']:.4f} / {values['extra']:.4f}  sev1 {values['common_s1']:.4f}")
    for row in rows_out:
        if row["severity"] == 1 and row["family"] in NOTE_FAMILIES:
            log(f"linear: {row['family']:<15} s1 image {row['image_shift_s1']:+.3f}, ridge prediction "
                f"{row['ridge_shift_s1']:+.3f}, neighbours {row['knn_shift_s1']:+.3f} (bank SD, signed channel mean)")
    return {"r2": r2, "summary": summary, "rows": rows_out}


ANALYSES = {"check": check, "depth": depth, "stability": stability, "categories": categories, "spread": spread,
            "flatten": flatten, "arms": arms, "twin": twin, "severity": severity, "ablation": ablation, "linear": linear,
            "statistic": statistic, "threshold": threshold, "opposite": opposite}


def main(names) -> None:
    ctx = Context()
    results_path = ROOT / "docs" / "results" / "paper-evidence" / "stored.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in names or ANALYSES:
        started = time.time()
        log(f"--- {name}")
        results[name] = ANALYSES[name](ctx)
        write_json("stored", results)
        log(f"--- {name} done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
