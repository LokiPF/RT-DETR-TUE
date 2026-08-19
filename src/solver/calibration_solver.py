"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import datetime
from pathlib import Path

import torch

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .tue_engine import evaluate, collect_persistence_one_epoch
from .calibration_engine import collect_conformal_distances_one_epoch
from .tue_split import load_split, subset_dataloader

from supervisely.nn.training import train_logger


class CalibrationSolver(BaseSolver):

    def _create_frechet_mean(self, buckets):
        # Store means only for nonempty layer/class buckets.
        frechet_means = {}

        for layer_id in range(buckets.num_layers):
            layer_means = {}

            for class_id in range(buckets.num_classes):
                if buckets.count(layer_id, class_id) == 0:
                    continue

                layer_means[class_id] = buckets.frechet_mean(
                    layer_id=layer_id,
                    class_id=class_id,
                )

            frechet_means[layer_id] = layer_means

        return frechet_means

    def fit(self):
        print("Starting persistence analysis")
        self.train()  # this is necessary to load the train dataloader
        # self.eval() Not required because reinits the model

        args = self.cfg
        start_time = time.time()

        data_loader = self.train_dataloader

        split_path = args.yaml_cfg.get("train_calibration_split_path")
        if split_path is not None:
            split = load_split(split_path)
            data_loader = subset_dataloader(self.train_dataloader, split["frechet_indices"])
            print(
                f"Fitting Fréchet means on the frechet partition "
                f"({len(split['frechet_indices'])} images from {split_path})."
            )
        if hasattr(data_loader, "set_epoch"):
            data_loader.set_epoch(0)

        if (
                dist_utils.is_dist_available_and_initialized()
                and hasattr(data_loader, "sampler")
                and hasattr(data_loader.sampler, "set_epoch")
        ):
            data_loader.sampler.set_epoch(0)

        model = self.ema.module if self.ema is not None else self.model

        if hasattr(model, "module"):
            model = model.module

        confidence_threshold = args.yaml_cfg.get(
            "persistence_confidence_threshold", 0.5
        )
        decoder_layers = args.yaml_cfg.get("persistence_decoder_layers", None)
        class_ids = args.yaml_cfg.get("calibration_class_ids")
        if class_ids is not None:
            class_ids = sorted({int(class_id) for class_id in class_ids})
            print(f"Restricting Fréchet means to class IDs: {class_ids}")

        buckets_score, buckets_bbox, persistence_stats = collect_persistence_one_epoch(
            model=model,
            matcher=self.criterion.matcher,
            data_loader=data_loader,
            device=self.device,
            epoch=0,
            confidence_threshold=confidence_threshold,
            decoder_layers=decoder_layers,
            class_ids=class_ids,
            print_freq=args.yaml_cfg.get("print_freq", 10),
            data_fraction=1.0
        )

        frechet_means_score = self._create_frechet_mean(
            buckets_score
        )

        frechet_means_bbox = {
            bbox_layer_id: self._create_frechet_mean(bucket)
            for bbox_layer_id, bucket in buckets_bbox.items()
        }

        persistence_state = {
            "frechet_means_score": frechet_means_score,
            "frechet_means_bbox": frechet_means_bbox,
            "counts": {
                "score": buckets_score.counts,
                "bbox": {
                    bbox_layer_id: bucket.counts
                    for bbox_layer_id, bucket in buckets_bbox.items()
                },
            },
            "metadata": persistence_stats,
            "confidence_threshold": confidence_threshold,
            "decoder_layers": decoder_layers,
            "class_ids": class_ids,
            "bbox_head_layers": sorted(buckets_bbox),
        }

        if self.output_dir and dist_utils.is_main_process():
            self.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = self.output_dir / "frechet_means.pth"
            torch.save(persistence_state, output_path)

            print(f"Saved Fréchet means to {output_path}")

        elapsed = time.time() - start_time
        elapsed_string = str(
            datetime.timedelta(seconds=int(elapsed))
        )

        print(f"Persistence analysis time: {elapsed_string}")

        return persistence_state

    def fit_conformal(self):
        print("Starting conformal distance calibration")
        self.train()  # loads model, criterion (matcher), and dataloaders
        args = self.cfg
        start_time = time.time()

        # ---- Load the Frechet means produced by fit(). ----
        frechet_path = args.yaml_cfg.get("frechet_means_path")
        if frechet_path is None:
            frechet_path = self.output_dir / "frechet_means.pth"
        frechet_path = Path(frechet_path)
        if not frechet_path.exists():
            raise FileNotFoundError(
                f"Frechet means not found at {frechet_path}; run fit() first "
                "or set cfg.frechet_means_path."
            )

        persistence_state = torch.load(
            frechet_path,
            map_location="cpu",
            weights_only=False,
        )
        frechet_means_score = persistence_state["frechet_means_score"]
        frechet_means_bbox = persistence_state["frechet_means_bbox"]

        # ---- Calibration split: must be held out from the fitting data. ----
        split_path = args.yaml_cfg.get("train_calibration_split_path")
        if split_path is not None:
            split = load_split(split_path)
            calibration_loader = subset_dataloader(
                self.train_dataloader, split["calibration_indices"]
            )
            print(
                f"Conformal calibration on the calibration partition "
                f"({len(split['calibration_indices'])} images from {split_path})."
            )
        else:
            calibration_loader = getattr(self, "val_dataloader", None)
            if calibration_loader is None:
                print(
                    "WARNING: no train/calibration split and no val_dataloader; falling back "
                    "to the train dataloader. Conformal calibration is only valid on data "
                    "disjoint from the Frechet-mean fitting split."
                )
                calibration_loader = self.train_dataloader
            else:
                print(
                    "No train_calibration_split_path set; calibrating on the val split. "
                    "For a train-only split, generate one with tools/split_train_calibration.py."
                )

        if hasattr(calibration_loader, "set_epoch"):
            calibration_loader.set_epoch(0)
        if (
            dist_utils.is_dist_available_and_initialized()
            and hasattr(calibration_loader, "sampler")
            and hasattr(calibration_loader.sampler, "set_epoch")
        ):
            calibration_loader.sampler.set_epoch(0)

        model = self.ema.module if self.ema is not None else self.model
        if hasattr(model, "module"):
            model = model.module

        confidence_threshold = args.yaml_cfg.get("conformal_confidence_threshold")
        if confidence_threshold is None:
            confidence_threshold = persistence_state.get(
                "confidence_threshold",
                args.yaml_cfg.get("persistence_confidence_threshold", 0.5),
            )
        decoder_layers = persistence_state.get(
            "decoder_layers",
            args.yaml_cfg.get("persistence_decoder_layers", None),
        )
        fitted_class_ids = persistence_state.get("class_ids")
        requested_class_ids = args.yaml_cfg.get("calibration_class_ids")
        if requested_class_ids is None:
            class_ids = fitted_class_ids
        else:
            class_ids = sorted({int(class_id) for class_id in requested_class_ids})
            if fitted_class_ids is not None:
                missing = sorted(set(class_ids) - set(map(int, fitted_class_ids)))
                if missing:
                    raise ValueError(
                        "Conformal calibration requested classes without fitted "
                        f"Fréchet means: {missing}"
                    )
        print(
            f"Conformal confidence threshold: {confidence_threshold:g} "
            f"(Fréchet metadata: {persistence_state.get('confidence_threshold', 'missing')})"
        )
        if class_ids is not None:
            print(f"Restricting conformal calibration to class IDs: {class_ids}")

        calibration_population = args.yaml_cfg.get(
            "conformal_calibration_population", "correct"
        )

        distances_score, distances_bbox, conformal_stats = (
            collect_conformal_distances_one_epoch(
                model=model,
                matcher=self.criterion.matcher,
                data_loader=calibration_loader,
                device=self.device,
                frechet_means_score=frechet_means_score,
                frechet_means_bbox=frechet_means_bbox,
                epoch=0,
                confidence_threshold=confidence_threshold,
                decoder_layers=decoder_layers,
                class_ids=class_ids,
                print_freq=args.yaml_cfg.get("print_freq", 10),
                data_fraction=args.yaml_cfg.get("calibration_data_fraction", 1.0),
                calibration_population=calibration_population,
            )
        )

        conformal_state = {
            "conformal_distances_score": distances_score.to_sorted_arrays(),
            "conformal_distances_bbox": {
                bbox_layer_id: collector.to_sorted_arrays()
                for bbox_layer_id, collector in distances_bbox.items()
            },
            "counts": {
                "score": distances_score.counts,
                "bbox": {
                    bbox_layer_id: collector.counts
                    for bbox_layer_id, collector in distances_bbox.items()
                },
            },
            "metadata": conformal_stats,
            "confidence_threshold": confidence_threshold,
            "decoder_layers": decoder_layers,
            "class_ids": class_ids,
            "calibration_population": calibration_population,
            "frechet_means_path": str(frechet_path),
        }

        if self.output_dir and dist_utils.is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "conformal_distances.pth"
            torch.save(conformal_state, output_path)
            print(f"Saved conformal distances to {output_path}")

        elapsed_string = str(
            datetime.timedelta(seconds=int(time.time() - start_time))
        )
        print(f"Conformal calibration time: {elapsed_string}")

        return conformal_state

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    def _strip_state_dict(self, state_dict):
        if not self.cfg.yaml_cfg['save_optimizer'] and "optimizer" in state_dict:
            state_dict.pop("optimizer")
        if not self.cfg.yaml_cfg['save_ema'] and "ema" in state_dict:
            state_dict.pop("model")  # keep ema as a model
