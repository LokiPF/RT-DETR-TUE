"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import datetime
import time

import torch

from ..misc import dist_utils
from ._solver import BaseSolver
from .tue_engine import collect_persistence_one_epoch, evaluate
from .tue_split import load_split, subset_dataloader


class TUESolver(BaseSolver):
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
            data_loader = subset_dataloader(
                self.train_dataloader,
                split["frechet_indices"],
            )
            print(
                "Fitting Fréchet means on the Fréchet partition "
                f"({len(split['frechet_indices'])} images from {split_path})."
            )

        # Makes distributed sampling deterministic.
        if hasattr(data_loader, "set_epoch"):
            data_loader.set_epoch(0)

        if (
            dist_utils.is_dist_available_and_initialized()
            and hasattr(data_loader, "sampler")
            and hasattr(data_loader.sampler, "set_epoch")
        ):
            data_loader.sampler.set_epoch(0)

        # Prefer EMA parameters when available.
        model = self.ema.module if self.ema is not None else self.model

        # Remove DistributedDataParallel wrapping.
        if hasattr(model, "module"):
            model = model.module

        confidence_threshold = args.yaml_cfg.get(
            "persistence_confidence_threshold", 0.5
        )
        decoder_layers = args.yaml_cfg.get("persistence_decoder_layers", None)

        buckets_score, buckets_bbox, persistence_stats = collect_persistence_one_epoch(
            model=model,
            matcher=self.criterion.matcher,
            data_loader=data_loader,
            device=self.device,
            epoch=0,
            confidence_threshold=confidence_threshold,
            decoder_layers=decoder_layers,
            print_freq=args.yaml_cfg.get("print_freq", 10),
            data_fraction=1.0,
        )

        frechet_means_score = self._create_frechet_mean(buckets_score)

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
        elapsed_string = str(datetime.timedelta(seconds=int(elapsed)))

        print(f"Persistence analysis time: {elapsed_string}")

        return persistence_state

    def val(
        self,
    ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

    def _strip_state_dict(self, state_dict):
        if not self.cfg.yaml_cfg["save_optimizer"] and "optimizer" in state_dict:
            state_dict.pop("optimizer")
        if not self.cfg.yaml_cfg["save_ema"] and "ema" in state_dict:
            state_dict.pop("model")  # keep ema as a model
