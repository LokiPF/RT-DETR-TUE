"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .tue_engine import evaluate, collect_persistence_one_epoch

from supervisely.nn.training import train_logger


class TUESolver(BaseSolver):

    def fit(self):
        print("Starting persistence analysis")
        self.train() # this is necessary to load the train dataloader
        self.eval()

        args = self.cfg
        start_time = time.time()

        data_loader = self.train_dataloader

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

        buckets, persistence_stats = collect_persistence_one_epoch(
            model=model,
            data_loader=data_loader,
            device=self.device,
            epoch=0,
            confidence_threshold=getattr(
                args,
                "persistence_confidence_threshold",
                0.8,
            ),
            decoder_layers=getattr(
                args,
                "persistence_decoder_layers",
                None,
            ),
            print_freq=getattr(args, "print_freq", 10),
        )

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

        persistence_state = {
            "frechet_means": frechet_means,
            "counts": buckets.counts,
            "metadata": persistence_stats,
            "confidence_threshold": getattr(
                args,
                "persistence_confidence_threshold",
                0.8,
            ),
            "decoder_layers": getattr(
                args,
                "persistence_decoder_layers",
                None,
            ),
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