"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import datetime
import json
import time
from pathlib import Path

import torch
from supervisely.nn.training import train_logger

from ..misc import dist_utils
from ._solver import BaseSolver
from .det_engine import build_frechet_means, evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(
        self,
    ):
        print("Start training")
        self.train()
        args = self.cfg

        # # print("Freezing RT-DETR weights. Only training UncTemp head")
        # for name, param in self.model.named_parameters():
        #     if "dec_bbox_logvar_head" not in name:
        #         param.requires_grad = False

        n_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print(f"number of trainable parameters: {n_parameters}")

        best_stat = {
            "epoch": -1,
        }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        train_logger.train_started(total_epochs=(args.epoches - start_epcoch))
        for epoch in range(start_epcoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_logger.epoch_started(total_steps=len(self.train_dataloader))
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch + 1:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    state_dict = self.state_dict()
                    self._strip_state_dict(state_dict)
                    dist_utils.save_on_master(state_dict, checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat["epoch"] == epoch and self.output_dir:
                    state_dict = self.state_dict()
                    self._strip_state_dict(state_dict)
                    dist_utils.save_on_master(state_dict, self.output_dir / "best.pth")

            print(f"best_stat: {best_stat}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

            train_logger.epoch_finished()

        train_logger.train_finished()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

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
        pass
        # if not self.cfg.yaml_cfg['save_optimizer'] and "optimizer" in state_dict:
        #     state_dict.pop("optimizer")
        # if not self.cfg.yaml_cfg['save_ema'] and "ema" in state_dict:
        #     state_dict.pop("model")  # keep ema as a model

    def build_tue_frechet_means(self):
        self.eval()

        module = self.ema.module if self.ema else self.model

        state = build_frechet_means(
            model=module,
            criterion=self.criterion,
            data_loader=self.val_dataloader,
            device=self.device,
            min_confidence=getattr(
                self.cfg,
                "tue_calibration_confidence",
                0.5,
            ),
        )

        if dist_utils.is_main_process():
            output_path = Path(
                getattr(
                    self.cfg,
                    "frechet_means_output",
                    self.output_dir / "frechet_means.pth",
                )
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(state, output_path)
            print(f"Saved Fréchet means to {output_path}")
            print("Counts per layer/class:")
            print(state["counts"])
