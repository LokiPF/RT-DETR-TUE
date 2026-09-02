"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import datetime
import json
import os
import time
from pathlib import Path

from ..misc import dist_utils
from ._solver import BaseSolver
from .clas_engine import evaluate, train_one_epoch


def save_checkpoint(state, checkpoint_path):
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    try:
        dist_utils.save_on_master(state, temporary_path)

        if dist_utils.is_main_process():
            os.replace(temporary_path, checkpoint_path)

    except Exception:
        if dist_utils.is_main_process() and temporary_path.exists():
            temporary_path.unlink()
        raise


class ClasSolver(BaseSolver):
    def fit(
        self,
    ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Number of params:", n_parameters)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        best_acc = -1.0

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.ema,
                epoch=epoch,
                device=self.device,
            )
            self.lr_scheduler.step()
            self.last_epoch += 1

            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(
                module, self.criterion, self.val_dataloader, self.device
            )

            if output_dir:
                checkpoint_paths = [output_dir / "checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if args.checkpoint_freq > 0 and (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    state = self.state_dict()
                    state["last_epoch"] = epoch
                    save_checkpoint(state, checkpoint_path)
                if test_stats["acc"] > best_acc:
                    checkpoint_path = output_dir / "best.pth"
                    state = self.state_dict()
                    state["last_epoch"] = epoch
                    save_checkpoint(state, checkpoint_path)
                    best_acc = test_stats["acc"]

            print(f"Best acc: {best_acc}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if output_dir and dist_utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")
