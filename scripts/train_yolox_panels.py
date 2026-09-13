#!/usr/bin/env python3
"""Train the fixed NicoComic YOLOX panel experiment on CPU or CUDA."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from yolox.exp import get_exp
from yolox.utils import ModelEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-mosaic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.batch_size < 1 or args.workers < 0:
        raise SystemExit("Batch size must be positive and workers cannot be negative")

    random.seed(20260914)
    np.random.seed(20260914)
    torch.manual_seed(20260914)

    exp = get_exp(str(args.experiment), None)
    exp.data_dir = str(args.data_dir)
    exp.data_num_workers = args.workers
    epochs = exp.max_epoch if args.epochs is None else args.epochs
    if epochs < 1:
        raise SystemExit("Epochs must be positive")
    exp.max_epoch = epochs

    loader = exp.get_data_loader(args.batch_size, False, args.no_mosaic, False)
    model = exp.get_model().train().to(args.device)
    optimizer = exp.get_optimizer(args.batch_size)
    scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * args.batch_size, len(loader))
    ema = ModelEMA(model)
    start_epoch = 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "latest_ckpt.pth"
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if "ema" in saved:
            ema.ema.load_state_dict(saved["ema"])
            ema.updates = saved.get("ema_updates", 0)
        start_epoch = saved["epoch"] + 1

    for epoch in range(start_epoch, epochs):
        if not args.no_mosaic and epoch >= epochs - min(exp.no_aug_epochs, epochs):
            loader.close_mosaic()
        started = time.perf_counter()
        losses = []
        for iteration, (images, targets, _, _) in zip(range(len(loader)), loader):
            images = images.to(args.device)
            targets = targets.to(args.device)
            images, targets = exp.preprocess(images, targets, exp.input_size)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images, targets)
            loss = outputs["total_loss"]
            if not torch.isfinite(loss):
                raise SystemExit(f"Non-finite loss at epoch {epoch + 1}, batch {iteration + 1}")
            loss.backward()
            optimizer.step()
            ema.update(model)

            progress = epoch * len(loader) + iteration + 1
            learning_rate = scheduler.update_lr(progress)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            losses.append(float(loss.detach().cpu()))

        record = {
            "epoch": epoch + 1,
            "loss": sum(losses) / len(losses),
            "seconds": time.perf_counter() - started,
            "learningRate": learning_rate,
        }
        print(json.dumps(record), flush=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.ema.state_dict(),
                "ema_updates": ema.updates,
                "optimizer": optimizer.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, checkpoint)


if __name__ == "__main__":
    main()
