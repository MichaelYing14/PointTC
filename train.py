"""Train or finetune one of the six released PointTC configurations."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pointtc.common import load_checkpoint, load_config, resolve_device, seed_everything, training_loss
from pointtc.data import ShapeNetDataset
from pointtc.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--pretrained", help="Initialize parameters; reset optimizer and epoch")
    initialization.add_argument("--resume", help="Resume a checkpoint created by this train.py")
    return parser.parse_args()


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    item = state["numpy"]
    np.random.set_state((item[0], np.asarray(item[1], dtype=np.uint32), *item[2:]))
    generator.set_state(state["loader"])


def average_batches(model, loader, config, device, snr, limit=None, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    samples = 0
    with torch.set_grad_enabled(training):
        for batch_index, (points, _) in enumerate(loader):
            if limit is not None and batch_index >= limit:
                break
            points = points.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(points, snr=snr)
            loss, cd, symbols = training_loss(outputs, config["method"], config["rate_weight"],
                                               num_symbols=config["num_symbols"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in batch {batch_index}")
            if training:
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError(f"Non-finite gradients in batch {batch_index}")
                optimizer.step()
            count = len(points)
            totals += np.asarray([loss.item(), cd.item(), symbols.item()]) * count
            samples += count
    if not samples:
        raise ValueError("No samples processed; check data, batch size, and batch limits")
    return dict(zip(("loss", "cd", "symbols"), (totals / samples).tolist()))


def main(args):
    config = load_config(args.config)
    for key in ("epochs", "batch_size"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for value in (config["epochs"], config["batch_size"], args.max_train_batches, args.max_val_batches):
        if value is not None and value <= 0:
            raise ValueError("Epoch, batch size and batch limits must be positive")
    if config["method"] == "ca_ra" and config["batch_size"] < 2:
        raise ValueError("PointTC+CA+RA training requires batch_size >= 2 for BatchNorm")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    out = Path(args.output_dir)
    if (out / "last.pt").exists() and not args.resume:
        raise FileExistsError("Output contains a training checkpoint; choose a new output or --resume")
    train_data = ShapeNetDataset(args.data_root, split="train")
    val_data = ShapeNetDataset(args.data_root, split="val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=args.workers, drop_last=config["method"] == "ca_ra",
                              generator=generator)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=args.workers)
    model = build_model(config["method"], config["channel"], num_symbols=config["num_symbols"]).to(device)
    payload = None
    if args.pretrained or args.resume:
        payload = load_checkpoint(model, args.pretrained or args.resume, config["method"], config["channel"],
                                  allow_ca_to_ra=bool(args.pretrained),
                                  allow_channel_transfer=bool(args.pretrained))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    start_epoch, best_loss = 0, math.inf
    if args.resume:
        needed = ("optimizer_state_dict", "scheduler_state_dict", "epoch", "best_val_loss", "rng_state")
        if any(key not in payload for key in needed):
            raise ValueError("Resume requires a full training checkpoint; use --pretrained for release weights")
        for key in ("train_snr", "val_snrs", "rate_weight", "learning_rate", "weight_decay", "batch_size"):
            if payload["config"].get(key) != config[key]:
                raise ValueError(f"Resume config differs in {key}; use --pretrained for a new run")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch, best_loss = payload["epoch"] + 1, payload["best_val_loss"]
        restore_rng(payload["rng_state"], generator)
    if start_epoch >= config["epochs"]:
        raise ValueError("Requested epochs already completed by resume checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "run.json").write_text(json.dumps({"seed": args.seed, "device": str(device),
        "torch": str(torch.__version__), "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches}, indent=2) + "\n")
    fields = ["epoch", "train_loss", "train_cd", "train_symbols", "val_loss", "val_cd", "val_symbols"]
    history = out / "history.csv"
    with history.open("a" if args.resume else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not args.resume or handle.tell() == 0:
            writer.writeheader()
        for epoch in range(start_epoch, config["epochs"]):
            train_values = average_batches(model, train_loader, config, device, config["train_snr"],
                                           args.max_train_batches, optimizer)
            val_values = [average_batches(model, val_loader, config, device, float(snr), args.max_val_batches)
                          for snr in config["val_snrs"]]
            val = {key: sum(item[key] for item in val_values) / len(val_values) for key in val_values[0]}
            scheduler.step()
            row = {"epoch": epoch + 1, **{"train_" + k: v for k, v in train_values.items()},
                   **{"val_" + k: v for k, v in val.items()}}
            writer.writerow(row)
            handle.flush()
            improved = val["loss"] < best_loss
            best_loss = min(best_loss, val["loss"])
            state = {"format_version": 1, "config": config, "epoch": epoch,
                     "best_val_loss": best_loss, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(), "rng_state": rng_state(generator)}
            temporary = out / "last.pt.tmp"
            torch.save(state, temporary)
            temporary.replace(out / "last.pt")
            if improved:
                torch.save(state, out / "best.pt")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(parse_args())
